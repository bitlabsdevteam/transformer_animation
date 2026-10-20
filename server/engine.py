"""GPT-2 (small) forward + backward engine that keeps every intermediate tensor.

The stage names mirror what the reverse-engineered Transformer Explainer bundle
exposes (embedding -> 12 x [LN, QKV, attention scores/scaled/masked/softmax, out,
residual, LN, MLP] -> final LN -> logits), extended with the gradient of a
next-token cross-entropy loss at every stage.
"""

from __future__ import annotations

import base64
import math
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_ID = "openai-community/gpt2"
MAX_TOKENS = 32
BINS = 64
TOP_N = 50

LAYER_STAGES = ["ln1", "q", "k", "v", "attn", "ctx", "attn_out", "resid1",
                "ln2", "fc", "gelu", "mlp_out", "resid2"]

STAGE_INFO = {
    "embed": ("Embedding", "Token embedding + positional embedding"),
    "ln1": ("LayerNorm 1", "Normalise each token vector, then scale (γ) and shift (β)"),
    "q": ("Query", "Q = LN(x)·W_q + b_q"),
    "k": ("Key", "K = LN(x)·W_k + b_k"),
    "v": ("Value", "V = LN(x)·W_v + b_v"),
    "attn": ("Attention", "softmax(mask(Q·Kᵀ/√d_k)) per head"),
    "ctx": ("Head outputs", "Σ_j P[t,j]·V[j], heads concatenated"),
    "attn_out": ("Attn projection", "concat(heads)·W_O + b_O"),
    "resid1": ("Residual 1", "x + attention output"),
    "ln2": ("LayerNorm 2", "Normalise before the MLP"),
    "fc": ("MLP expand", "LN(x)·W_fc + b_fc  (768 → 3072)"),
    "gelu": ("GELU", "0.5·z·(1 + tanh(√(2/π)(z + 0.044715z³)))"),
    "mlp_out": ("MLP project", "GELU(z)·W_proj + b_proj  (3072 → 768)"),
    "resid2": ("Residual 2", "x + MLP output"),
    "ln_f": ("Final LayerNorm", "Normalise the last hidden state"),
    "logits": ("Logits", "LN_f(x)·W_teᵀ (tied embedding)"),
}


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode()


def bin_max_abs(x: np.ndarray, bins: int = BINS) -> np.ndarray:
    """(T, D) -> (T, bins): the signed value with the largest magnitude in each bin."""
    t, d = x.shape
    g = x.reshape(t, bins, d // bins)
    idx = np.abs(g).argmax(axis=2)
    return np.take_along_axis(g, idx[..., None], axis=2)[..., 0]


@dataclass
class Run:
    ids: list[int]
    tokens: list[str]
    target: int
    loss: float
    acts: dict[str, torch.Tensor] = field(default_factory=dict)
    grads: dict[str, torch.Tensor] = field(default_factory=dict)


class Engine:
    def __init__(self) -> None:
        self.tok = GPT2TokenizerFast.from_pretrained(MODEL_ID)
        model = GPT2LMHeadModel.from_pretrained(MODEL_ID).float().eval()
        self.cfg = model.config
        self.n_layer, self.n_head, self.d = self.cfg.n_layer, self.cfg.n_head, self.cfg.n_embd
        self.d_head = self.d // self.n_head
        self.eps = self.cfg.layer_norm_epsilon
        # HF Conv1D stores weights as [in, out], so every projection is x @ W + b.
        self.W = {k.removeprefix("transformer."): v.detach().clone().requires_grad_(True)
                  for k, v in model.state_dict().items()
                  if not k.endswith(".attn.bias") and not k.endswith("masked_bias") and k != "lm_head.weight"}
        self.runs: OrderedDict[str, Run] = OrderedDict()
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ forward
    def _ln(self, x, prefix):
        return F.layer_norm(x, (self.d,), self.W[f"{prefix}.weight"], self.W[f"{prefix}.bias"], self.eps)

    def forward(self, ids: list[int], keep=None):
        """Pure forward pass. `keep(name, tensor)` records each intermediate."""
        keep = keep or (lambda _n, t: t)
        W, T, H, dh = self.W, len(ids), self.n_head, self.d_head
        idx = torch.tensor(ids)
        x = keep("embed", W["wte.weight"][idx] + W["wpe.weight"][torch.arange(T)])
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        for l in range(self.n_layer):
            p = f"h.{l}"
            h = keep(f"L{l}.ln1", self._ln(x, f"{p}.ln_1"))
            qkv = h @ W[f"{p}.attn.c_attn.weight"] + W[f"{p}.attn.c_attn.bias"]
            q, k, v = (keep(f"L{l}.{n}", t) for n, t in zip("qkv", qkv.split(self.d, dim=1)))
            qh, kh, vh = (t.view(T, H, dh).transpose(0, 1) for t in (q, k, v))
            scores = keep(f"L{l}.scores", qh @ kh.transpose(-1, -2))
            scaled = keep(f"L{l}.scaled", scores / math.sqrt(dh))
            masked = keep(f"L{l}.masked", scaled.masked_fill(causal, float("-inf")))
            probs = keep(f"L{l}.attn", torch.softmax(masked, dim=-1))  # dropout = identity at inference
            ctx = keep(f"L{l}.ctx", (probs @ vh).transpose(0, 1).reshape(T, self.d))
            a = keep(f"L{l}.attn_out", ctx @ W[f"{p}.attn.c_proj.weight"] + W[f"{p}.attn.c_proj.bias"])
            x = keep(f"L{l}.resid1", x + a)
            h2 = keep(f"L{l}.ln2", self._ln(x, f"{p}.ln_2"))
            fc = keep(f"L{l}.fc", h2 @ W[f"{p}.mlp.c_fc.weight"] + W[f"{p}.mlp.c_fc.bias"])
            g = keep(f"L{l}.gelu", gelu_new(fc))
            m = keep(f"L{l}.mlp_out", g @ W[f"{p}.mlp.c_proj.weight"] + W[f"{p}.mlp.c_proj.bias"])
            x = keep(f"L{l}.resid2", x + m)
        xf = keep("ln_f", self._ln(x, "ln_f"))
        return keep("logits", xf @ W["wte.weight"].T)

    # ----------------------------------------------------------------- run
    def tokenize(self, text: str) -> list[int]:
        ids = self.tok.encode(text if text.strip() else " ")
        return ids[-MAX_TOKENS:]

    def run(self, text: str, target: int | None = None) -> tuple[str, dict]:
        n_total = len(self.tok.encode(text if text.strip() else " "))
        ids = self.tokenize(text)
        with self.lock:
            for w in self.W.values():
                w.grad = None
            acts: dict[str, torch.Tensor] = {}

            def keep(name, t):
                t.retain_grad()
                acts[name] = t
                return t

            logits = self.forward(ids, keep)
            last = logits[-1]
            if target is None:
                target = int(last.argmax())
            loss = F.cross_entropy(last[None], torch.tensor([target]))
            loss.backward()
            run = Run(ids=ids, tokens=[self.tok.decode([i]) for i in ids], target=target,
                      loss=float(loss.detach()),
                      acts={k: v.detach() for k, v in acts.items()},
                      grads={k: (v.grad if v.grad is not None else torch.zeros_like(v)).detach()
                             for k, v in acts.items()})
            wnorms = {k: float(w.grad.norm()) for k, w in self.W.items() if w.grad is not None}
        run_id = uuid.uuid4().hex[:12]
        self.runs[run_id] = run
        while len(self.runs) > 16:
            self.runs.popitem(last=False)
        payload = self._payload(run_id, run, wnorms)
        payload["truncated_from"] = n_total if n_total > len(ids) else None
        return run_id, payload

    def _payload(self, run_id: str, run: Run, wnorms: dict) -> dict:
        stages = []

        def vec_stage(key, layer, name):
            val = run.acts[key].numpy()
            grad = run.grads[key].numpy()
            bv, bg = bin_max_abs(val), bin_max_abs(grad)
            stages.append({
                "key": key, "layer": layer, "name": name, "kind": "vec",
                "dim": int(val.shape[1]), "shape": [int(val.shape[0]), BINS],
                "val": b64(bv), "grad": b64(bg),
                "gnorm": np.linalg.norm(grad, axis=1).tolist(),
                "vscale": float(np.percentile(np.abs(bv), 99) or 1.0),
                "gscale": float(np.percentile(np.abs(bg), 99) or 1e-12),
            })

        vec_stage("embed", -1, "embed")
        for l in range(self.n_layer):
            for n in LAYER_STAGES:
                key = f"L{l}.{n}"
                if n == "attn":
                    p = run.acts[key].numpy()
                    gp = run.grads[key].numpy()
                    stages.append({
                        "key": key, "layer": l, "name": n, "kind": "attn",
                        "shape": list(p.shape), "val": b64(p), "grad": b64(gp),
                        "gnorm": np.linalg.norm(gp, axis=(0, 2)).tolist(),
                        "vscale": 1.0, "gscale": float(np.percentile(np.abs(gp), 99) or 1e-12),
                    })
                else:
                    vec_stage(key, l, n)
        vec_stage("ln_f", self.n_layer, "ln_f")

        last = run.acts["logits"][-1]
        glast = run.grads["logits"][-1]
        top = torch.topk(last, TOP_N).indices.tolist()
        if run.target not in top:
            top[-1] = run.target
        lv = last[top].numpy()
        gv = glast[top].numpy()
        stages.append({
            "key": "logits", "layer": self.n_layer + 1, "name": "logits", "kind": "logits",
            "shape": [1, len(top)], "val": b64(lv), "grad": b64(gv),
            "gnorm": [float(glast.norm())], "vscale": float(np.abs(lv).max()),
            "gscale": float(np.abs(gv).max() or 1e-12),
        })
        return {
            "run_id": run_id, "tokens": run.tokens, "ids": run.ids, "T": len(run.ids),
            "target": {"id": run.target, "token": self.tok.decode([run.target])},
            "loss": run.loss,
            "config": {"n_layer": self.n_layer, "n_head": self.n_head, "d_model": self.d,
                       "d_head": self.d_head, "bins": BINS, "vocab": self.cfg.vocab_size},
            "top": [{"id": i, "token": self.tok.decode([i]), "logit": float(last[i]),
                     "grad": float(glast[i])} for i in top],
            "stages": stages,
            "stage_info": STAGE_INFO,
            "weight_grad_norms": wnorms,
        }
