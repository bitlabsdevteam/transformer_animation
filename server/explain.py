"""Turn one selected cell of one stage into forward + backward calculations with real numbers.

Forward: recompute the cell from its inputs and weights.
Backward: re-derive dL/d(cell) by hand from the downstream gradients (one chain-rule
step) and compare it with the value PyTorch autograd produced.
"""

from __future__ import annotations

import math

import torch

from engine import BINS, STAGE_INFO, Engine, Run

TOP_TERMS = 8


def _terms(a: torch.Tensor, b: torch.Tensor, labels_a, labels_b):
    """Top contributing products a_i*b_i of a dot product, plus the full sum."""
    prods = a * b
    order = prods.abs().argsort(descending=True)[:TOP_TERMS].tolist()
    return {
        "rows": [{"i": i, "la": labels_a(i), "lb": labels_b(i), "a": float(a[i]), "b": float(b[i]),
                  "prod": float(prods[i])} for i in order],
        "n": int(a.numel()),
        "sum": float(prods.sum()),
    }


def _check(manual: float, auto: float) -> dict:
    err = abs(manual - auto)
    return {"manual": manual, "autograd": auto, "abs_err": err,
            "ok": err <= 1e-4 * max(1.0, abs(auto)) + 1e-7}


def _ln_backward_row(x: torch.Tensor, gy: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    mu = x.mean()
    sigma = torch.sqrt(x.var(unbiased=False) + eps)
    xhat = (x - mu) / sigma
    gx = gy * gamma
    return (gx - gx.mean() - xhat * (gx * xhat).mean()) / sigma


class Explainer:
    def __init__(self, engine: Engine) -> None:
        self.e = engine

    # ---------------------------------------------------------------- helpers
    def _w(self, name):
        return self.e.W[name].detach()

    def _resolve_dim(self, run: Run, key: str, t: int, dim: int | None, bin_: int | None, mode: str) -> int:
        if dim is not None:
            return dim
        src = run.grads[key] if mode == "backward" else run.acts[key]
        row = src[t]
        w = row.numel() // BINS
        b = int(bin_ or 0)
        return b * w + int(row[b * w:(b + 1) * w].abs().argmax())

    def _ln_forward(self, x: torch.Tensor, prefix: str, d: int) -> dict:
        eps = self.e.eps
        mu = float(x.mean())
        var = float(x.var(unbiased=False))
        sigma = math.sqrt(var + eps)
        gamma = float(self._w(f"{prefix}.weight")[d])
        beta = float(self._w(f"{prefix}.bias")[d])
        xhat = (float(x[d]) - mu) / sigma
        return {
            "formula": r"y_d = \gamma_d \cdot \frac{x_d - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_d",
            "steps": [
                {"label": r"x_d", "value": float(x[d])},
                {"label": r"\mu = \tfrac{1}{768}\sum_i x_i", "value": mu},
                {"label": r"\sigma^2 = \tfrac{1}{768}\sum_i (x_i-\mu)^2", "value": var},
                {"label": r"\hat{x}_d = (x_d-\mu)/\sqrt{\sigma^2+10^{-5}}", "value": xhat},
                {"label": r"\gamma_d", "value": gamma},
                {"label": r"\beta_d", "value": beta},
            ],
            "result": gamma * xhat + beta,
        }

    def _residual_input_grad(self, run: Run, key: str, t: int, d: int) -> dict:
        """dL/dx for a residual-stream tensor = residual path + LayerNorm path."""
        L = self.e.n_layer
        if key == "embed":
            res_key, ln_key, ln_prefix = "L0.resid1", "L0.ln1", "h.0.ln_1"
        elif key.endswith("resid1"):
            l = int(key[1:key.index(".")])
            res_key, ln_key, ln_prefix = f"L{l}.resid2", f"L{l}.ln2", f"h.{l}.ln_2"
        else:
            l = int(key[1:key.index(".")])
            if l + 1 < L:
                res_key, ln_key, ln_prefix = f"L{l + 1}.resid1", f"L{l + 1}.ln1", f"h.{l + 1}.ln_1"
            else:
                res_key, ln_key, ln_prefix = None, "ln_f", "ln_f"
        x = run.acts[key][t]
        via_res = float(run.grads[res_key][t, d]) if res_key else 0.0
        via_ln = float(_ln_backward_row(x, run.grads[ln_key][t], self._w(f"{ln_prefix}.weight"), self.e.eps)[d])
        return {
            "formula": (r"\begin{aligned}\frac{\partial L}{\partial x_d} &= \underbrace{\frac{\partial L}{\partial r_d}}_{\text{residual skip}} \\"
                        r"&+ \underbrace{\frac{1}{\sigma}\Big(g_d\gamma_d - \overline{g\gamma} - \hat{x}_d\,\overline{g\gamma\hat{x}}\Big)}_{\text{through LayerNorm}}\end{aligned}"),
            "steps": [
                {"label": rf"\partial L/\partial r_d \;(\texttt{{{res_key or 'none'}}})", "value": via_res},
                {"label": rf"\text{{LayerNorm path }}(\texttt{{{ln_key}}})", "value": via_ln},
            ],
            **_check(via_res + via_ln, float(run.grads[key][t, d])),
        }

    # ---------------------------------------------------------------- main
    def explain(self, run: Run, key: str, token: int = -1, dim: int | None = None, bin_: int | None = None,
                head: int = 0, key_token: int | None = None, mode: str = "forward") -> dict:
        e, T = self.e, len(run.ids)
        t = token % T
        name = key.split(".")[-1] if key.startswith("L") else key
        layer = int(key[1:key.index(".")]) if key.startswith("L") else None
        p = f"h.{layer}" if layer is not None else None
        tok = run.tokens
        out = {"key": key, "name": name, "layer": layer, "title": STAGE_INFO[name][0],
               "desc": STAGE_INFO[name][1], "token": t, "token_str": tok[t]}
        A, G = run.acts, run.grads
        dh = e.d_head

        if name == "logits":
            v = run.target if dim is None else int(dim)
            xf = A["ln_f"][T - 1]
            wte = self._w("wte.weight")
            probs = torch.softmax(A["logits"][T - 1], -1)
            out.update(token=T - 1, token_str=tok[T - 1], dim=v, dim_label=e.tok.decode([v]))
            out["forward"] = {
                "formula": r"z_v = \sum_{d} \mathrm{LN_f}(x)_d \, W^{te}_{v,d},\qquad p_v = \frac{e^{z_v}}{\sum_u e^{z_u}}",
                "terms": _terms(xf, wte[v], lambda i: f"lnf[{i}]", lambda i: f"wte[{v},{i}]"),
                "steps": [{"label": r"p_v", "value": float(probs[v])},
                          {"label": r"L = -\log p_{\text{target}}", "value": run.loss}],
                "result": float(A["logits"][T - 1, v]),
            }
            manual = float(probs[v]) - (1.0 if v == run.target else 0.0)
            out["backward"] = {
                "formula": r"\frac{\partial L}{\partial z_v} = p_v - \mathbb{1}[v = \text{target}]",
                "steps": [{"label": r"p_v", "value": float(probs[v])},
                          {"label": r"\mathbb{1}[v=\text{target}]", "value": 1.0 if v == run.target else 0.0}],
                **_check(manual, float(G["logits"][T - 1, v])),
            }
            return out

        if name == "attn":
            h = head
            j = key_token if key_token is not None else int(A[key][h, t].argmax())
            j = min(j, T - 1)
            q = A[f"L{layer}.q"][t, h * dh:(h + 1) * dh]
            k = A[f"L{layer}.k"][j, h * dh:(h + 1) * dh]
            row = A[f"L{layer}.masked"][h, t]
            P = A[key][h, t]
            out.update(head=h, key_token=j, key_token_str=tok[j])
            out["forward"] = {
                "formula": (r"s_{tj} = q_t \cdot k_j,\quad \tilde{s}_{tj} = \frac{s_{tj}}{\sqrt{64}},"
                            r"\quad P_{tj} = \frac{e^{\tilde{s}_{tj}}}{\sum_{m \le t} e^{\tilde{s}_{tm}}}"),
                "terms": _terms(q, k, lambda i: f"q[{h * dh + i}]", lambda i: f"k[{h * dh + i}]"),
                "steps": [
                    {"label": r"s_{tj} = q_t\cdot k_j", "value": float(A[f"L{layer}.scores"][h, t, j])},
                    {"label": r"\tilde{s}_{tj} = s_{tj}/8", "value": float(A[f"L{layer}.scaled"][h, t, j])},
                    {"label": r"\text{mask: } j > t \Rightarrow -\infty", "value": "kept" if j <= t else "-inf"},
                    {"label": r"\sum_{m\le t} e^{\tilde{s}_{tm}}", "value": float(torch.exp(row[:t + 1]).sum())},
                ],
                "row": [{"j": m, "tok": tok[m], "p": float(P[m])} for m in range(t + 1)],
                "result": float(P[j]),
            }
            ctx_g = G[f"L{layer}.ctx"][t, h * dh:(h + 1) * dh]
            vj = A[f"L{layer}.v"][j, h * dh:(h + 1) * dh]
            gP_row = (G[f"L{layer}.ctx"][t, h * dh:(h + 1) * dh][None] * A[f"L{layer}.v"][:, h * dh:(h + 1) * dh]).sum(-1)
            gS = float(P[j] * (gP_row[j] - (P * gP_row).sum()))
            out["backward"] = {
                "formula": (r"\begin{aligned}\frac{\partial L}{\partial P_{tj}} &= \frac{\partial L}{\partial o_t}\cdot v_j \\"
                            r"\frac{\partial L}{\partial \tilde{s}_{tj}} &= P_{tj}\Big(\frac{\partial L}{\partial P_{tj}}"
                            r" - \sum_m P_{tm}\frac{\partial L}{\partial P_{tm}}\Big)\end{aligned}"),
                "terms": _terms(ctx_g, vj, lambda i: f"∂o[{h * dh + i}]", lambda i: f"v[{h * dh + i}]"),
                "steps": [{"label": r"\partial L/\partial \tilde{s}_{tj}\ \text{(manual)}", "value": gS},
                          {"label": r"\partial L/\partial \tilde{s}_{tj}\ \text{(autograd)}",
                           "value": float(G[f"L{layer}.masked"][h, t, j])}],
                **_check(float(gP_row[j]), float(G[key][h, t, j])),
            }
            return out

        d = self._resolve_dim(run, key, t, dim, bin_, mode)
        out.update(dim=d)
        fwd: dict
        bwd: dict

        def linear_fwd(x, wname, col, xl, formula):
            Wm, b = self._w(f"{wname}.weight"), self._w(f"{wname}.bias")
            res = _terms(x, Wm[:, col], lambda i: f"{xl}[{i}]", lambda i: f"W[{i},{col}]")
            return {"formula": formula, "terms": res,
                    "steps": [{"label": r"\sum_i x_i W_{i,d}", "value": res["sum"]},
                              {"label": r"b_d", "value": float(b[col])}],
                    "result": res["sum"] + float(b[col])}

        def linear_bwd(g_out, wname, row, gl, formula):
            Wm = self._w(f"{wname}.weight")
            res = _terms(g_out, Wm[row], lambda i: f"∂{gl}[{i}]", lambda i: f"W[{row},{i}]")
            return {"formula": formula, "terms": res, **_check(res["sum"], float(G[key][t, d]))}

        if name == "embed":
            wte, wpe = self._w("wte.weight"), self._w("wpe.weight")
            fwd = {"formula": r"x_{t,d} = W^{te}_{\text{id}_t, d} + W^{pe}_{t, d}",
                   "steps": [{"label": rf"W^{{te}}_{{{run.ids[t]},{d}}}", "value": float(wte[run.ids[t], d])},
                             {"label": rf"W^{{pe}}_{{{t},{d}}}", "value": float(wpe[t, d])}],
                   "result": float(wte[run.ids[t], d] + wpe[t, d])}
            bwd = self._residual_input_grad(run, key, t, d)
        elif name in ("ln1", "ln2", "ln_f"):
            src = {"ln1": f"L{layer - 1}.resid2" if layer else "embed", "ln2": f"L{layer}.resid1",
                   "ln_f": f"L{e.n_layer - 1}.resid2"}[name]
            prefix = {"ln1": f"{p}.ln_1", "ln2": f"{p}.ln_2", "ln_f": "ln_f"}[name]
            fwd = self._ln_forward(A[src][t], prefix, d)
            if name == "ln_f":
                gl = G["logits"][t]
                res = _terms(gl, self._w("wte.weight")[:, d], lambda i: f"∂z[{i}]", lambda i: f"wte[{i},{d}]")
                bwd = {"formula": r"\frac{\partial L}{\partial y_d} = \sum_v \frac{\partial L}{\partial z_v} W^{te}_{v,d}",
                       "terms": res, **_check(res["sum"], float(G[key][t, d]))}
            elif name == "ln2":
                bwd = linear_bwd(G[f"L{layer}.fc"][t], f"{p}.mlp.c_fc", d, "fc",
                                 r"\frac{\partial L}{\partial y_i} = \sum_j \frac{\partial L}{\partial f_j} W^{fc}_{i,j}")
            else:
                g_qkv = torch.cat([G[f"L{layer}.q"][t], G[f"L{layer}.k"][t], G[f"L{layer}.v"][t]])
                bwd = linear_bwd(g_qkv, f"{p}.attn.c_attn", d, "qkv",
                                 r"\frac{\partial L}{\partial y_i} = \sum_{o \in q,k,v} \frac{\partial L}{\partial o} W^{qkv}_{i,o}")
        elif name in ("q", "k", "v"):
            off = {"q": 0, "k": e.d, "v": 2 * e.d}[name]
            fwd = linear_fwd(A[f"L{layer}.ln1"][t], f"{p}.attn.c_attn", off + d, "ln1",
                             rf"{name}_{{t,d}} = \sum_i \mathrm{{LN}}(x)_{{t,i}} W^{{{name.upper()}}}_{{i,d}} + b_d")
            h, c = divmod(d, dh)
            gS = G[f"L{layer}.scores"][h]  # dL/d(raw q·k) = dL/d(scaled) / 8
            if name == "q":
                res = _terms(gS[t], A[f"L{layer}.k"][:, d], lambda j: f"∂s[{t},{j}]", lambda j: f"k[{j},{d}]")
                formula = r"\frac{\partial L}{\partial q_{t,d}} = \sum_j \frac{\partial L}{\partial s_{tj}} k_{j,d}"
            elif name == "k":
                res = _terms(gS[:, t], A[f"L{layer}.q"][:, d], lambda i: f"∂s[{i},{t}]", lambda i: f"q[{i},{d}]")
                formula = r"\frac{\partial L}{\partial k_{j,d}} = \sum_t \frac{\partial L}{\partial s_{tj}} q_{t,d}"
            else:
                Pm = A[f"L{layer}.attn"][h]
                res = _terms(Pm[:, t], G[f"L{layer}.ctx"][:, d], lambda i: f"P[{i},{t}]", lambda i: f"∂o[{i},{d}]")
                formula = r"\frac{\partial L}{\partial v_{j,d}} = \sum_t P_{tj} \frac{\partial L}{\partial o_{t,d}}"
            bwd = {"formula": formula, "terms": res, **_check(res["sum"], float(G[key][t, d]))}
            out.update(head=h)
        elif name == "ctx":
            h = d // dh
            Pm, Vm = A[f"L{layer}.attn"][h, t], A[f"L{layer}.v"][:, d]
            res = _terms(Pm, Vm, lambda j: f"P[{t},{j}]", lambda j: f"v[{j},{d}]")
            fwd = {"formula": r"o_{t,d} = \sum_{j \le t} P_{tj}\, v_{j,d}", "terms": res, "result": res["sum"]}
            bwd = linear_bwd(G[f"L{layer}.attn_out"][t], f"{p}.attn.c_proj", d, "a",
                             r"\frac{\partial L}{\partial o_i} = \sum_d \frac{\partial L}{\partial a_d} W^{O}_{i,d}")
            out.update(head=h)
        elif name == "attn_out":
            fwd = linear_fwd(A[f"L{layer}.ctx"][t], f"{p}.attn.c_proj", d, "o",
                             r"a_{t,d} = \sum_i o_{t,i} W^{O}_{i,d} + b_d")
            bwd = {"formula": r"\frac{\partial L}{\partial a} = \frac{\partial L}{\partial r}\quad(r = x + a)",
                   **_check(float(G[f"L{layer}.resid1"][t, d]), float(G[key][t, d]))}
        elif name in ("resid1", "resid2"):
            if name == "resid1":
                x_in = A[f"L{layer - 1}.resid2" if layer else "embed"][t, d]
                sub, sub_l = A[f"L{layer}.attn_out"][t, d], r"\mathrm{Attn}(x)_d"
            else:
                x_in, sub, sub_l = A[f"L{layer}.resid1"][t, d], A[f"L{layer}.mlp_out"][t, d], r"\mathrm{MLP}(x)_d"
            fwd = {"formula": r"r_d = x_d + f(x)_d",
                   "steps": [{"label": r"x_d", "value": float(x_in)}, {"label": sub_l, "value": float(sub)}],
                   "result": float(x_in + sub)}
            bwd = self._residual_input_grad(run, key, t, d)
        elif name == "fc":
            fwd = linear_fwd(A[f"L{layer}.ln2"][t], f"{p}.mlp.c_fc", d, "ln2",
                             r"f_{t,j} = \sum_i \mathrm{LN}(x)_{t,i} W^{fc}_{i,j} + b_j")
            z = A[key][t, d]
            u = math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)
            th = math.tanh(float(u))
            dgelu = 0.5 * (1 + th) + 0.5 * float(z) * (1 - th ** 2) * math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * float(z) ** 2)
            gg = float(G[f"L{layer}.gelu"][t, d])
            bwd = {"formula": r"\frac{\partial L}{\partial z} = \mathrm{GELU}'(z)\cdot\frac{\partial L}{\partial g}",
                   "steps": [{"label": r"\mathrm{GELU}'(z)", "value": dgelu},
                             {"label": r"\partial L/\partial g", "value": gg}],
                   **_check(dgelu * gg, float(G[key][t, d]))}
        elif name == "gelu":
            z = float(A[f"L{layer}.fc"][t, d])
            u = math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)
            fwd = {"formula": r"g = \tfrac{1}{2} z \big(1 + \tanh(\sqrt{2/\pi}\,(z + 0.044715 z^3))\big)",
                   "steps": [{"label": "z", "value": z}, {"label": r"\sqrt{2/\pi}(z+0.044715z^3)", "value": u},
                             {"label": r"\tanh(\cdot)", "value": math.tanh(u)}],
                   "result": 0.5 * z * (1 + math.tanh(u))}
            bwd = linear_bwd(G[f"L{layer}.mlp_out"][t], f"{p}.mlp.c_proj", d, "m",
                             r"\frac{\partial L}{\partial g_j} = \sum_d \frac{\partial L}{\partial m_d} W^{proj}_{j,d}")
        elif name == "mlp_out":
            fwd = linear_fwd(A[f"L{layer}.gelu"][t], f"{p}.mlp.c_proj", d, "g",
                             r"m_{t,d} = \sum_j g_{t,j} W^{proj}_{j,d} + b_d")
            bwd = {"formula": r"\frac{\partial L}{\partial m} = \frac{\partial L}{\partial r}\quad(r = x + m)",
                   **_check(float(G[f"L{layer}.resid2"][t, d]), float(G[key][t, d]))}
        else:
            raise KeyError(key)

        fwd["stored"] = float(A[key][t, d])
        fwd["ok"] = abs(fwd["result"] - fwd["stored"]) <= 1e-3 * max(1.0, abs(fwd["stored"]))
        out["forward"], out["backward"] = fwd, bwd
        return out
