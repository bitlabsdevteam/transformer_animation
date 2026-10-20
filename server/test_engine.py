"""The hand-written forward/backward must agree with Hugging Face's GPT2LMHeadModel."""

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from engine import LAYER_STAGES, MODEL_ID, Engine
from explain import Explainer

TEXT = "Data visualization empowers users to"


@pytest.fixture(scope="module")
def engine():
    return Engine()


@pytest.fixture(scope="module")
def hf():
    return GPT2LMHeadModel.from_pretrained(MODEL_ID).float().eval()


def test_logits_match_hf(engine, hf):
    ids = engine.tokenize(TEXT)
    with torch.no_grad():
        ours = engine.forward(ids)
        ref = hf(torch.tensor([ids])).logits[0]
    assert torch.allclose(ours, ref, atol=1e-3, rtol=1e-4)


def test_weight_gradients_match_hf(engine, hf):
    ids = engine.tokenize(TEXT)
    _, payload = engine.run(TEXT)
    target = payload["target"]["id"]
    hf.zero_grad()
    logits = hf(torch.tensor([ids])).logits[0, -1]
    F.cross_entropy(logits[None], torch.tensor([target])).backward()
    for ours, theirs in [("h.0.attn.c_attn.weight", "transformer.h.0.attn.c_attn.weight"),
                         ("h.5.mlp.c_fc.weight", "transformer.h.5.mlp.c_fc.weight"),
                         ("h.11.ln_2.weight", "transformer.h.11.ln_2.weight"),
                         ("wpe.weight", "transformer.wpe.weight")]:
        g_ref = dict(hf.named_parameters())[theirs].grad
        g_ours = engine.W[ours].grad
        assert torch.allclose(g_ours, g_ref, atol=1e-5, rtol=1e-3), ours


def test_every_stage_explains_consistently(engine):
    run_id, payload = engine.run(TEXT)
    run = engine.runs[run_id]
    ex = Explainer(engine)
    T = payload["T"]
    keys = ["embed", "ln_f", "logits"] + [f"L{l}.{n}" for l in (0, 6, 11) for n in LAYER_STAGES]
    for key in keys:
        for t in (0, T - 1):
            for mode in ("forward", "backward"):
                kw = {"head": 3} if key.endswith("attn") else {"bin_": 5}
                out = ex.explain(run, key, token=t, mode=mode, **kw)
                if "ok" in out["forward"]:
                    assert out["forward"]["ok"], (key, t, out["forward"])
                b = out["backward"]
                assert b["ok"], (key, t, mode, b["manual"], b["autograd"])


def test_gradient_flows_to_every_stage(engine):
    _, payload = engine.run(TEXT)
    for s in payload["stages"]:
        assert max(s["gnorm"]) > 0, s["key"]
