# Transformer Explainer 3D

A rebuild of the poloclub Transformer Explainer, reverse-engineered from the live site (see
[docs/REVERSE_ENGINEERING.md](docs/REVERSE_ENGINEERING.md)). It adds two things:

- **Any text in, full 3D activations out.** Every intermediate tensor of GPT-2 small (12 blocks × 13 stages,
  plus embedding, final LayerNorm and logits) is drawn as an instanced 3D bar field.
- **Forward and backward passes.** It animates data flowing embedding → logits, then the gradient of the
  next-token loss flowing back. Click any bar to see the exact calculation with real numbers. The gradient
  is re-derived by hand and checked against PyTorch autograd.

## Run

```bash
cd server
uv run uvicorn app:app --port 8765
# open http://127.0.0.1:8765
```

The first start downloads GPT-2 weights from Hugging Face (~550 MB, cached afterwards).

## Use

- Type text and press **Run**. Up to the last 32 tokens are used.
- **▶ Forward** / **◀ Backward**: animated sweeps. Use **Space** to play or pause and **← / →** to step.
- **Activations / Gradients** switches what the bars show. Activations: amber = positive, blue = negative.
  Gradients: magenta = positive, cyan = negative. Each stage is normalised to its own 99th percentile.
- Each bar is one token × one bin of 12 dimensions (48 for the 3072-wide MLP), showing the value with the
  largest magnitude in that bin. The inspector resolves the exact dimension.
- Click a candidate in **Next token** to make it the backward-pass target. **Append** generates.
- The left panel shows ‖∂L/∂x‖ per block (log scale): how much gradient reaches each layer.

## Layout

```
server/engine.py      GPT-2 forward from the state_dict, keeps every stage, autograd backward, payload
server/explain.py     per-cell forward recomputation + hand-derived backward vs autograd
server/app.py         FastAPI: /api/run, /api/detail, static web/
server/test_engine.py checks against Hugging Face GPT2LMHeadModel
web/                  Three.js (importmap, three@0.160.0) + KaTeX 0.16.9 front end
docs/                 reverse-engineering notes and evidence IDs
```

Tests: `cd server && uv run pytest -q`
