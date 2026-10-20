# Reverse engineering: poloclub Transformer Explainer

Target: <https://poloclub.github.io/transformer-explainer/> (live build, analysed 2026-10-20).
The GitHub repository was **not** cloned. Everything below comes from the shipped site, captured with
REA (Reverse Engineer Anything) through Chrome DevTools Protocol.

Labels: **Observed** = seen in shipped bytes or runtime. **Inferred** = reasoned from observations.
**Design** = a choice made for this rebuild.

## REA evidence

| Evidence ID | Operation | What it established |
|---|---|---|
| `ev_c426c670…45b8f5` | `analyze_web_bundle` | 14 scripts. SvelteKit/Vite chunk graph (`entry/app`, `nodes/0..2`, `chunks/*`). `nodes/2.DCrB1VOc.js` (7 MB) holds the app, transformers.js and onnxruntime-web. Tokenizer special tokens `<\|endoftext\|>`. |
| `ev_fbf93bed…d528c` | `inspect_web_page` | Resource inventory: KaTeX 0.12 CSS/fonts from jsDelivr, article images (`QKV.png`, `attention.png`, `embedding.png`, `mlp.png`, `softmax.png`). No model fetch was seen in the attach window. **Inferred**: it happened before REA attached (fresh profile, so no cache). |
| `ev_27645440…acb6` | `inspect_web_page` + script sources | Retained bytes of all page scripts (exceeded MCP frame, exported via `export_evidence_bundle`). |
| `ev_5d6c59c9…5484` | `export_web_scripts` | Digest-verified export of the 12 module scripts used for static reading below. |

## Observed pipeline

1. **Framework**: SvelteKit + Vite. Svelte stores drive everything: `inputText`, `tokens`, `tokenIds`,
   `modelData`, `modelSession`, `temperature`, `sampling`, `predictedToken`, `expandedBlock`, `blockIdx`.
   Equations use KaTeX. **Inferred**: the flow animation (`showFlowAnimation`, minified as `hz` in `chunks/kbXKSa8s.js`) is a
   GSAP-style timeline. It was matched by shape (`timeline({onStart,onComplete})`, `fromTo`), not by name.
2. **Tokenizer**: `AutoTokenizer.from_pretrained("Xenova/gpt2")` (transformers.js). Tokens are shown via
   `tokenizer.decode([id])` for each id.
3. **Model download**: 63 chunks `${base}/model-v2/gpt2.onnx.part{0..62}` (10 MiB each, ~600 MB total).
   They're fetched in parallel, cached in Cache Storage `onnx-model-cache-v2`, concatenated into one
   `Blob`, and opened with `InferenceSession.create(blobURL)`. The runtime is onnxruntime-web **1.23.0**, with WASM from
   `cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/`.
4. **Before the model loads**: `fakeRunWithCachedData` replays one of 5 bundled example results (`ex0..ex4`).
5. **Inference** (`runModel` → `getData`): input `{input: int64[1, T]}`. Outputs used:
   - `linear_output`: logits for every position (vocab 50257).
   - For each block `l` and head `h`: `block_{l}_attn_head_{h}_attn`, `…_attn_scaled`, `…_attn_masked`,
     `…_attn_softmax`, `…_attn_dropout`. These are the only intermediate tensors the site reads (`targetTensors`).
6. **Sampling** (`getProbabilities`): sort logits, keep the top 50, divide by temperature, then
   - top-k: set ranks ≥ k to −∞ and softmax;
   - top-p: softmax, take the cumulative cutoff, renormalise.
   `randomChoice` then samples from the result. Temperature and sampling changes re-run only this step (`adjustTemperature`).
7. **Visualization**: the embedding, QKV, attention-matrix, MLP and probability views are SVG/HTML.
   **Inferred**: those activation matrices are illustrative colour patterns, except attention, which uses the
   real per-head tensors above, and the logits/probabilities.
8. **No backward pass**: there are no gradient tensors, loss, or training path anywhere in the bundle.

## What the rebuild changes (Design)

| Original | Rebuild |
|---|---|
| ONNX export, attention tensors only | A hand-written GPT-2 forward (`server/engine.py`) from the HF `openai-community/gpt2` weights keeps **every** stage: embedding, LN1, Q, K, V, scores, scaled, masked, softmax, Σ P·V, attention projection, residual, LN2, MLP fc, GELU, MLP projection, residual, final LN, logits |
| Forward only | Backward pass: loss = −log p(target \| text) on the last position (target defaults to the argmax, and you can click any candidate to change it). PyTorch autograd provides ∂L/∂(stage) for every stage, plus ‖∂L/∂W‖ for every weight |
| 2D SVG + GSAP | Three.js 3D: one instanced bar field per stage (tokens × 64 dimension bins, max-\|x\| per bin). Attention is drawn as 12 causal T×T tiles. Animated forward and backward sweeps |
| Static formulas | `server/explain.py` recomputes any selected cell with real numbers. It also re-derives the gradient by hand from downstream gradients (one chain-rule step) and checks it against autograd |
| Runs fully in browser | Needs a local Python server (PyTorch autograd is the source of exact gradients), so it's not static-hostable |

Sampling (temperature → top-50 → top-k / top-p) is reimplemented to match the observed algorithm.

## Verification

`server/test_engine.py`:
- The hand-written forward's logits match `GPT2LMHeadModel` (atol 1e-3).
- Weight gradients for `h.0.attn.c_attn`, `h.5.mlp.c_fc`, `h.11.ln_2` and `wpe` match HF autograd.
- For every stage in blocks 1, 7 and 12, the forward recomputation equals the stored activation, and the
  hand-derived backward step equals the autograd gradient.
- Gradient reaches every stage.

## Unknowns

- The exact ONNX export script and the 5 cached example payloads weren't examined in detail. They aren't needed for the rebuild.
- Source maps weren't published, so names come from the minified bundle, which kept many of its identifiers.
