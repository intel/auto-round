# Wan Sparse Attention Plan

- Hook Wan through `WanAttention.processor`, not by globally monkey-patching `torch.nn.functional.scaled_dot_product_attention`.
- `run_wan.py` uses Diffusers `WanAttnProcessor`, which routes attention through `dispatch_attention_fn(...)`, so the processor layer is the narrowest reliable integration point.
- First slice only sparsifies Wan self-attention where `Q/K/V` come from the latent sequence and the attention is square.
- Wan cross-attention and any unsupported self-attention shapes fall back to the original Diffusers processor unchanged.
- The sparse execution path should use the existing ARK preprocess+kernel helper `sparge_sage2_attn_meansim_topk_xpu(...)`.
- Validation target is one end-to-end `run_wan.py` smoke run using `/home/yiliu7/workspace/venvs/diffuser/bin/python`.
- Non-goals for this slice:
  - generic Diffusers-wide sparse patching
  - sparse Wan cross-attention
  - performance tuning or parity harnessing
