# DeepSeek V4 NVFP4

Requires Transformers with `DeepseekV4GroupedLinear` support (CPU regressions use
Transformers 5.14.1). Native FP8 checkpoints use the full Transformers dequantizing
loader, not the generic meta/disk-stream loader, even with `AR_DISK_STREAM_MODEL=1`.
Plan for substantial host RAM, including MoE unfusing copies.

## Usage

Use representative tokenized calibration data and enough RAM/device memory for the model:

```python
from auto_round import AutoRound

quantizer = AutoRound(
    "/path/to/DeepSeek-V4-Flash",
    scheme="NVFP4",
    iters=0,
    disable_opt_rtn=True,
    disable_model_free=True,
    batch_size=1,
    nsamples=128,
    seqlen=2048,
    device_map="auto",
)
quantizer.quantize_and_save("./DeepSeek-V4-Flash-NVFP4", format="auto_round:llm_compressor")
```

`format="auto_round"` is also supported for grouped packing. Decoder replay keeps
real `input_ids` for hash routing and caches dictionary RoPE inputs.

## Supported constraints

- Grouped `wo_a` / Transformers `self_attn.o_a_proj`: NVFP4 weight quantization, 4 bits,
  group size 16, no bias; input width divisible by 16 and output width divisible
  by the number of groups. Unsupported grouped formats fail explicitly.
- Activations: NVFP4 with a calibrated global scale and group size 16, or
  weight-only (`act_bits=16` in the layer configuration). Other grouped activation
  formats are not supported.
- Both exporters retain NVFP4 packed bytes, group scales, and global scales.
  Saved V4 tensor keys and module-name metadata use the native Hub namespace:
  `layers.N.ffn.experts.E.w1/w2/w3`, `layers.N.attn.wo_a`, `embed`, and `head`,
  not the loaded Transformers `model.layers.N.mlp/self_attn` namespace.
  Unquantized fused experts are split into native per-expert matrices. Only
  names/layout views change; no source FP8 quantization is reapplied. Full saves
  and immediate shard saves follow the same naming contract, while the in-memory
  model remains in the Transformers namespace.
  Compatible Transformers saves use `save_original_format=False` for packed
  parameters **and buffers**, without changing source conversion metadata.
- AutoRound inference selects `auto_round:torch_nvfp4` for grouped layers when
  the backend is `auto`. It unpacks weights and performs grouped BMM in the input
  dtype. This is a correctness reference, not a fast serving kernel.

## Validation and serving

CPU tests cover native-loader selection/dequantization arguments, routing IDs,
dictionary RoPE, both grouped packers, packed-state reloads, and a tiny real V4
attention RTN/export run. The tiny run leaves MoE weights full precision; it does
not validate full-checkpoint loading, expert calibration coverage, or accuracy.

Serving this export in vLLM requires a separate, compatible DeepSeek V4 NVFP4
adapter for quantization metadata and grouped projections. Native Hub names do
not by themselves establish serving compatibility. This change does **not**
provide that adapter or establish native vLLM serving support. GPU kernels,
full-model generation, and thinking accuracy have not been tested here.