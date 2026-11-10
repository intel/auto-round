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

Serving this export in vLLM requires compatible configuration mapping and
DeepSeek V4 grouped NVFP4 runtime support. Native Hub names do
not by themselves establish serving compatibility. This change does **not**
provide that adapter or establish native vLLM serving support. GPU kernels,
full-model generation, and thinking accuracy have not been tested here.

## AutoRound / vLLM responsibility boundary

Keep PR #2606 focused on V4 loading/calibration, grouped NVFP4 packing, and
restoring **actual weight keys** to the native Hub namespace without corrupting
packed bytes. The existing config-name reverter only gains a V4 dispatch.

Reuse upstream configuration mapping rather than adding a second generic mapper:
[compressed-tensors #1009](https://github.com/vllm-project/compressed-tensors/pull/1009)
resolves matching regex targets to concrete checkpoint names in
`QuantizationConfig.from_pretrained`;
[vLLM #66342](https://github.com/vllm-project/vllm/pull/66342) maps config names
on load, then consolidates ignores and same-scheme targets. Both are open at
review time. They do not restore tensor keys or implement grouped NVFP4 kernels.
AutoRound's NVFP4 exporter builds config groups directly, so #1009 is not an
automatic replacement for its existing config finalization path.

Keep only necessary runtime work separate: grouped `wo_a` with TP/RoPE handling
and independent fused-partition scales; reject unsupported expert scales instead
of silently merging them. Validate a fresh export with upstream mapping and
runtime support before claiming compatibility. No private adapter flags or
runtime checkpoint-repair protocol are introduced by this PR.