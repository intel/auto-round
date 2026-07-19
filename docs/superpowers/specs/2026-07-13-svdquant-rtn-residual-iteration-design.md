# SVDQuant RTN Residual Iteration Design

## Goal

Add model-independent residual iteration to AutoRound's SVDQuant transform while reusing the existing RTN MXFP4 quantization implementation. Export the selected result as a packed Nunchaku-compatible MXFP4 safetensors artifact so QDQ, kernel, and end-to-end quality can be compared.

The default `residual_iters=1` must preserve the current decomposition and the behavior used by the validated DeepCompressor/Nunchaku model:

```text
L1 = rank_r(W)
R1 = W - L1
Q1 = RTN_MXFP4(R1)
W_hat = Q1 + L1
```

The low-rank branch, smooth tensors, and bias remain in the configured high-precision dtype. Only the residual weight is quantized to MXFP4.

## Scope

The first implementation supports RTN as the quantizer inside the outer residual loop. It remains independent of model architecture and operates on ordinary two-dimensional Linear weights.

In scope:

- `residual_iters` with a default of `1`.
- Alternating low-rank approximation and RTN QDQ for `residual_iters > 1`.
- Existing AutoRound MXFP4 data types, including `mx_fp4` and `mx_fp4e2m1`.
- Group size 32 validation for deployable MXFP4.
- Weight-error objective for selecting the best iteration.
- Optional early stopping when an iteration does not improve the objective.
- Unit tests for one-round compatibility and multi-round behavior.
- AutoRound-owned E2M1 value encoding and UE8M0 scale encoding.
- Nunchaku-compatible `qweight` and `wscales` packing.
- Nunchaku-compatible low-rank, smooth, bias, and quantization metadata.
- A model-adapter interface that keeps architecture-specific naming and fusion out of the SVDQuant core.
- One reference runtime adapter for the existing FLUX/Nunchaku comparison, implemented outside the generic transform and packer.
- Exported-artifact QDQ validation.

Out of scope for the first implementation:

- Running SignRound optimization inside every outer iteration.
- Activation-aware output-error selection. This can be added without changing the iteration interface.
- Joint optimization of smooth scales and low-rank factors.

The first implementation establishes the generic packer and adapter interface, then uses a FLUX adapter as its first end-to-end consumer. FLUX rules must not enter the residual iteration, RTN adapter, or generic packer.

## Configuration

Extend `SVDQuantConfig` with:

```python
residual_iters: int = 1
residual_early_stop: bool = False
residual_quant_method: str = "rtn"
```

Validation rules:

- `residual_iters >= 1`.
- `residual_quant_method` is a compatibility argument fixed to `"rtn"` by design.
- `residual_iters=1` does not require an outer-loop quantizer call during the transform. This preserves the existing pipeline: SVDQuant creates the residual Linear, then the configured downstream RTN or SignRound quantizer processes it.
- For `residual_iters>1`, the layer must have a quantizable weight scheme. MXFP4 deployment requires group size 32.

Keeping the one-round path unchanged is important: a user can still combine `residual_iters=1` with either RTN or SignRound, and the existing quantizer remains the owner of final quantization parameters.

## Algorithm

For `residual_iters=1`, retain the current implementation:

```text
L1 = truncated_svd(W_smooth, rank)
R1 = W_smooth - L1
```

For `residual_iters=N`, where `N > 1`:

```text
Q0 = 0
best_error = infinity

for k in 1..N:
    Lk = truncated_svd(W_smooth - Q(k-1), rank)
    Rk = W_smooth - Lk
    Qk = rtn_qdq(Rk, layer_quantization_scheme)
    error_k = ||W_smooth - (Qk + Lk)||_F^2
    retain (Lk, Rk) when error_k improves best_error
    optionally stop after the first non-improving iteration

materialize the retained residual Rbest and low-rank factors Lbest
```

The retained residual is not stored as QDQ values in `residual_linear.weight`. It remains the high-precision `Rbest` so the existing downstream RTN pass can produce and record the final quantization state using its normal code path. `Qk` is used only to update the next low-rank target and evaluate candidates.

This intentionally reuses AutoRound's registered quantization function selected from the layer's `data_type`, `bits`, `group_size`, and related scheme attributes. SVDQuant must not implement a second MXFP4 numerical quantizer.

## Components

### Residual QDQ Adapter

Add a small internal adapter owned by the SVDQuant transform. It resolves the existing AutoRound quantization function and performs QDQ without packing:

```python
qdq_residual(weight, scheme) -> dequantized_weight
```

It depends only on AutoRound's data-type registry and layer quantization attributes. It does not import DeepCompressor or Nunchaku.

### SVDQuant Transform

The transform owns the alternating loop because it owns the low-rank factors. It asks the adapter for RTN QDQ values but does not own scale search internals or packed output.

### Downstream Quantizer

RTN or SignRound continues to quantize the selected `residual_linear` after decomposition. The low-rank Linear modules remain marked as 16-bit and are excluded from downstream quantization.

### Exporter

The exporter receives the selected high-precision residual and BF16 low-rank factors. It quantizes and packs the residual with an AutoRound-owned Nunchaku MXFP4 residual provider. Residual iteration does not change the serialized Linear schema:

```text
qweight, wscales, lora_down, lora_up, smooth, smooth_orig, bias
```

The default Nunchaku export must not emit `residual.weight`. That key remains available only in an explicitly selected debug export.

The packer must encode:

- residual values as FP4 E2M1, two values per byte;
- one UE8M0 scale per 32 input-channel values;
- Nunchaku's required weight and scale tile order;
- BF16 low-rank tensors in `(in_features, rank)` and `(out_features, rank)` layouts;
- required padding without changing the logical dimensions recorded by the model adapter.

The packer may reuse AutoRound's numerical MXFP4 QDQ function for choosing values and scales, but packing is a serialization responsibility and must have its own reversible unpack/QDQ reference implementation. It must not import DeepCompressor or Nunchaku.

### Metadata

The metadata must follow the structure recognized by the already validated Nunchaku model rather than the current prototype schema:

```json
{
  "method": "svdquant",
  "weight": {
    "dtype": "fp4_e2m1_all",
    "scale_dtype": "ue8m0",
    "group_size": 32
  },
  "activation": {
    "dtype": "fp4_e2m1_all",
    "scale_dtype": "ue8m0",
    "group_size": 32
  },
  "rank": 32
}
```

This JSON is stored in the safetensors `quantization_config` metadata field. `model_class`, the model configuration, and any runtime-specific metadata are supplied by the model adapter. Export must reject an artifact advertised as runtime-loadable when required model metadata is absent.

### Model Adapter

Nunchaku runtimes load complete supported model architectures, not arbitrary collections of Linear layers. Define an adapter boundary:

```python
adapter.map_module(module_name, module) -> one or more export records
adapter.metadata(model) -> dict[str, str]
adapter.validate(tensors, metadata) -> None
```

The identity adapter is useful for packer and QDQ tests but produces a generic intermediate artifact. A runtime-loadable adapter owns model-specific fused projection rules, key names, padding constraints, and model metadata. This keeps the quantization algorithm reusable while still allowing a real Nunchaku end-to-end comparison.

### Export Entry Point

Expose the implementation through AutoRound's normal format mechanism, with a direct Python API retained for tests:

```python
model.save_quantized(
    output_dir,
    format="svdquant_nunchaku",
    weight_dtype="mx_fp4e2m1",
    group_size=32,
    model_adapter=adapter,
)
```

The export path must consume the quantization scheme selected by AutoRound rather than silently replacing it with fixed defaults.

## Error Handling

- Reject `residual_iters < 1` during config validation.
- Reject unsupported outer-loop methods with an actionable message.
- For a non-quantized layer or missing quantization scheme, use the one-round decomposition only when `residual_iters=1`; otherwise fail rather than silently using a different quantizer.
- If SVD or QDQ produces non-finite values, keep the last finite best candidate. If no finite candidate exists, fail with the module name and iteration index.
- Padding required by group quantization is handled by the existing AutoRound QDQ function and must be reverted before computing the objective.
- Reject runtime MXFP4 export when group size is not 32 or when the scale type is not UE8M0.
- Reject mixed or unsupported residual data types unless the model adapter explicitly supports them.
- Validate every exported tensor's expected dtype, rank, packed dimensions, and required metadata before writing the final file.

## Testing

CPU tests use small deterministic Linear layers.

1. `residual_iters=1` produces the same residual and low-rank factors as the current one-shot implementation within dtype tolerance.
2. The low-rank modules and smooth tensor remain BF16 when configured as BF16.
3. `residual_iters>1` calls the registered RTN QDQ function with the layer's data type and group size.
4. Multi-round reconstruction error is no worse than the retained first-round candidate.
5. Early stopping retains the best candidate rather than the last candidate.
6. Invalid iteration counts and unsupported methods raise clear errors.
7. `mx_fp4e2m1`, group size 32, completes QDQ with correct shape and finite values.
8. A downstream RTN pass quantizes only `residual_linear`; low-rank modules remain unquantized.
9. E2M1 and UE8M0 encode/decode tests cover zero, signs, representable values, saturation, subnormal behavior, and exponent boundaries.
10. Pack then unpack/QDQ reconstructs the same values and scales produced by the RTN MXFP4 reference path.
11. `qweight` and `wscales` shapes, dtypes, padding, and tile order match a known-good tensor from the validated DeepCompressor artifact.
12. Safetensors metadata is recognized as `mxfp4`, includes rank and activation format, and contains no debug `residual.weight` key.
13. The identity adapter round-trip test validates generic Linear records without architecture dependencies.
14. A runtime adapter integration test loads the exported artifact through Nunchaku and compares one Linear kernel against the reference QDQ result.
15. The end-to-end comparison uses identical BF16 base model, prompt, seed, scheduler, and step count for BF16, known-good MXFP4, and AutoRound MXFP4 outputs.
16. Existing SVDQuant and exporter tests continue to pass after the prototype metadata expectations are updated.

## Comparison Gate

The feature is not complete merely because safetensors writing succeeds. Completion requires these gates in order:

1. Weight-level RTN MXFP4 QDQ matches the exported pack/unpack reference.
2. Packed layout matches known-good Nunchaku tensors for equivalent logical inputs.
3. Nunchaku loads the exported artifact without key, shape, dtype, or metadata fallback.
4. A Nunchaku Linear kernel agrees with unpacked QDQ within the established BF16 tolerance.
5. A supported model adapter completes an end-to-end inference comparison without noise.

## SignRound Boundary

SignRound is deliberately not an outer-iteration method. Residual outer iteration is
always the stateless RTN QDQ operation, regardless of whether the downstream pipeline
uses RTN or SignRound for final residual quantization:

```text
SVDQuant RTN residual outer iteration -> downstream RTN or SignRound quantizer
```

This keeps low-rank/residual decomposition independent of block calibration and leaves
SignRound's existing wrapper, optimizer, and best-parameter collection as the sole
owners of final SignRound quantization.
