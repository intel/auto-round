# SVDQuant RTN Residual Iteration Design

## Goal

Add model-independent residual iteration to AutoRound's SVDQuant transform while reusing the existing RTN MXFP4 quantization implementation.

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

Out of scope for the first implementation:

- Running SignRound optimization inside every outer iteration.
- FLUX-specific module naming, fusion, or checkpoint conversion.
- Nunchaku tensor packing. Packing consumes the selected residual later through the exporter.
- Activation-aware output-error selection. This can be added without changing the iteration interface.
- Joint optimization of smooth scales and low-rank factors.

## Configuration

Extend `SVDQuantConfig` with:

```python
residual_iters: int = 1
residual_early_stop: bool = False
residual_quant_method: str = "rtn"
```

Validation rules:

- `residual_iters >= 1`.
- The first version accepts only `residual_quant_method="rtn"` when `residual_iters > 1`.
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

The exporter receives the final quantized residual and BF16 low-rank factors. Residual iteration does not change the generic schema:

```text
qweight, wscales, lora_down, lora_up, smooth, smooth_orig, bias
```

## Error Handling

- Reject `residual_iters < 1` during config validation.
- Reject unsupported outer-loop methods with an actionable message.
- For a non-quantized layer or missing quantization scheme, use the one-round decomposition only when `residual_iters=1`; otherwise fail rather than silently using a different quantizer.
- If SVD or QDQ produces non-finite values, keep the last finite best candidate. If no finite candidate exists, fail with the module name and iteration index.
- Padding required by group quantization is handled by the existing AutoRound QDQ function and must be reverted before computing the objective.

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
9. Existing SVDQuant and generic exporter tests continue to pass.

## Future SignRound Support

SignRound outer iteration requires block calibration and optimization, not just a stateless layer QDQ call. It should later be implemented as a pipeline-level alternating coordinator:

```text
update low-rank factors -> optimize residual block with SignRound -> evaluate -> repeat
```

That work must reuse SignRound's existing wrapper, calibration IO, optimizer, and best-parameter collection. It should not be simulated by embedding a second SignRound implementation inside SVDQuant.
