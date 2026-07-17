# SVDQuant AWQ Smooth Search Design

## Goal

Replace SVDQuant's fixed-alpha SmoothQuant formula with an activation-aware
scale search that reuses AutoRound's AWQ search infrastructure. Enabling
SVDQuant smoothing must search candidate scales using the configured weight
QDQ behavior and output reconstruction error.

The first implementation intentionally searches scales before SVD
decomposition. It does not run an SVD for every search candidate.

## User-Facing Behavior

SVDQuant smoothing remains disabled by default. When enabled, it always uses
AWQ-style grid search:

```bash
--enable_svdquant_smooth \
--svdquant_smooth_n_grid 20
```

The Python API exposes the equivalent configuration:

```python
SVDQuantConfig(
    smooth_enabled=True,
    smooth_n_grid=20,
)
```

The existing `--svdquant_smooth_alpha` CLI argument and
`SVDQuantConfig.smooth_alpha` field are removed without a compatibility alias.
The fixed activation-amax/weight-amax formula is removed.

`smooth_n_grid` defaults to 20 and must be an integer greater than or equal to
2. It controls only SVDQuant smooth search and does not implicitly change a
separate AWQ transform in the same pipeline.

## Architecture

### Shared AWQ Search Service

Extract the reusable parts of `AWQTransform` into an internal activation-aware
scale search service. The service owns:

- candidate ratio generation;
- activation and weight scale construction;
- per-layer quantization parameter resolution;
- dispatch through AWQ's existing `QDQTool`;
- candidate output-error evaluation; and
- best-candidate selection.

The service accepts a consumer-provided candidate evaluator and scale consumer.
It must not mutate upstream normalization layers or assume AWQ scale folding.
`AWQTransform` continues to use the service and applies its selected scale by
folding it into the smooth and balance layers as it does today.

The QDQ path must resolve the actual per-layer scheme, including MXFP4 data
type, group size, symmetry, optimized-RTN setting, and layer overrides. Search
must therefore evaluate the same weight quantization behavior used by the
terminal block quantizer.

### SVDQuant Consumer

For every targeted `torch.nn.Linear`, SVDQuant retains bounded calibration
inputs for the current block. It uses those inputs and the shared AWQ service
to evaluate candidate channel scales.

For a candidate scale `s`, the equivalent transformed Linear is:

```text
x_hat = x / s
W_hat = W * s
```

The search reference is the original floating-point Linear output. The search
candidate output uses QDQ on `W_hat` and evaluates the quantized Linear with
`x_hat`. Mean squared output error selects the best candidate. Bias is included
in both reference and candidate outputs when present, so it does not receive
channel scaling.

After selecting `s`, SVDQuant performs its existing decomposition once:

```text
W_hat = residual + low_rank
```

The residual follows the existing RTN residual-iteration path and is later
quantized by the terminal quantizer. The low-rank branch stays in the configured
high-precision dtype. The runtime wrapper stores `smooth = 1 / s`, preserving
the current forward and Nunchaku export contract:

```text
x_hat = x * smooth
output = residual(x_hat) + low_rank(x_hat)
```

### Calibration Lifetime

Calibration inputs are block-local and released after that block's scales have
been selected. The implementation must not retain all FLUX activations for the
whole model. Input sampling or token reduction must be deterministic and
bounded by an explicit internal limit shared with the AWQ error-evaluation
path.

If smoothing is disabled, SVDQuant must not register activation hooks or retain
calibration inputs.

## Search Semantics

The first version uses the existing AWQ ratio grid and duo-scaling formula.
It does not expose a fixed alpha and does not provide a fixed/search mode.

Candidate failures are handled independently. A candidate producing non-finite
scales, QDQ values, outputs, or error is discarded. If all candidates fail, the
quantization run fails with the module name and failure reason instead of
silently selecting identity smoothing.

Ties are resolved deterministically by retaining the first candidate with the
minimum finite error.

## Export Compatibility

The exported Nunchaku representation does not change. The selected runtime
smooth tensor is exported with the same key, shape, dtype, and direction used
by the current SVDQuant exporter. The residual MXFP4 metadata and low-rank
weights are unchanged.

No dependency on DeepCompressor or Nunchaku may be introduced into AutoRound's
quantization implementation. Nunchaku remains relevant only to the existing
export format and external runtime validation.

## Error Handling

- Reject `smooth_n_grid < 2` during configuration construction.
- Require calibration when `smooth_enabled=True`.
- Fail clearly when a target Linear has no usable calibration inputs.
- Fail clearly when QDQ cannot resolve the target layer's quantization scheme.
- Include the global module name in all search failures.
- Never silently fall back to the removed fixed formula or identity smoothing.

## Testing

Unit tests must cover:

- the fixed `smooth_alpha` API and CLI are absent;
- smoothing defaults to disabled and `smooth_n_grid` defaults to 20;
- invalid grid sizes are rejected;
- candidate generation and deterministic tie behavior;
- QDQ parameter resolution for MXFP4 and per-layer overrides;
- a constructed example where search chooses the known lower-error scale;
- non-finite candidates are skipped and all-candidate failure is reported;
- smoothing-disabled execution registers no calibration hooks;
- selected runtime smooth preserves floating-point forward equivalence before
  residual quantization;
- SVD decomposition runs once per Linear, not once per grid candidate;
- activation buffers are released after each block; and
- existing AWQ behavior and tests remain unchanged after extraction.

Integration validation must run a small SVDQuant + MXFP4 QDQ case through the
AutoRound pipeline and verify its reconstructed output against the reference.
FLUX validation then uses the existing CLI and Nunchaku export/load smoke path.

## Out Of Scope

- Running SVD and residual QDQ for every scale candidate.
- A two-stage AWQ shortlist plus SVD-aware refinement.
- AWQ clipping as part of SVDQuant smooth search.
- Fixed-alpha compatibility behavior.
- Changes to residual iteration, low-rank rank selection, or Nunchaku kernels.

