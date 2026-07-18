# SVDQuant Low-Rank-Aware Smooth Search Design

## Goal

Replace AutoRound SVDQuant's fixed-alpha smoothing with a low-rank-aware grid
search. The search must rank smooth scales using the deployed structure: a
high-precision low-rank branch plus an MXFP4-QDQ residual branch.

AutoRound must not import external compressor implementations or Nunchaku. The
algorithm and numerical data flow are implemented with native AutoRound
components.

## User-Facing Behavior

Smoothing remains opt-in:

```bash
--enable_svdquant_smooth \
--svdquant_smooth_num_grids 20
```

The Python API is:

```python
SVDQuantConfig(
    smooth_enabled=True,
    smooth_num_grids=20,
)
```

The fixed `--svdquant_smooth_alpha` option and `smooth_alpha` Python field are
removed without aliases. There is no fixed/search mode.

`smooth_num_grids` defaults to 20 and must be an integer of at least 2. Smaller
values may be used for smoke tests. It does not change the final low-rank outer
iteration count, which remains controlled by `residual_iters` and
`residual_early_stop`.

## Reference Semantics

The implementation uses these SVDQuant search settings:

```yaml
objective: OutputsError
strategy: GridSearch
granularity: Layer
spans: [[AbsMax, AbsMax]]
alpha: 0.5
beta: -2
num_grids: 20
allow_low_rank: true
fuse_when_possible: false
```

In GridSearch mode, the configured positive `alpha` selects a candidate family;
it is not a fixed alpha value. For `num_grids = N`, define:

```text
choices = [1/N, 2/N, ..., (N-1)/N]
```

The candidates are ordered as:

```text
(0, 0)
(a, 0)       for a in choices
(a, 1-a)     for a in choices
```

This yields `2N - 1` candidates, or 39 candidates when `N=20`.

## Scale Construction

For each smooth search group, collect the shared projection inputs and calculate
per-input-channel absolute maxima. A group contains one or more Linear weights
with the same input width and shared runtime scale:

```text
x_span[j] = max(abs(x[..., j]))
w_span[j] = max(abs(concat(W_group, dim=0)[:, j]))
```

For candidate `(alpha, beta)`:

```text
scale = x_span**alpha / w_span**beta
```

`(0, 0)` produces identity scale. Zero span entries are normalized to one, and
non-finite scales invalidate the candidate.

The equivalent transformed Linear is:

```text
x_hat = x / scale
W_hat = W * scale
```

AutoRound's runtime wrapper stores `smooth = 1 / scale`, preserving its current
forward and Nunchaku export direction.

## Search Groups And Output Objective

Layer granularity refers to the evaluation module surrounding a projection
group, not necessarily one `torch.nn.Linear`. For example, Q/K/V projections
that share an input also share a smooth scale and are evaluated by temporarily
patching the projections and running their attention parent. The objective is
the squared error of the parent module output.

Introduce a model-independent `SmoothSearchGroup` contract containing:

- one or more projection Linear modules and their global names;
- the shared projection-input cache key;
- the evaluation module;
- cached evaluation inputs and keyword arguments;
- output normalization needed for MSE; and
- optional output splits for a shared low-rank branch.

Model adapters discover these groups. The initial Flux adapter covers regular
QKV, added QKV, attention output, and feed-forward projection groups. Unknown
models fall back to one Linear per group with local Linear output evaluation;
this fallback preserves generality but is not claimed to reproduce a registered
model adapter's parent-output objective.

The quantization core depends only on the `SmoothSearchGroup` protocol, not on
Flux classes. Export adapters remain separate from calibration adapters.

## SVD-Aware Candidate Evaluation

Every candidate is evaluated with low-rank enabled. For candidate scale `s`:

1. Form every `W_hat_i = W_i * s` in the group.
2. Concatenate group weights along output channels and compute one shared
   rank-`rank` truncated SVD.
3. Cast the low-rank factors through `low_rank_dtype` to model deployment
   rounding.
4. Split the shared output factor back across the group's Linear modules and
   form each `R_i = W_hat_i - low_rank_i`.
5. Run every `R_i` through the same RTN QDQ function and resolved MXFP4 scheme
   used by final residual quantization.
6. Temporarily install the QDQ residuals, split low-rank branches, and input
   smoothing on the projection group.
7. Run the group's evaluation module with cached parent inputs and kwargs.
8. Compare its normalized output with the original floating-point parent output
   using layer-level squared error.

Candidate SVD/low-rank values are temporary and are discarded after scoring.
After selecting the best scale, AutoRound runs final residual iteration against
the selected grouped `W_hat` using the same shared-down/split-up structure.
Smooth calibration and final low-rank calibration remain separate phases.

The final residual iteration policy is not silently changed. Existing AutoRound
settings remain explicit and allow the caller to request
`residual_iters=100` plus early stop for a long quality-oriented run.

## Reuse Boundary

This is not an AWQ transform. Do not reuse AWQ's full-weight candidate evaluator,
activation-mean statistic, normalized-weight mean, scale folding, or clipping.

Reuse is limited to lower-level AutoRound facilities:

- terminal quantization scheme resolution;
- the same RTN/MXFP4 QDQ implementation used by residual iteration;
- block calibration scheduling and activation hooks;
- generic finite-error/best-candidate utilities when extraction reduces real
  duplication without coupling SVDQuant to `AWQTransform`.

AWQ behavior and public configuration remain unchanged.

## Calibration Lifetime

Projection inputs, parent evaluation inputs, kwargs, and floating-point outputs
are retained only for the current block. Capture is deterministic, CPU-backed
where practical, and bounded by explicit sample/token limits. Buffers and all
temporary module patches are released after scale selection and on every
exception path.

When smoothing is disabled, SVDQuant installs no calibration hooks and allocates
no input buffers.

## Selection And Failure Behavior

- Evaluate candidates in the exact order specified above.
- Select the first candidate with strictly lowest finite layer output error.
- A non-finite candidate is skipped with debug-level context.
- If no candidate is valid, fail with the global module name and reason.
- Missing calibration inputs are an error when smoothing is enabled; never
  silently fall back to identity or the removed fixed formula.
- Bias is never channel-scaled.

## Export Compatibility

Nunchaku tensor names, shapes, metadata, and runtime direction do not change.
The selected `smooth = 1 / scale` is exported through the existing adapter.
Residual MXFP4 tensors and BF16 low-rank branches use the existing exporter.

## Testing

Unit coverage must verify:

- fixed-alpha CLI/API removal and `smooth_num_grids` validation;
- exact candidate ordering and candidate count;
- Flux grouping for QKV, added QKV, attention output, and feed-forward paths;
- shared-down/split-up low-rank decomposition for multi-Linear groups;
- parent-module output MSE rather than isolated Linear MSE for registered groups;
- generic single-Linear fallback behavior;
- AbsMax/AbsMax scale construction against an independent reference;
- identity candidate behavior;
- one grouped truncated SVD call per candidate plus final configured iteration;
- low-rank dtype rounding is included in candidate error;
- residual candidate QDQ receives the resolved MXFP4 scheme;
- a constructed case selects the independently computed minimum-error candidate;
- first-candidate tie behavior;
- non-finite candidate skipping and all-candidate failure;
- missing calibration input failure;
- input buffer cleanup on success and failure;
- smoothing-disabled zero-overhead behavior;
- unchanged final wrapper and export contracts; and
- unchanged AWQ tests.

Integration validation uses a small AutoRound SVDQuant + MXFP4 QDQ model first,
then a one-sample/one-step FLUX smoke with a reduced grid. A quality run uses
`smooth_num_grids=20` and the requested final residual iteration settings.

## Out Of Scope

- Calling or importing another compressor implementation.
- Reusing AWQ full-weight search as a quality shortcut.
- AWQ clipping or upstream norm folding.
- Attention-specific y-x smoothing.
- Changing the Nunchaku kernel or file format.
- Automatically forcing final `residual_iters=100`.
