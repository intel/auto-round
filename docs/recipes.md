# Quantization recipes

Recipes are the shared recommendation source for the Python API and CLI.
Omitting `recipe` selects `ar`. A recipe chooses a complete algorithm pipeline,
calibration defaults, and optional per-layer recommendations. Scheme definitions
and algorithm implementations remain independent of recipes.

## Built-in plans

| Recipe | Initial pipeline | Iterations | Samples |
| --- | --- | --- | --- |
| `ar` | SignRound | 200 | 128 |
| `ar-light` | SignRound, learning rate 0.005 | 50 | 128 |
| `ar-best` | SignRound | 1000 | 512 |
| `rtn` | Plain RTN | — | 1 |
| `opt-rtn` | Optimized RTN | — | 128 |
| `awq` | AWQ smoothing → plain RTN | — | 128 |

All initial plans use sequence length 2048 and batch size 8. `default`, `light`,
`best`, and `opt_rtn` are aliases for `ar`, `ar-light`, `ar-best`, and `opt-rtn`.
The three `ar` tiers express cost/accuracy goals: their pipelines may gain
validated scheme-specific recommendations in future releases. They do not
guarantee an accuracy ordering on every model. Core calibration requirements
still apply, including static activation calibration for RTN when required.

## Usage and overrides

```python
from auto_round import AutoRound, SignRoundConfig

compressor = AutoRound("Qwen/Qwen3-0.6B", scheme="MXFP4", recipe="ar-best")

# Explicit values win even when equal to an older API default.
compressor = AutoRound("Qwen/Qwen3-0.6B", recipe="ar-best", nsamples=128)

# Replace the entire algorithm pipeline, retaining ar-best calibration defaults.
compressor = AutoRound("Qwen/Qwen3-0.6B", recipe="ar-best", alg_configs=SignRoundConfig(iters=20))
```

```bash
auto_round Qwen/Qwen3-0.6B --scheme MXFP4 --recipe ar-best
auto_round Qwen/Qwen3-0.6B --recipe ar-best --nsamples 128 --iters 20
auto_round Qwen/Qwen3-0.6B --recipe ar-best --algorithm awq
auto_round list recipes
auto_round list recipe awq
```

Explicit algorithm configs replace the recipe pipeline, rather than appending to
it. Their algorithm parameters are not overwritten by recipe stage parameters.
The deprecated `algorithm=` Python argument and CLI `--algorithm` retain their
existing algorithm-selection semantics. Explicit `iters=0` still selects RTN.
Explicit layer entries override recipe entries per pattern and per field; a
scalar scheme entry replaces the entire entry. Capability checks still reject
unsupported combinations regardless of recipe overrides.

`auto_round_best`, `auto_round_light`, `auto_round_rtn`, and `auto_round_opt_rtn`
select the corresponding recipe. Their hyphenated spellings work identically.
An explicit `--recipe` overrides the shortcut's default selection.

## Inspecting and extending recipes

```python
from auto_round.recipes import get_recipe, list_recipes, resolve_recipe

print(list_recipes())
definition = get_recipe("ar-best")
plan = resolve_recipe(definition)
print(plan.pipeline, plan.calibration, plan.applied_overlays)
configs = plan.build_configs()
```

Definitions and resolved plans are independent snapshots. Each `build_configs()`
call creates fresh algorithm configs. A constructed compressor exposes its
recommendation as `compressor.resolved_recipe`; this is the selected recommendation,
not a replacement for the effective config after explicit overrides and validation.

For custom recipes, use `Recipe`, `AlgorithmSpec`, `RecipeOverlay`, and
`RecipeContext` from `auto_round.recipes`. Pass a `Recipe` directly to `AutoRound`
or register it with `register_recipe` to use its name in Python and an in-process
CLI. Registration rejects duplicate names and aliases. Overlays run in declared
order; later matches override earlier ones. An overlay's `pipeline` replaces the
whole pipeline; `stage_params` patches stages by stable stage name and rejects
unknown stages. Calibration keys are `dataset`, `seqlen`, `nsamples`, and `batch_size`.

The public resolver accepts caller-provided model metadata without loading
models. Entry points supply metadata from already-loaded models and look up
configuration metadata for string model inputs only when overlays exist.
If configuration lookup is unavailable, the context has no model metadata.
Existing model-aware layer
policies still run after model loading, for both Python and CLI. Their definitions
live in `auto_round/recipes/layer_policies.py`, with compatibility adapters at the
original entry points. Shape restrictions and model compatibility patches remain
outside recipes.

Future `ar-auto` recommendation strategies can use this resolution boundary and
extend the context. `ar-auto` is not currently registered. AutoScheme remains
the independent mixed-bit scheme-selection feature.
