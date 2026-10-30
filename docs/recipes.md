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
print(plan.pipeline, plan.calibration)
configs = plan.build_configs()
```

`resolve_recipe()` returns an independent, processed `Recipe` copy. Each `build_configs()`
call creates fresh algorithm configs. A constructed compressor exposes its
recommendation as `compressor.resolved_recipe`; this is the selected recommendation,
not a replacement for the effective config after explicit overrides and validation.

For custom recipes, use `Recipe` and `AlgorithmSpec`. A recipe can provide an
ordinary Python function to adjust its defaults:

```python
from auto_round.recipes import AlgorithmSpec, Recipe, resolve_recipe


def configure(recipe, *, scheme, model_config):
    if scheme == "MXFP4":
        recipe.calibration["nsamples"] = 256
    if getattr(model_config, "model_type", None) == "some_model":
        recipe.layer_config["lm_head"] = "BF16"


custom = Recipe(
    "custom",
    pipeline=(AlgorithmSpec("quantizer", "auto_round", {"iters": 50}),),
    process=configure,
)
plan = resolve_recipe(custom, scheme="MXFP4")
```

This example illustrates the API, not a benchmarked recommendation. The function
edits an independent copy of the recipe and returns nothing. It can replace
`recipe.pipeline` directly or modify algorithm parameters, calibration, and layer
defaults using ordinary Python. There is no conditional patch language or merge
order to learn. Calibration keys are `dataset`, `seqlen`, `nsamples`, and `batch_size`.

Pass a `Recipe` directly to `AutoRound` or register it with `register_recipe` to
use its name in Python and an in-process CLI. Registration rejects duplicate names
and aliases. The resolver accepts `scheme` and `model_config` directly. Entry
points provide the model; configuration metadata is looked up for string inputs
only when a processing function exists. If lookup is unavailable, `model_config`
is `None`. No model weights are loaded during recipe processing.
Existing mixed-scheme handling remains in `auto_round/schemes.py`, and model layer
defaults remain in `auto_round/special_model_handler.py`. They run after model
loading for both Python and CLI. Shape restrictions and model compatibility
patches also remain outside recipes.

Future `ar-auto` recommendation strategies can use the same processing function.
`ar-auto` is not currently registered. AutoScheme remains
the independent mixed-bit scheme-selection feature.
