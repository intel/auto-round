from types import SimpleNamespace

import pytest

from auto_round import AutoRound
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig


def test_builtin_recipes():
    from auto_round.recipes import list_recipes, resolve_recipe

    assert {"ar", "ar-light", "ar-best", "rtn", "opt-rtn", "awq"} <= set(list_recipes())
    for name, iters, nsamples in [("ar", 200, 128), ("ar-light", 50, 128), ("ar-best", 1000, 512)]:
        plan = resolve_recipe(name)
        assert plan.build_configs()[0].iters == iters
        assert plan.calibration["nsamples"] == nsamples
    assert resolve_recipe("light").build_configs()[0].lr == 0.005
    assert resolve_recipe("rtn").build_configs()[0].disable_opt_rtn is True
    assert resolve_recipe("opt_rtn").build_configs()[0].disable_opt_rtn is False
    assert [type(config).__name__ for config in resolve_recipe("awq").build_configs()] == ["AWQConfig", "RTNConfig"]


def test_recipe_results_are_isolated():
    from auto_round.recipes import get_recipe, resolve_recipe

    definition = get_recipe("ar-best")
    definition.calibration["nsamples"] = 1
    plan = resolve_recipe("ar-best")
    plan.calibration["nsamples"] = 2
    plan.pipeline[0].params["iters"] = 3
    assert resolve_recipe("ar-best").calibration["nsamples"] == 512
    assert resolve_recipe("ar-best").build_configs()[0].iters == 1000


def test_context_overlays_replace_pipeline_then_patch_stages():
    from auto_round.recipes import AlgorithmSpec, Recipe, RecipeContext, RecipeOverlay, resolve_recipe

    recipe = Recipe(
        name="contextual",
        pipeline=(AlgorithmSpec("quantizer", "auto_round", {"iters": 200}),),
        overlays=(
            RecipeOverlay(
                "mxfp4",
                matches=lambda context: context.scheme == "MXFP4",
                pipeline=(AlgorithmSpec("smooth", "awq"), AlgorithmSpec("quantizer", "auto_round", {"iters": 50})),
            ),
            RecipeOverlay(
                "tiny-model",
                matches=lambda context: context.model_type == "tiny",
                stage_params={"quantizer": {"iters": 7}},
                calibration={"nsamples": 16},
                layer_config={"lm_head": "BF16"},
            ),
        ),
    )
    plan = resolve_recipe(recipe, RecipeContext(scheme="MXFP4", model_type="tiny"))
    assert [stage.algorithm for stage in plan.pipeline] == ["awq", "auto_round"]
    assert plan.build_configs()[1].iters == 7
    assert plan.calibration["nsamples"] == 16
    assert plan.layer_config == {"lm_head": "BF16"}
    assert plan.applied_overlays == ("mxfp4", "tiny-model")
    assert len(resolve_recipe(recipe).pipeline) == 1


def test_recipe_validation():
    from auto_round.recipes import AlgorithmSpec, Recipe, RecipeOverlay, resolve_recipe

    with pytest.raises(ValueError, match="Unknown recipe"):
        resolve_recipe("ar-bset")
    recipe = Recipe(
        "broken",
        (AlgorithmSpec("quantizer", "auto_round"),),
        overlays=(RecipeOverlay("bad", matches=lambda context: True, stage_params={"missing": {"iters": 1}}),),
    )
    with pytest.raises(ValueError, match="missing"):
        resolve_recipe(recipe)


@pytest.fixture
def capture_entry(monkeypatch):
    from auto_round import autoround

    def capture(model, scheme, configs, **kwargs):
        return SimpleNamespace(model=model, scheme=scheme, configs=configs, **kwargs)

    monkeypatch.setattr(autoround, "_CompressorBuilder", capture)


def test_default_entry_is_ar(capture_entry):
    default = AutoRound("dummy")
    explicit = AutoRound("dummy", recipe="ar")
    assert default.configs[0].iters == explicit.configs[0].iters == 200
    assert default.nsamples == explicit.nsamples == 128


def test_entry_recipe_and_explicit_calibration(capture_entry):
    entry = AutoRound("dummy", recipe="ar-best", nsamples=128, batch_size=1, seqlen=32)
    assert entry.configs[0].iters == 1000
    assert (entry.nsamples, entry.batch_size, entry.seqlen) == (128, 1, 32)
    assert AutoRound("dummy", recipe="ar-best").nsamples == 512


def test_explicit_pipeline_replaces_recipe_but_inherits_calibration(capture_entry):
    config = SignRoundConfig(iters=3, lr=0.25)
    entry = AutoRound("dummy", recipe="ar-best", alg_configs=config)
    assert entry.configs[0].iters == 3
    assert entry.configs[0].lr == 0.25
    assert config.iters == 3
    assert entry.nsamples == 512
    rtn = AutoRound("dummy", recipe="ar-best", alg_configs=RTNConfig(disable_opt_rtn=True))
    assert len(rtn.configs) == 1
    assert isinstance(rtn.configs[0], RTNConfig)
    assert not hasattr(rtn.configs[0], "iters")


def test_legacy_entry_selectors_override_recipe(capture_entry):
    assert isinstance(AutoRound("dummy", iters=0).configs[0], RTNConfig)
    awq = AutoRound("dummy", recipe="ar-best", algorithm="awq")
    assert [type(config).__name__ for config in awq.configs] == ["AWQConfig", "RTNConfig"]
    assert awq.nsamples == 512


def test_entry_applies_context_and_layer_overrides(capture_entry):
    from auto_round.recipes import AlgorithmSpec, Recipe, RecipeOverlay

    recipe = Recipe(
        "custom",
        (AlgorithmSpec("quantizer", "auto_round"),),
        overlays=(
            RecipeOverlay(
                "tiny", matches=lambda context: context.model_type == "tiny", layer_config={"proj": {"bits": 8}}
            ),
        ),
    )
    model = SimpleNamespace(config=SimpleNamespace(model_type="tiny"))
    entry = AutoRound(model, recipe=recipe, layer_config={"proj": {"bits": 4}})
    assert entry.layer_config["proj"]["bits"] == 4


@pytest.mark.parametrize("recipe", ["ar", "ar-light", "ar-best", "awq", "rtn", "opt-rtn"])
def test_cli_python_recipe_parity(recipe, capture_entry, monkeypatch):
    from auto_round.cli import main
    from auto_round.cli.algorithms import AlgorithmHandler

    captured = {}
    monkeypatch.setattr(main, "tune", lambda args: captured.setdefault("args", args))
    main.start(argv=["dummy", "--recipe", recipe])
    args = captured["args"]
    configs = AlgorithmHandler.build_configs(args, main._extract_common_quantization_kwargs(args))
    entry = AutoRound("dummy", recipe=recipe)
    assert [type(config) for config in configs] == [type(config) for config in entry.configs]
    assert args.nsamples == entry.nsamples
    for cli_config, api_config in zip(configs, entry.configs):
        for field in ("iters", "lr", "disable_opt_rtn"):
            assert getattr(cli_config, field, None) == getattr(api_config, field, None)


def test_cli_explicit_pipeline_and_shortcut_override(monkeypatch):
    from auto_round.cli import main
    from auto_round.cli.algorithms import AlgorithmHandler

    captured = {}
    monkeypatch.setattr(main, "tune", lambda args: captured.setdefault("args", args))
    main.start("best", argv=["dummy", "--recipe", "ar-light", "--algorithm", "awq", "--nsamples", "512"])
    args = captured["args"]
    configs = AlgorithmHandler.build_configs(args, {})
    assert args.recipe == "ar-light"
    assert args.nsamples == 512
    assert [type(config).__name__ for config in configs] == ["AWQConfig", "RTNConfig"]


def test_cli_explicit_values_override_recipe_stage_parameters(monkeypatch):
    from auto_round.cli import main
    from auto_round.cli.algorithms import AlgorithmHandler

    captured = {}
    monkeypatch.setattr(main, "tune", lambda args: captured.setdefault("args", args))
    main.start(argv=["dummy", "--recipe", "ar-best", "--iters", "50", "--disable_minmax_tuning"])
    config = AlgorithmHandler.build_configs(captured["args"], {})[0]
    assert config.iters == 50
    assert config.enable_minmax_tuning is False


def test_cli_recipe_listing(capsys):
    from auto_round.cli.main import list_item

    list_item(["recipes"])
    assert "ar-best" in capsys.readouterr().out
    list_item(["recipe", "awq"])
    output = capsys.readouterr().out
    assert "awq" in output and "rtn" in output


def test_cli_context_recipe_preserves_stage_defaults_and_layers(monkeypatch, capture_entry):
    from auto_round.cli import main
    from auto_round.cli.algorithms import AlgorithmHandler
    from auto_round.recipes import AlgorithmSpec, Recipe, RecipeOverlay, register_recipe, registry

    monkeypatch.setattr(registry, "_RECIPES", dict(registry._RECIPES))
    monkeypatch.setattr(registry, "_ALIASES", dict(registry._ALIASES))
    register_recipe(
        Recipe(
            "test-context",
            (AlgorithmSpec("quantizer", "auto_round"),),
            calibration={"nsamples": 32},
            overlays=(
                RecipeOverlay(
                    "mx",
                    matches=lambda context: context.scheme == "MXFP4",
                    stage_params={"quantizer": {"enable_minmax_tuning": False, "iters": 7}},
                    layer_config={"proj": "BF16"},
                ),
            ),
        )
    )
    captured = {}
    monkeypatch.setattr(main, "tune", lambda args: captured.setdefault("args", args))
    main.start(argv=["dummy", "--recipe", "test-context", "--scheme", "MXFP4"])
    config = AlgorithmHandler.build_configs(captured["args"], {})[0]
    assert config.enable_minmax_tuning is False
    assert config.iters == 7
    entry = AutoRound("dummy", recipe="test-context", scheme="MXFP4")
    assert entry.layer_config == {"proj": "BF16"}


def test_entry_keeps_recipe_resolution_for_inspection(capture_entry):
    entry = AutoRound("dummy", recipe="best")
    assert entry.resolved_recipe.name == "ar-best"


def test_cli_zero_iters_retains_rtn_search_options(monkeypatch):
    from auto_round.cli import main
    from auto_round.cli.algorithms import AlgorithmHandler

    captured = {}
    monkeypatch.setattr(main, "tune", lambda args: captured.setdefault("args", args))
    main.start(argv=["dummy", "--iters", "0", "--enable_neuqi", "--enable_opt_rtn"])
    config = AlgorithmHandler.build_configs(captured["args"], {})[0]
    assert isinstance(config, RTNConfig)
    assert config.enable_neuqi is True


def test_cli_explicit_signround_uses_config_defaults(monkeypatch):
    from auto_round.cli import main
    from auto_round.cli.algorithms import AlgorithmHandler

    captured = {}
    monkeypatch.setattr(main, "tune", lambda args: captured.setdefault("args", args))
    main.start(argv=["dummy", "--recipe", "ar-best", "--algorithm", "auto_round"])
    configs = AlgorithmHandler.build_configs(captured["args"], {})
    assert configs[0].iters == 200
    assert captured["args"].nsamples == 512


def test_recipe_context_looks_up_metadata_only_when_needed(monkeypatch, capture_entry):
    from transformers import AutoConfig

    from auto_round.recipes import AlgorithmSpec, Recipe, RecipeOverlay

    calls = []

    def load_config(name, **kwargs):
        calls.append(name)
        return SimpleNamespace(model_type="tiny")

    monkeypatch.setattr(AutoConfig, "from_pretrained", load_config)
    AutoRound("dummy")
    assert calls == []
    recipe = Recipe(
        "model-aware",
        (AlgorithmSpec("quantizer", "auto_round"),),
        overlays=(
            RecipeOverlay(
                "tiny",
                matches=lambda context: context.model_type == "tiny",
                stage_params={"quantizer": {"iters": 7}},
            ),
        ),
    )
    assert AutoRound("dummy", recipe=recipe).configs[0].iters == 7
    assert calls == ["dummy"]


def test_custom_recipe_inherits_shared_calibration_defaults():
    from auto_round.recipes import AlgorithmSpec, Recipe, resolve_recipe

    plan = resolve_recipe(Recipe("custom", (AlgorithmSpec("quantizer", "rtn"),)))
    assert plan.calibration == {"batch_size": 8, "seqlen": 2048, "nsamples": 128}


def test_entry_rejects_invalid_recipe_type(capture_entry):
    with pytest.raises(TypeError, match="recipe must"):
        AutoRound("dummy", recipe=None)


@pytest.mark.parametrize("entry_name,recipe_name", [("run_best", "ar-best"), ("run_light", "ar-light")])
def test_legacy_shortcuts_select_named_recipe(monkeypatch, entry_name, recipe_name):
    import sys

    from auto_round.cli import main

    captured = {}
    monkeypatch.setattr(main, "tune", lambda args: captured.setdefault("args", args))
    monkeypatch.setattr(sys, "argv", ["auto_round_shortcut", "dummy"])
    getattr(main, entry_name)()
    assert captured["args"].recipe == recipe_name


def test_callable_factory_defaults_survive_recipe_parameters(monkeypatch):
    from auto_round.algorithms.registry import get_algorithm_entry
    from auto_round.recipes import AlgorithmSpec, Recipe, resolve_recipe

    monkeypatch.setattr(
        get_algorithm_entry("auto_round"), "config_factory", lambda: SignRoundConfig(iters=17, momentum=0.25)
    )
    recipe = Recipe("factory", (AlgorithmSpec("quantizer", "auto_round", {"lr": 0.1}),))
    config = resolve_recipe(recipe).build_configs()[0]
    assert config.iters == 17
    assert config.momentum == 0.25
    assert config.lr == 0.1


def test_alias_factory_defaults_survive_recipe_and_cli_overrides(monkeypatch):
    from auto_round.cli import main
    from auto_round.cli.algorithms import AlgorithmHandler
    from auto_round.recipes import AlgorithmSpec, Recipe, register_recipe, registry, resolve_recipe

    monkeypatch.setattr(registry, "_RECIPES", dict(registry._RECIPES))
    monkeypatch.setattr(registry, "_ALIASES", dict(registry._ALIASES))
    recipe = Recipe(
        "random-rotation",
        (
            AlgorithmSpec("rotation", "random_hadamard", {"block_size": 32}),
            AlgorithmSpec("quantizer", "auto_round"),
        ),
    )
    register_recipe(recipe)
    assert resolve_recipe(recipe).build_configs()[0].hadamard_type == "random_hadamard"
    captured = {}
    monkeypatch.setattr(main, "tune", lambda args: captured.setdefault("args", args))
    main.start(argv=["dummy", "--recipe", recipe.name, "--rotation_block_size", "64", "--bits", "4"])
    configs = AlgorithmHandler.build_configs(captured["args"], {"bits": 4})
    assert configs[0].hadamard_type == "random_hadamard"
    assert configs[0].block_size == 64
    assert configs[1].bits == 4


def test_cli_recipe_respects_stage_common_argument_opt_out(monkeypatch):
    import auto_round.algorithms.registry as algorithms
    from auto_round.algorithms.config import AlgorithmConfig
    from auto_round.algorithms.registry import register_algorithm
    from auto_round.algorithms.transforms.base import BaseRotationConfig
    from auto_round.cli import main
    from auto_round.cli.algorithms import AlgorithmHandler
    from auto_round.recipes import AlgorithmSpec, Recipe, register_recipe, registry

    class StrictRotationConfig(AlgorithmConfig, BaseRotationConfig):
        cli_include_common_args = False

        def __init__(self):
            pass

    monkeypatch.setattr(registry, "_RECIPES", dict(registry._RECIPES))
    monkeypatch.setattr(registry, "_ALIASES", dict(registry._ALIASES))
    monkeypatch.setattr(algorithms, "_ALG_REGISTRY", dict(algorithms._ALG_REGISTRY))
    monkeypatch.setattr(algorithms, "_ALIAS_TO_NAME", dict(algorithms._ALIAS_TO_NAME))
    register_algorithm("strict-rotation", config_factory=StrictRotationConfig)
    register_recipe(
        Recipe(
            "strict-pipeline",
            (AlgorithmSpec("rotation", "strict-rotation"), AlgorithmSpec("quantizer", "auto_round")),
        )
    )
    captured = {}
    monkeypatch.setattr(main, "tune", lambda args: captured.setdefault("args", args))
    main.start(argv=["dummy", "--recipe", "strict-pipeline", "--bits", "4", "--group_size", "32"])
    configs = AlgorithmHandler.build_configs(captured["args"], {"bits": 4, "group_size": 32})
    assert isinstance(configs[0], StrictRotationConfig)
    assert configs[1].bits == 4
    assert configs[1].group_size == 32


def test_direct_tune_applies_recipe_before_building_configs(monkeypatch):
    from auto_round.cli import main
    from auto_round.cli.algorithms import AlgorithmHandler
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(["dummy", "--recipe", "ar-best", "--nsamples", "128"])

    def check_configs(args, common_kwargs):
        assert args.recipe == "ar-best"
        assert args.nsamples == 128
        assert args.batch_size == 8
        assert args.seqlen == 2048
        assert args._recipe_plan.build_configs()[0].iters == 1000
        raise StopIteration("configuration inspected")

    monkeypatch.setattr(AlgorithmHandler, "build_configs", check_configs)
    with pytest.raises(StopIteration, match="configuration inspected"):
        main.tune(args)


def test_dataset_omission_inherits_recipe_but_explicit_none_is_deferred(capture_entry):
    from auto_round.recipes import AlgorithmSpec, Recipe

    recipe = Recipe(
        "dataset", (AlgorithmSpec("quantizer", "auto_round"),), calibration={"dataset": "custom-calibration"}
    )
    assert AutoRound("dummy", recipe=recipe).dataset == "custom-calibration"
    assert AutoRound("dummy", recipe=recipe, dataset=None).dataset is None
    assert AutoRound("dummy").dataset is None


def test_callable_factory_recipe_still_validates_overrides(monkeypatch):
    from auto_round.algorithms.registry import get_algorithm_entry
    from auto_round.recipes import AlgorithmSpec

    entry = get_algorithm_entry("hadamard")
    monkeypatch.setattr(entry, "config_factory", lambda: entry.alias_factories["random_hadamard"]())
    with pytest.raises(ValueError):
        AlgorithmSpec("rotation", "hadamard", {"backend": "invalid"}).build_config()
