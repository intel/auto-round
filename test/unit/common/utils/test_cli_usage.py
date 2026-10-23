import argparse
import inspect
import json
import shutil
import sys
from test.helpers import get_model_path

import pytest

from auto_round.cli.main import run, run_light


def _assert_cli_ok(monkeypatch, argv, entry=run):
    """Run an in-process CLI entry point and assert it succeeds.

    ``argv`` is the full argv (including argv[0]); it replaces ``sys.argv`` so
    the entry point parses it exactly like a real command line invocation.
    Help/eval paths exit via argparse with code 0, while quantization paths
    return normally. Any non-zero ``SystemExit`` indicates a CLI failure.
    """
    monkeypatch.setattr(sys, "argv", list(argv))
    try:
        entry()
    except SystemExit as exc:  # argparse help/version exits with code 0
        assert exc.code in (0, None), f"cmd line test fail, exit code {exc.code}"


class TestAutoRoundCmd:

    @pytest.fixture(autouse=True)
    def setup_save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)
        shutil.rmtree("../../tmp_autoround", ignore_errors=True)

    def test_auto_round_cmd(self, monkeypatch):
        _assert_cli_ok(monkeypatch, ["auto_round", "-h"])

    def test_auto_round_cmd2(self, monkeypatch, tiny_opt_model_path, tmp_path):
        calibration_path = tmp_path / "calibration.json"
        calibration_path.write_text(json.dumps(["AutoRound CLI local calibration sample. " * 32]))

        _assert_cli_ok(
            monkeypatch,
            [
                "auto_round",
                "--model",
                tiny_opt_model_path,
                "--seqlen",
                "32",
                "--iter",
                "2",
                "--nsamples",
                "1",
                "--dataset",
                str(calibration_path),
                "--format",
                "auto_gptq,auto_round",
                "--output_dir",
                self.save_dir,
            ],
        )

    def test_auto_round_cmd3_routes_eval_task_by_task_without_quantizing(self, monkeypatch):
        from auto_round.cli import main as cli_main

        captured = {}

        def fake_start(argv=None):
            captured["argv"] = argv

        monkeypatch.setattr(cli_main, "start", fake_start)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "auto_round",
                "--model",
                "dummy-model",
                "--seqlen",
                "8",
                "--iter",
                "1",
                "--nsamples",
                "1",
                "--eval_task_by_task",
                "--tasks",
                "openbookqa",
                "--bs",
                "32",
                "--limit",
                "2",
            ],
        )
        cli_main.run()

        assert captured["argv"][-7:] == [
            "--eval_task_by_task",
            "--tasks",
            "openbookqa",
            "--bs",
            "32",
            "--limit",
            "2",
        ]

    def test_auto_round_cmd4_applies_light_recipe(self, monkeypatch):
        from auto_round.cli import main as cli_main

        captured = {}
        monkeypatch.setattr(cli_main, "tune", lambda args: captured.setdefault("args", args))
        monkeypatch.setattr(sys, "argv", ["auto-round-light", "--model", "dummy-model"])

        run_light()

        args = captured["args"]
        assert args.iters == 50
        assert args.seqlen == 2048
        assert args.nsamples == 128

    def test_auto_round_cmd5(self, monkeypatch):
        _assert_cli_ok(monkeypatch, ["auto_round", "--eval", "-h"])

    def test_auto_round_cmd6(self, monkeypatch):
        _assert_cli_ok(monkeypatch, ["auto_round", "--eval", "--lmms", "-h"])

    def test_auto_round_cmd7_routes_mllm_without_quantizing(self, monkeypatch):
        from auto_round.cli import main as cli_main

        captured = {}
        monkeypatch.setattr(cli_main, "tune", lambda args: captured.setdefault("args", args))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "auto_round",
                "--mllm",
                "--model",
                "dummy-model",
                "--iter",
                "2",
                "--nsamples",
                "2",
                "--seqlen",
                "32",
                "--format",
                "auto_round",
            ],
        )

        run()

        args = captured["args"]
        assert args.model_name == "dummy-model"
        assert args.mllm is True
        assert args.iters == 2

    def test_auto_round_cmd8_routes_quant_nontext_module_without_quantizing(self, monkeypatch):
        from auto_round.cli import main as cli_main

        captured = {}

        def fake_start(argv=None):
            captured["argv"] = argv

        monkeypatch.setattr(cli_main, "start", fake_start)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "auto_round",
                "--mllm",
                "--iter",
                "2",
                "--nsamples",
                "2",
                "--model",
                "dummy-model",
                "--seqlen",
                "32",
                "--format",
                "auto_round",
                "--quant_nontext_module",
                "--output_dir",
                self.save_dir,
            ],
        )
        cli_main.run()

        assert "--quant_nontext_module" in captured["argv"]
        assert "--mllm" in captured["argv"]

    def test_layer_config(self):
        """Test --layer_config parsing without starting a quantization run."""
        from auto_round.cli.parser import build_quantize_parser

        layer_cfg = r"{fc1:{bits:8,data_type:int},fc2:{bits:16,data_type:int}}"
        args = build_quantize_parser().parse_args(
            [
                "--model",
                "dummy-model",
                "--seqlen",
                "8",
                "--iter",
                "0",
                "--disable_opt_rtn",
                "--layer_config",
                layer_cfg,
                "--format",
                "auto_round",
                "--output_dir",
                self.save_dir,
            ]
        )
        assert args.layer_config == layer_cfg


def test_diffusion_quantize_cli_keeps_inference_steps_separate():
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(
        [
            "--model",
            "dummy-model",
            "--calib_num_inference_steps",
            "6",
            "--num_inference_steps",
            "12",
        ]
    )

    assert args.calib_num_inference_steps == 6
    assert args.num_inference_steps == 12


def test_run_rtn_uses_zero_shot_recipe(monkeypatch):
    from auto_round.cli import main as cli_main

    captured = {}

    def fake_tune(args):
        captured["args"] = args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "auto_round_rtn",
            "--model",
            "dummy-model",
        ],
    )
    monkeypatch.setattr(cli_main, "tune", fake_tune)

    cli_main.run_rtn()

    args = captured["args"]
    assert args.model_name == "dummy-model"
    assert args.iters == 0
    assert args.disable_opt_rtn is True
    assert args.batch_size == 8
    assert args.nsamples == 1


def test_run_rtn_preserves_eval_args(monkeypatch, tmp_path):
    from auto_round.cli import main as cli_main

    captured = {}

    def fake_tune(args):
        captured["args"] = args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "auto_round_rtn",
            "--model",
            "dummy-model",
            "--tasks",
            "mmlu",
            "--format",
            "fake",
            "--output_dir",
            str(tmp_path / "out"),
            "--eval_model_dtype",
            "bf16",
        ],
    )
    monkeypatch.setattr(cli_main, "tune", fake_tune)

    cli_main.run_rtn()

    args = captured["args"]
    assert args.tasks == "mmlu"
    assert args.format == "fake"
    assert args.eval_model_dtype == "bf16"
    assert args.output_dir == str(tmp_path / "out")
    assert args.iters == 0
    assert args.disable_opt_rtn is True


def test_run_opt_rtn_uses_recipe(monkeypatch):
    from auto_round.cli import main as cli_main

    captured = {}

    def fake_tune(args):
        captured["args"] = args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "auto_round_opt_rtn",
            "--model",
            "dummy-model",
        ],
    )
    monkeypatch.setattr(cli_main, "tune", fake_tune)

    cli_main.run_opt_rtn()

    args = captured["args"]
    assert args.model_name == "dummy-model"
    assert args.iters == 0
    assert args.disable_opt_rtn is False
    assert args.batch_size == 8
    assert args.nsamples == 128


def test_model_free_cli_disable_opt_rtn(monkeypatch):
    from auto_round import AutoRound
    from auto_round.cli import main as cli_main
    from auto_round.cli.algorithms import AlgorithmHandler

    captured = {}
    monkeypatch.setattr(cli_main, "tune", lambda args: captured.setdefault("args", args))

    cli_main.start(
        argv=[
            "dummy-model",
            "--scheme",
            "NVFP4_E5M3",
            "--model_free",
            "--disable_opt_rtn",
        ]
    )

    args = captured["args"]
    configs = AlgorithmHandler.build_configs(args, cli_main._extract_common_quantization_kwargs(args))
    compressor = AutoRound(
        args.model,
        scheme=args.scheme,
        alg_configs=configs[-1],
        model_free=args.model_free,
        device_map="cpu",
        enable_torch_compile=False,
    )

    assert args.model_free is True
    assert args.iters == 0
    assert args.disable_opt_rtn is True
    assert configs[-1].disable_opt_rtn is True
    assert type(compressor).__name__ == "ModelFreeCompressor"
    assert compressor.disable_opt_rtn is True


def test_unknown_algorithm_help_exits_with_suggestion(monkeypatch):
    from auto_round.cli import main as cli_main

    monkeypatch.setattr(sys, "argv", ["auto_round", "--algorithm", "hadarmard", "--help"])

    with pytest.raises(SystemExit, match="Unknown algorithm 'hadarmard'. Did you mean 'hadamard'\\?"):
        cli_main.run()


def test_legacy_disable_flags_map_to_enable_bools():
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(
        [
            "--model",
            "dummy-model",
            "--disable_minmax_tuning",
            "--disable_quanted_input",
        ]
    )

    assert args.enable_minmax_tuning is False
    assert args.enable_quanted_input is False


@pytest.mark.parametrize("model_adapter", ["flux", "sdxl"])
def test_svdquant_cli_builds_hyphenated_options_before_rtn(model_adapter):
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
    from auto_round.cli.algorithms import AlgorithmHandler
    from auto_round.cli.parser import build_quantize_parser

    parser = build_quantize_parser()
    args = parser.parse_args(
        [
            "--model",
            "dummy-model",
            "--algorithm",
            "svdquant,rtn",
            "--svdquant-rank",
            "16",
            "--enable-svdquant-smooth",
            "--svdquant-smooth-num-grids",
            "39",
            "--svdquant-smooth-max-calibration-calls",
            "64",
            "--svdquant-residual-iters",
            "20",
            "--enable-svdquant-residual-early-stop",
            "--svdquant-low-rank-dtype",
            "fp32",
            "--svdquant-target-modules",
            "attn,ff",
            "--svdquant-exclude-modules",
            "proj_out",
            "--svdquant-model-adapter",
            model_adapter,
            "--disable_opt_rtn",
        ]
    )

    configs = AlgorithmHandler.build_configs(args, {})

    assert isinstance(configs[0], SVDQuantConfig)
    assert configs[0].rank == 16
    assert configs[0].smooth_enabled is True
    assert configs[0].smooth_num_grids == 39
    assert configs[0].smooth_max_calibration_calls == 64
    assert configs[0].residual_iters == 20
    assert configs[0].residual_early_stop is True
    assert configs[0].low_rank_dtype == "fp32"
    assert configs[0].target_modules == ["attn", "ff"]
    assert configs[0].exclude_modules == ["proj_out"]
    assert configs[0].model_adapter == model_adapter
    assert configs[1].__class__.__name__ == "RTNConfig"


def test_svdquant_cli_rejects_underscore_option_aliases():
    from auto_round.cli.parser import build_quantize_parser

    with pytest.raises(SystemExit):
        build_quantize_parser().parse_args(
            ["--model", "dummy-model", "--algorithm", "svdquant,rtn", "--svdquant_rank", "16"]
        )


def test_cli_fallback_derives_arguments_for_config_without_register_args(monkeypatch):
    import auto_round.cli.algorithms as algorithm_cli
    from auto_round.algorithms.config import AlgorithmConfig
    from auto_round.algorithms.registry import AlgRegistryEntry
    from auto_round.cli.algorithms import AlgorithmHandler
    from auto_round.cli.parser import build_quantize_parser

    class FallbackConfig(AlgorithmConfig):
        def __init__(self, *, iters: int = 3, enabled: bool = False):
            self.iters = iters
            self.enabled = enabled

    monkeypatch.setattr(
        algorithm_cli,
        "iter_algorithm_entries",
        lambda: [AlgRegistryEntry(name="fallback", config_factory=FallbackConfig)],
    )

    parser = build_quantize_parser()
    args = parser.parse_args(["--model", "dummy-model", "--algorithm", "fallback", "--iters", "7", "--enabled"])

    assert args.iters == 7
    assert args.enabled is True
    assert algorithm_cli._parameter_registry(FallbackConfig).parameters


def test_cli_explicit_registration_takes_precedence_over_constructor_fallback():
    from auto_round.algorithms.config import AlgorithmConfig
    from auto_round.cli.algorithms import _parameter_registry

    class ExplicitConfig(AlgorithmConfig):
        def __init__(self, *, internal_value: int = 1):
            self.internal_value = internal_value

        @classmethod
        def register_args(cls, registry):
            registry.add_argument("--public-value", field="internal_value", type=int)

    parameters = _parameter_registry(ExplicitConfig).parameters

    assert [parameter.option_strings for parameter in parameters] == [("--public-value",)]
    assert [parameter.field for parameter in parameters] == ["internal_value"]


def test_cli_shared_fallback_argument_is_routed_to_each_selected_config(monkeypatch):
    import auto_round.cli.algorithms as algorithm_cli
    from auto_round.algorithms.config import AlgorithmConfig
    from auto_round.algorithms.registry import AlgRegistryEntry
    from auto_round.cli.algorithms import AlgorithmHandler

    class FirstConfig(AlgorithmConfig):
        def __init__(self, *, iters: int = 3):
            self.iters = iters

    class SecondConfig(AlgorithmConfig):
        def __init__(self, *, iters: int = 5):
            self.iters = iters

    monkeypatch.setattr(
        algorithm_cli,
        "iter_algorithm_entries",
        lambda: [
            AlgRegistryEntry(name="first", config_factory=FirstConfig),
            AlgRegistryEntry(name="second", config_factory=SecondConfig),
        ],
    )

    parser = argparse.ArgumentParser()
    AlgorithmHandler.add_groups(parser)
    args = parser.parse_args(["--iters", "11"])

    assert algorithm_cli._parameter_registry(FirstConfig).config_kwargs(args) == {"iters": 11}
    assert algorithm_cli._parameter_registry(SecondConfig).config_kwargs(args) == {"iters": 11}


def test_build_configs_routes_shared_argument_to_selected_configs(monkeypatch):
    import auto_round.cli.algorithms as algorithm_cli
    from auto_round.algorithms.config import AlgorithmConfig
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
    from auto_round.algorithms.registry import AlgRegistryEntry
    from auto_round.cli.algorithms import AlgorithmHandler

    class SharedConfig(AlgorithmConfig):
        def __init__(self, *, iters: int = 5):
            self.iters = iters

    entries = {
        "auto_round": AlgRegistryEntry(name="auto_round", config_factory=SignRoundConfig),
        "shared": AlgRegistryEntry(name="shared", config_factory=SharedConfig),
    }
    monkeypatch.setattr(algorithm_cli, "get_algorithm_entry", entries.__getitem__)
    monkeypatch.setattr(algorithm_cli, "resolve_algorithm_names", lambda names, **_: list(names))

    args = argparse.Namespace(algorithm="auto_round,shared", iters=11, rotation_hadamard_type=None)
    configs = AlgorithmHandler.build_configs(args, {})

    assert isinstance(configs[0], SignRoundConfig)
    assert configs[0].iters == 11
    assert isinstance(configs[1], SharedConfig)
    assert configs[1].iters == 11


def test_cli_rejects_incompatible_shared_fallback_argument(monkeypatch):
    import auto_round.cli.algorithms as algorithm_cli
    from auto_round.algorithms.config import AlgorithmConfig
    from auto_round.algorithms.registry import AlgRegistryEntry
    from auto_round.cli.algorithms import AlgorithmHandler

    class IntegerConfig(AlgorithmConfig):
        def __init__(self, *, value: int = 1):
            self.value = value

    class StringConfig(AlgorithmConfig):
        def __init__(self, *, value: str = "value"):
            self.value = value

    monkeypatch.setattr(
        algorithm_cli,
        "iter_algorithm_entries",
        lambda: [
            AlgRegistryEntry(name="integer", config_factory=IntegerConfig),
            AlgRegistryEntry(name="string", config_factory=StringConfig),
        ],
    )

    with pytest.raises(ValueError, match="incompatible shared CLI argument.*--value"):
        AlgorithmHandler.add_groups(argparse.ArgumentParser())


def test_registered_cli_fields_are_accepted_by_config_constructors():
    import auto_round.cli.algorithms as algorithm_cli
    from auto_round.algorithms.registry import iter_algorithm_entries

    for entry in iter_algorithm_entries():
        if entry.config_factory is None:
            continue
        config_cls = entry.config_factory if isinstance(entry.config_factory, type) else type(entry.config_factory())
        accepted = set(inspect.signature(config_cls.__init__).parameters)
        accepted.discard("self")
        accepted.update(getattr(config_cls, "model_fields", {}))
        assert all(
            parameter.field in accepted for parameter in algorithm_cli._parameter_registry(config_cls).parameters
        )


def test_svdquant_cli_defaults_compose_before_signround():
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
    from auto_round.cli.algorithms import AlgorithmHandler
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(
        ["--model", "dummy-model", "--algorithm", "svdquant,auto_round", "--iters", "200"]
    )

    configs = AlgorithmHandler.build_configs(args, {})

    assert isinstance(configs[0], SVDQuantConfig)
    assert configs[0].rank == 32
    assert configs[0].smooth_enabled is False
    assert configs[0].residual_iters == 1
    assert configs[0].model_adapter == "auto"
    assert isinstance(configs[1], SignRoundConfig)


def _normalize_shared_layers(raw):
    if raw is None:
        return None
    normalized_groups = []
    for invocation in raw:
        if any("," in token for token in invocation):
            for token in invocation:
                group = [p.strip() for p in token.split(",") if p.strip()]
                if group:
                    normalized_groups.append(group)
        else:
            group = [p.strip() for p in invocation if p.strip()]
            if group:
                normalized_groups.append(group)
    return normalized_groups or None


def test_schemes_comma_space_separated():
    """--schemes accepts comma-separated and space-separated values."""
    from auto_round.cli.main import _normalize_scheme_list
    from auto_round.cli.parser import build_quantize_parser

    p = build_quantize_parser()
    assert _normalize_scheme_list(p.parse_args(["--bits", "4", "--schemes", "W4A16,W8A16"]).schemes) == "W4A16,W8A16"
    assert _normalize_scheme_list(p.parse_args(["--bits", "4", "--schemes", "W4A16", "W8A16"]).schemes) == "W4A16,W8A16"
    assert p.parse_args(["--model", "dummy"]).schemes is None


def test_legacy_auto_scheme_flags_alias_and_hidden():
    """--options/--avg_bits/--target_bits alias --schemes/--bits but are hidden from --help."""
    import io

    from auto_round.cli.main import _normalize_scheme_list
    from auto_round.cli.parser import build_quantize_parser

    p = build_quantize_parser()
    # legacy flags store into the canonical dests (bits / schemes)
    args = p.parse_args(["--avg_bits", "4", "--options", "W4A16,W8A16"])
    assert args.bits == 4.0
    assert _normalize_scheme_list(args.schemes) == "W4A16,W8A16"
    args = p.parse_args(["--target_bits", "3.5", "--option", "W2A16", "W4A16"])
    assert args.bits == 3.5
    assert _normalize_scheme_list(args.schemes) == "W2A16,W4A16"
    # singular aliases
    assert p.parse_args(["--bit", "8"]).bits == 8.0

    help_text = io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(help_text):
        p.print_help()
    help_text = help_text.getvalue()
    for flag in ("--options", "--option", "--avg_bits", "--target_bits"):
        assert flag not in help_text
    for flag in ("--schemes", "--bits", "--bit"):
        assert flag in help_text


def test_shared_layers_normalize():
    """--shared_layers: bare tokens → one group; comma tokens → one group each; a,b c,d → two groups."""
    from auto_round.cli.parser import build_quantize_parser

    p = build_quantize_parser()
    # bare tokens → one group
    assert _normalize_shared_layers(
        p.parse_args(["--model", "dummy", "--shared_layers", "l1", "l2"]).shared_layers
    ) == [["l1", "l2"]]
    # comma token → one group
    assert _normalize_shared_layers(p.parse_args(["--model", "dummy", "--shared_layers", "l1,l2"]).shared_layers) == [
        ["l1", "l2"]
    ]
    # a,b c,d → two groups (replaces --shared_layers a,b --shared_layers c,d)
    assert _normalize_shared_layers(
        p.parse_args(["--model", "dummy", "--shared_layers", "l1,l2", "l3,l4"]).shared_layers
    ) == [["l1", "l2"], ["l3", "l4"]]
    # multiple flags → multiple groups
    assert _normalize_shared_layers(
        p.parse_args(["--model", "dummy", "--shared_layers", "l1", "l2", "--shared_layers", "l3,l4"]).shared_layers
    ) == [["l1", "l2"], ["l3", "l4"]]
    assert p.parse_args(["--model", "dummy"]).shared_layers is None


def test_parse_max_shard_size():
    from auto_round.cli.parser import build_quantize_parser

    parser = build_quantize_parser()
    assert parser.parse_args([]).max_shard_size is None
    assert parser.parse_args(["--max_shard_size", "1GB"]).max_shard_size == "1GB"


# ---------------------------------------------------------------------------
# `auto_round_v2` and the deprecated `--enable_alg_ext`
# ---------------------------------------------------------------------------


def _build_cli_configs(argv, **overrides):
    """Build algorithm configs exactly like the quantize CLI does."""
    from auto_round.cli.algorithms import AlgorithmHandler
    from auto_round.cli.main import _extract_common_quantization_kwargs
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(["--model", "dummy-model"] + list(argv))
    if getattr(args, "iters", None) is None:
        args.iters = 200
    for key, value in overrides.items():
        setattr(args, key, value)
    return AlgorithmHandler.build_configs(args, _extract_common_quantization_kwargs(args))


def test_cli_auto_round_v2_selects_a_single_signround_v2_config():
    from auto_round.algorithms.quantization.sign_round.config import SignRoundV2Config

    configs = _build_cli_configs(["--algorithm", "auto_round_v2"])

    # auto_round_v2 must not drag in the implicit `auto_round` fallback, which
    # would chain two block quantizers back to back.
    assert [type(config).__name__ for config in configs] == ["SignRoundV2Config"]
    assert isinstance(configs[0], SignRoundV2Config)
    assert configs[0].enable_alg_ext is True


@pytest.mark.parametrize("alias", ["auto_round_v2", "auto_roundv2", "sign_round_v2", "signroundv2"])
def test_auto_round_v2_aliases_resolve_to_the_new_algorithm(alias):
    from auto_round.cli.algorithms import AlgorithmHandler

    assert AlgorithmHandler.resolve_alias(alias) == "auto_round_v2"


def test_cli_auto_round_v2_composes_with_awq():
    configs = _build_cli_configs(["--algorithm", "awq,auto_round_v2"])

    assert [type(config).__name__ for config in configs] == ["AWQConfig", "SignRoundV2Config"]


def test_cli_auto_round_v2_degrades_to_rtn_without_iters():
    configs = _build_cli_configs(["--algorithm", "auto_round_v2", "--iters", "0"])

    assert [type(config).__name__ for config in configs] == ["RTNConfig"]


def test_deprecated_enable_alg_ext_still_selects_the_v2_algorithm():
    from auto_round.algorithms.quantization.sign_round.config import SignRoundV2Config
    from auto_round.algorithms.registry import normalize_algorithm_config

    configs = _build_cli_configs(["--enable_alg_ext"])

    assert [type(config).__name__ for config in configs] == ["SignRoundConfig"]
    assert configs[0].enable_alg_ext is True
    # The entry point rewrites SignRoundConfig + enable_alg_ext into the v2 config,
    # which is what keeps the deprecated switch equivalent to the new spelling.
    assert isinstance(normalize_algorithm_config(configs[0]), SignRoundV2Config)


def test_v2_config_keeps_the_extension_enabled_however_it_is_built():
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig, SignRoundV2Config
    from auto_round.algorithms.registry import normalize_algorithm_config

    # normalize_algorithm_config rewrites __class__ without calling __init__.
    coerced = normalize_algorithm_config(SignRoundConfig(iters=1, enable_alg_ext=True))
    assert isinstance(coerced, SignRoundV2Config)
    assert coerced.enable_alg_ext is True

    # A direct assignment must not be able to switch the extension off either:
    # the SignRoundV2 quantizer branches on this flag.
    direct = SignRoundV2Config(iters=1)
    direct.enable_alg_ext = False
    assert direct.enable_alg_ext is True


def test_enable_alg_ext_is_advertised_as_deprecated_in_help():
    from auto_round.cli.algorithms import AlgorithmHandler

    v1_help = AlgorithmHandler.format_detail("auto_round")
    assert "--enable_alg_ext" in v1_help
    assert "Deprecated" in v1_help
    assert "auto_round_v2" in v1_help

    # v2 always has the extension on, so the switch must not be offered again
    # with a help text telling the user to select `--algorithm auto_round_v2`.
    v2_help = AlgorithmHandler.format_detail("auto_round_v2")
    assert "--enable_alg_ext" not in v2_help
    assert "--iters" in v2_help


def test_list_alg_includes_auto_round_v2():
    from auto_round.cli.algorithms import AlgorithmHandler

    assert "auto_round_v2" in AlgorithmHandler.format_listing()


# ---------------------------------------------------------------------------
# `rtn` / `opt-rtn` plain-vs-optimized RTN
# ---------------------------------------------------------------------------


def test_cli_rtn_algorithm_selects_plain_rtn():
    from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
    from auto_round.algorithms.registry import normalize_algorithm_config

    configs = _build_cli_configs(["--alg", "rtn"])

    assert [type(config).__name__ for config in configs] == ["RTNConfig"]
    assert configs[0].disable_opt_rtn is True
    # An explicitly forced plain RTN must not be re-normalized into the optimized one.
    assert type(normalize_algorithm_config(configs[0])) is RTNConfig
    assert not isinstance(configs[0], OptimizedRTNConfig)


@pytest.mark.parametrize("alias", ["opt-rtn", "opt_rtn", "optimized_rtn", "optimized-rtn"])
def test_cli_opt_rtn_algorithm_selects_the_optimized_path(alias):
    from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig
    from auto_round.algorithms.registry import normalize_algorithm_config

    configs = _build_cli_configs(["--alg", alias])

    # Exactly one block quantizer: opt-rtn must not append the implicit fallback.
    assert [type(config).__name__ for config in configs] == ["OptimizedRTNConfig"]
    assert isinstance(configs[0], OptimizedRTNConfig)
    assert configs[0].disable_opt_rtn is False
    assert configs[0].orig_disable_opt_rtn is False
    assert isinstance(normalize_algorithm_config(configs[0]), OptimizedRTNConfig)


@pytest.mark.parametrize(
    ("argv", "expected_disable_opt_rtn"),
    [
        (["--alg", "rtn", "--enable_opt_rtn"], True),
        (["--alg", "opt-rtn", "--disable_opt_rtn"], False),
    ],
)
def test_rtn_algorithm_name_wins_over_a_contradicting_flag(argv, expected_disable_opt_rtn):
    configs = _build_cli_configs(argv)

    assert len(configs) == 1
    assert configs[0].disable_opt_rtn is expected_disable_opt_rtn
    # `orig_disable_opt_rtn` marks the value as user policy, which is what stops
    # the W8A16/W8A8 efficiency heuristic and the AWQ policy from overriding it.
    assert configs[0].orig_disable_opt_rtn is expected_disable_opt_rtn


def test_cli_opt_rtn_composes_with_awq_without_extra_configs():
    from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig

    configs = _build_cli_configs(["--alg", "awq,opt-rtn"])

    assert [type(config).__name__ for config in configs] == ["AWQConfig", "OptimizedRTNConfig"]
    assert isinstance(configs[1], OptimizedRTNConfig)
    assert configs[1].disable_opt_rtn is False


@pytest.mark.parametrize(
    ("overrides", "expected_type"),
    [
        ({}, "RTNConfig"),
        ({"disable_opt_rtn": True}, "RTNConfig"),
        ({"disable_opt_rtn": False}, "RTNConfig"),
    ],
)
def test_implicit_rtn_path_keeps_flag_and_heuristic_behaviour(overrides, expected_type):
    """Without an explicit `--algorithm`, RTN must not be forced to plain.

    This is what keeps `auto-round-rtn` / `auto-round-opt-rtn` and the plain
    `--iters 0` path behaving exactly as before.
    """
    from auto_round.algorithms.registry import normalize_algorithm_config

    configs = _build_cli_configs(["--iters", "0"], **overrides)

    assert [type(config).__name__ for config in configs] == [expected_type]
    if overrides:
        # An explicit flag is user policy, exactly as before.
        assert configs[0].disable_opt_rtn is overrides["disable_opt_rtn"]
        assert configs[0].orig_disable_opt_rtn is overrides["disable_opt_rtn"]
    else:
        # No flag: `orig` stays None so the engine keeps owning the decision
        # (heuristic default for a W4A16 int scheme is the optimized path).
        assert configs[0].disable_opt_rtn is False
        assert configs[0].orig_disable_opt_rtn is None
    assert type(normalize_algorithm_config(configs[0])).__name__ == (
        "OptimizedRTNConfig" if configs[0].disable_opt_rtn is False else "RTNConfig"
    )


def test_list_alg_includes_opt_rtn():
    from auto_round.cli.algorithms import AlgorithmHandler

    listing = AlgorithmHandler.format_listing()

    assert "- rtn" in listing
    assert "opt-rtn" in listing


def test_opt_rtn_help_lists_the_rtn_switches():
    from auto_round.cli.algorithms import AlgorithmHandler

    help_text = AlgorithmHandler.format_detail("opt-rtn")

    assert "--disable_opt_rtn" in help_text
    assert "--enable_opt_rtn" in help_text


# ---------------------------------------------------------------------------
# `auto_round --help`
# ---------------------------------------------------------------------------


def test_bare_help_prints_the_concise_overview(capsys, monkeypatch):
    from auto_round.cli import main as cli_main

    monkeypatch.setattr(sys, "argv", ["auto_round", "--help"])
    cli_main.run()
    overview = capsys.readouterr().out

    assert "common arguments" in overview
    assert "auto_round quantize --help" in overview
    assert "--algorithm auto_round_v2" in overview
    # A curated subset, not the full quantize argument list.
    assert len(overview.splitlines()) < 100
    for noisy_flag in ("--svdquant-rank", "--kl_top_k", "--static_kv_dtype", "--target_bits"):
        assert noisy_flag not in overview


@pytest.mark.parametrize("argv", [["auto_round", "quantize", "--help"], ["auto_round", "help", "quantize"]])
def test_explicit_quantize_help_still_prints_every_argument(capsys, monkeypatch, argv):
    from auto_round.cli import main as cli_main

    monkeypatch.setattr(sys, "argv", argv)
    try:
        cli_main.run()
    except SystemExit as exc:  # `quantize --help` exits through argparse with code 0
        assert exc.code in (0, None)
    help_text = capsys.readouterr().out

    assert "auto_round quantize" in help_text
    for flag in ("--svdquant-rank", "--kl_top_k", "--static_kv_dtype", "--enable_alg_ext"):
        assert flag in help_text
    assert "auto_round_v2" in help_text


def test_algorithm_help_is_still_reported_for_the_new_algorithm(capsys, monkeypatch):
    from auto_round.cli import main as cli_main

    monkeypatch.setattr(sys, "argv", ["auto_round", "--algorithm", "auto_round_v2", "--help"])
    cli_main.run()
    help_text = capsys.readouterr().out

    assert "algorithm(s): auto_round_v2" in help_text
    assert "--iters" in help_text


def test_overview_entries_all_resolve_to_documented_quantize_flags():
    from auto_round.cli.main import _OVERVIEW_ARGUMENTS, _overview_parser

    parser = _overview_parser()
    rendered = {action.dest for action in parser._actions}
    missing = [dest for dest, _ in _OVERVIEW_ARGUMENTS if dest not in rendered]

    assert not missing, f"overview entries missing from the quantize parser: {missing}"
    for action in parser._actions:
        assert action.help not in (None, argparse.SUPPRESS), f"{action.dest} has no help text"


def test_every_quantize_flag_has_help_text():
    from auto_round.cli.parser import build_quantize_parser

    undocumented = [action.option_strings for action in build_quantize_parser()._actions if not action.help]

    assert undocumented == []


def test_quantize_help_has_no_empty_algorithm_group():
    import contextlib
    import io

    from auto_round.cli.parser import build_quantize_parser

    help_text = io.StringIO()
    with contextlib.redirect_stdout(help_text):
        build_quantize_parser().print_help()
    help_text = help_text.getvalue()

    # `auto_round_v2` reuses the SignRound switches, so it must not add a bare header.
    assert "Algorithm: auto_round:" in help_text
    assert "Algorithm: auto_round_v2:" not in help_text
