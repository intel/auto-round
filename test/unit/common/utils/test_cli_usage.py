import argparse
import inspect
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

    @pytest.mark.timeout(90)
    def test_auto_round_cmd2(self, monkeypatch, tiny_opt_model_path):
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
                "--format",
                "auto_gptq,auto_round",
                "--output_dir",
                self.save_dir,
                "--tasks",
                "piqa",
                "--limit",
                "2",
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

    @pytest.mark.timeout(90)
    def test_auto_round_cmd4(self, monkeypatch):
        _assert_cli_ok(
            monkeypatch,
            [
                "auto_round",
                "--seqlen",
                "8",
                "--iter",
                "2",
                "--nsamples",
                "8",
                "--output_dir",
                self.save_dir,
                "--tasks",
                "lambada_openai",
                "--limit",
                "2",
            ],
            entry=run_light,
        )

    def test_auto_round_cmd5(self, monkeypatch):
        _assert_cli_ok(monkeypatch, ["auto_round", "--eval", "-h"])

    def test_auto_round_cmd6(self, monkeypatch):
        _assert_cli_ok(monkeypatch, ["auto_round", "--eval", "--lmms", "-h"])

    @pytest.mark.timeout(90)
    def test_auto_round_cmd7(self, monkeypatch, tiny_qwen_vl_model_path):
        _assert_cli_ok(
            monkeypatch,
            [
                "auto_round",
                "--mllm",
                "--model",
                tiny_qwen_vl_model_path,
                "--iter",
                "2",
                "--nsamples",
                "2",
                "--seqlen",
                "32",
                "--format",
                "auto_round",
                "--output_dir",
                self.save_dir,
            ],
        )

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


def test_svdquant_cli_builds_hyphenated_options_before_rtn():
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
            "flux",
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
    assert configs[0].model_adapter == "flux"
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


def _normalize_options(raw):
    if raw is None:
        return None
    flat = ",".join(raw)
    return ",".join(p.strip() for p in flat.split(",") if p.strip())


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


def test_options_comma_space_separated():
    """--options accepts comma-separated and space-separated values."""
    from auto_round.cli.parser import build_quantize_parser

    p = build_quantize_parser()
    assert _normalize_options(p.parse_args(["--avg_bits", "4", "--options", "W4A16,W8A16"]).options) == "W4A16,W8A16"
    assert _normalize_options(p.parse_args(["--avg_bits", "4", "--options", "W4A16", "W8A16"]).options) == "W4A16,W8A16"
    assert p.parse_args(["--model", "dummy"]).options is None


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
