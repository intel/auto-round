import shutil
import sys
from test.helpers import get_model_path

import pytest

from auto_round.cli.main import run, run_light
from auto_round.utils import parse_layer_config_arg


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

    def test_auto_round_cmd(self, monkeypatch, tiny_opt_model_path, tiny_qwen_vl_model_path):
        _assert_cli_ok(monkeypatch, ["auto_round", "-h"])

    @pytest.mark.timeout(90)
    def test_auto_round_cmd2(self, monkeypatch, tiny_opt_model_path, tiny_qwen_vl_model_path):
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

    @pytest.mark.timeout(90)
    def test_auto_round_cmd3(self, monkeypatch, tiny_opt_model_path, tiny_qwen_vl_model_path):
        _assert_cli_ok(
            monkeypatch,
            [
                "auto_round",
                "--model",
                tiny_opt_model_path,
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

    @pytest.mark.timeout(90)
    def test_auto_round_cmd4(self, monkeypatch, tiny_opt_model_path, tiny_qwen_vl_model_path):
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

    def test_auto_round_cmd5(self, monkeypatch, tiny_opt_model_path, tiny_qwen_vl_model_path):
        _assert_cli_ok(monkeypatch, ["auto_round", "--eval", "-h"])

    def test_auto_round_cmd6(self, monkeypatch, tiny_opt_model_path, tiny_qwen_vl_model_path):
        _assert_cli_ok(monkeypatch, ["auto_round", "--eval", "--lmms", "-h"])

    @pytest.mark.timeout(90)
    def test_auto_round_cmd7(self, monkeypatch, tiny_opt_model_path, tiny_qwen_vl_model_path):
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

    @pytest.mark.timeout(60)
    def test_auto_round_cmd8(self, monkeypatch, tiny_opt_model_path, tiny_qwen_vl_model_path):
        _assert_cli_ok(
            monkeypatch,
            [
                "auto_round",
                "--mllm",
                "--iter",
                "2",
                "--nsamples",
                "2",
                "--model",
                tiny_qwen_vl_model_path,
                "--seqlen",
                "32",
                "--format",
                "auto_round",
                "--quant_nontext_module",
                "--output_dir",
                self.save_dir,
            ],
        )

    def test_layer_config(self, monkeypatch, tiny_opt_model_path):
        """Test --layer_config with unquoted JSON-like syntax."""
        layer_cfg = r"{fc1:{bits:8,data_type:int},fc2:{bits:16,data_type:int}}"
        _assert_cli_ok(
            monkeypatch,
            [
                "auto_round",
                "--model",
                tiny_opt_model_path,
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
            ],
        )


def test_parse_layer_config():
    """Unit test for parse_layer_config_arg parsing logic."""
    result = parse_layer_config_arg("{mtp:{bits:8,data_type:int},mtp.fc:{bits:16,data_type:int}}")
    assert result == {
        "mtp": {"bits": 8, "data_type": "int"},
        "mtp.fc": {"bits": 16, "data_type": "int"},
    }


def test_parse_layer_config_with_quoted_regex_keys():
    """Quoted JSON-like input with regex paths should remain usable from the CLI."""
    result = parse_layer_config_arg(
        r'{"model.language_model.layers.\\d+.self_attn..":{"bits":"8"},'
        r'"model.language_model.layers.\\d+.router..*":{"bits":"8"}}'
    )
    assert result == {
        r"model.language_model.layers.\d+.self_attn..": {"bits": 8},
        r"model.language_model.layers.\d+.router..*": {"bits": 8},
    }


def test_parse_layer_config_with_single_escaped_regex_keys():
    """Shell-passed strings often contain single backslashes that are invalid JSON but still recoverable."""
    result = parse_layer_config_arg(
        r'{"model.language_model.layers.\d+.self_attn..":{"bits":"8"},'
        r'"model.language_model.layers.\d+.mlp..":{"bits":"8"}}'
    )
    assert result == {
        r"model.language_model.layers.\d+.self_attn..": {"bits": 8},
        r"model.language_model.layers.\d+.mlp..": {"bits": 8},
    }


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
