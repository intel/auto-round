import os
import shutil
import sys

import pytest
import torch

from auto_round.utils import parse_layer_config_arg

from ...helpers import get_model_path

AUTO_ROUND_PATH = __file__.split("/")
AUTO_ROUND_PATH = "/".join(AUTO_ROUND_PATH[: AUTO_ROUND_PATH.index("test")])


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

    def test_auto_round_cmd(self, tiny_opt_model_path, tiny_qwen_vl_model_path):
        python_path = sys.executable
        res = os.system(f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

    def test_auto_round_cmd2(self, tiny_opt_model_path, tiny_qwen_vl_model_path):
        python_path = sys.executable
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {tiny_opt_model_path} --seqlen 32 --iter 2 --nsamples 1 --format auto_gptq,auto_round --output_dir {self.save_dir} --tasks piqa --limit 2"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

    def test_auto_round_cmd3(self, tiny_opt_model_path, tiny_qwen_vl_model_path):
        python_path = sys.executable
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {tiny_opt_model_path} --seqlen 8 --iter 1 --nsamples 1 --eval_task_by_task --tasks openbookqa --bs 32 --limit 2"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

    def test_auto_round_cmd4(self, tiny_opt_model_path, tiny_qwen_vl_model_path):
        python_path = sys.executable
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -c 'from auto_round.__main__ import run_light; run_light()' --seqlen 8 --iter 2 --nsamples 8 --output_dir {self.save_dir} --tasks lambada_openai --limit 2"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

    def test_auto_round_cmd5(self, tiny_opt_model_path, tiny_qwen_vl_model_path):
        python_path = sys.executable
        res = os.system(f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --eval -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

    def test_auto_round_cmd6(self, tiny_opt_model_path, tiny_qwen_vl_model_path):
        python_path = sys.executable
        res = os.system(f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --eval --lmms -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

    def test_auto_round_cmd7(self, tiny_opt_model_path, tiny_qwen_vl_model_path):
        python_path = sys.executable
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --mllm --model {tiny_qwen_vl_model_path} --iter 2 --nsamples 2 --seqlen 32 --format auto_round --output_dir {self.save_dir}"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

    def test_auto_round_cmd8(self, tiny_opt_model_path, tiny_qwen_vl_model_path):
        python_path = sys.executable
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --mllm --iter 2 --nsamples 2 --model {tiny_qwen_vl_model_path} --seqlen 32 --format auto_round"
            f" --quant_nontext_module --output_dir {self.save_dir}"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

    def test_layer_config(self, tiny_opt_model_path):
        """Test --layer_config with unquoted JSON-like syntax."""
        python_path = sys.executable
        layer_cfg = r"{fc1:{bits:8,data_type:int},fc2:{bits:16,data_type:int}}"
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round"
            f" --model {tiny_opt_model_path} --seqlen 8 --iter 0 --disable_opt_rtn"
            f" --layer_config '{layer_cfg}' --format auto_round --output_dir {self.save_dir}"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test with --layer_config fail, please have a check"


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


def test_rtn_accepts_disable_opt_rtn_flag():
    from auto_round.cli.algorithms import AlgorithmHandler
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(["--model", "dummy-model", "--algorithm", "rtn", "--disable_opt_rtn"])
    configs = AlgorithmHandler.build_configs(args, {})

    assert args.disable_opt_rtn is True
    assert configs[0].disable_opt_rtn is True


def test_cli_builds_svdquant_config_with_smoothing_disabled_by_default():
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
    from auto_round.cli.algorithms import AlgorithmHandler
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(["--model", "dummy-model", "--algorithm", "rtn", "--enable_svdquant"])
    configs = AlgorithmHandler.build_configs(args, {})

    assert args.svdquant_smooth_enabled is False
    assert args.svdquant_smooth_num_grids == 20
    assert isinstance(configs[0], SVDQuantConfig)
    assert configs[0].smooth_enabled is False
    assert configs[0].smooth_num_grids == 20


def test_cli_enables_svdquant_smoothing_explicitly():
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
    from auto_round.cli.algorithms import AlgorithmHandler
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(
        [
            "--model",
            "dummy-model",
            "--algorithm",
            "rtn",
            "--enable_svdquant",
            "--enable_svdquant_smooth",
            "--svdquant_smooth_num_grids",
            "8",
        ]
    )
    configs = AlgorithmHandler.build_configs(args, {})

    assert args.svdquant_smooth_enabled is True
    assert isinstance(configs[0], SVDQuantConfig)
    assert configs[0].smooth_enabled is True
    assert configs[0].smooth_num_grids == 8


def test_cli_builds_svdquant_residual_iteration_config():
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
    from auto_round.cli.algorithms import AlgorithmHandler
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(
        [
            "--model",
            "dummy-model",
            "--algorithm",
            "rtn",
            "--enable_svdquant",
            "--svdquant_residual_iters",
            "3",
            "--enable_svdquant_residual_early_stop",
            "--svdquant_residual_quant_method",
            "rtn",
            "--svdquant_model_adapter",
            "flux",
        ]
    )
    configs = AlgorithmHandler.build_configs(args, {})

    assert isinstance(configs[0], SVDQuantConfig)
    assert configs[0].residual_iters == 3
    assert configs[0].residual_early_stop is True
    assert configs[0].residual_quant_method == "rtn"
    assert args.svdquant_model_adapter == "flux"


def test_svdquant_nunchaku_cli_passes_model_adapter_to_exporter():
    from auto_round.cli.main import _build_save_kwargs
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(
        [
            "--model",
            "dummy-model",
            "--enable_svdquant",
            "--format",
            "svdquant_nunchaku",
            "--svdquant_model_adapter",
            "flux",
        ]
    )

    assert _build_save_kwargs(args, ["svdquant_nunchaku"]) == {"model_adapter": "flux"}


def test_svdquant_nunchaku_cli_requires_transform():
    from auto_round.cli.main import _build_save_kwargs
    from auto_round.cli.parser import build_quantize_parser

    args = build_quantize_parser().parse_args(["--model", "dummy-model", "--format", "svdquant_nunchaku"])

    with pytest.raises(ValueError, match="requires --enable_svdquant"):
        _build_save_kwargs(args, ["svdquant_nunchaku"])


def test_flux_adapter_supplies_svdquant_target_modules():
    from auto_round.cli.main import _apply_svdquant_adapter_defaults
    from auto_round.cli.parser import build_quantize_parser
    from auto_round.export.svdquant_adapters import FLUX_SVDQUANT_TARGET_MODULES

    args = build_quantize_parser().parse_args(
        ["--model", "dummy", "--enable_svdquant", "--svdquant_model_adapter", "flux"]
    )
    _apply_svdquant_adapter_defaults(args)

    assert args.svdquant_target_modules.split(",") == list(FLUX_SVDQUANT_TARGET_MODULES)


def test_svdquant_cli_help_uses_underscore_option_names():
    from auto_round.cli.parser import build_quantize_parser

    help_text = build_quantize_parser().format_help()

    for option in (
        "--enable_svdquant_smooth",
        "--svdquant_smooth_num_grids",
        "--enable_svdquant_residual_early_stop",
        "--svdquant_rank",
        "--svdquant_residual_iters",
        "--svdquant_residual_quant_method",
        "--svdquant_model_adapter",
    ):
        assert option in help_text
    assert "--svdquant-rank" not in help_text


@pytest.mark.parametrize(
    "legacy_option",
    [
        "--svdquant-rank",
        "--svdquant-smooth-enabled",
        "--no-svdquant-smooth-enabled",
        "--svdquant_smooth_alpha",
        "--svdquant-residual-iters",
        "--svdquant-residual-early-stop",
        "--no-svdquant-residual-early-stop",
        "--svdquant-residual-quant-method",
        "--svdquant-model-adapter",
    ],
)
def test_svdquant_cli_rejects_removed_hyphen_options(legacy_option):
    from auto_round.cli.parser import build_quantize_parser

    with pytest.raises(SystemExit):
        build_quantize_parser().parse_args([legacy_option])


def test_auto_adapter_detects_flux_target_modules_from_loaded_transformer():
    from types import SimpleNamespace

    from auto_round.cli.main import _apply_svdquant_adapter_defaults
    from auto_round.cli.parser import build_quantize_parser
    from auto_round.export.svdquant_adapters import FLUX_SVDQUANT_TARGET_MODULES

    args = build_quantize_parser().parse_args(["--model", "dummy", "--enable_svdquant"])
    model = SimpleNamespace(transformer=SimpleNamespace(config={"_class_name": "FluxTransformer2DModel"}))

    _apply_svdquant_adapter_defaults(args, model)

    assert args.svdquant_target_modules.split(",") == list(FLUX_SVDQUANT_TARGET_MODULES)


def test_auto_adapter_detects_flux_target_modules_from_local_pipeline(tmp_path):
    from auto_round.cli.main import _apply_svdquant_adapter_defaults
    from auto_round.cli.parser import build_quantize_parser
    from auto_round.export.svdquant_adapters import FLUX_SVDQUANT_TARGET_MODULES

    (tmp_path / "model_index.json").write_text('{"_class_name": "FluxPipeline"}', encoding="utf-8")
    args = build_quantize_parser().parse_args(["--model", str(tmp_path), "--enable_svdquant"])

    _apply_svdquant_adapter_defaults(args, str(tmp_path))

    assert args.svdquant_target_modules.split(",") == list(FLUX_SVDQUANT_TARGET_MODULES)
