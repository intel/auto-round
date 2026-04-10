import os
import shutil
import sys

import pytest

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

        # Test llm script
        res = os.system(f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {tiny_opt_model_path} --seqlen 32 --iter 2 --nsamples 1 --format auto_gptq,auto_round --output_dir {self.save_dir}  --tasks piqa"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {tiny_opt_model_path} --seqlen 8 --iter 1 --nsamples 1 --eval_task_by_task --tasks openbookqa --bs 32"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -c 'from auto_round.__main__ import run_light; run_light()' --seqlen 8 --iter 2 --nsamples 8 --output_dir {self.save_dir} --tasks lambada_openai"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        # test mllm script

        # test auto_round_mllm --eval help
        res = os.system(f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --eval -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        # test auto_round_mllm --lmms help
        res = os.system(f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --eval --lmms -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --mllm --model {tiny_qwen_vl_model_path} --iter 2 --nsamples 2 --seqlen 32 --format auto_round --output_dir {self.save_dir}"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --mllm --iter 2 --nsamples 2 --model {tiny_qwen_vl_model_path} --seqlen 32 --format auto_round"
            f" --quant_nontext_module --output_dir {self.save_dir} "
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
