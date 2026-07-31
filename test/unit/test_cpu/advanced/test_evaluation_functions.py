# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CPU tests for evaluation utility functions.
Lightweight tests focusing on key utility functions without heavy model loading.

Run with: pytest test/test_cpu/advanced/test_evaluation_functions.py
"""

import argparse
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch


class TestSelectGgufEvalFile:
    """Test GGUF text file selection for evaluation."""

    def test_selects_only_text_gguf_when_mixed_format_filename_differs(self, tmp_path):
        from auto_round.eval.evaluation import select_gguf_eval_file

        (tmp_path / "gemma-4-E4B-it-7.5B-Q2_K_S.gguf").touch()
        (tmp_path / "mmproj-model.gguf").touch()

        gguf_file, candidates = select_gguf_eval_file(str(tmp_path), ["gguf:q2_k_mixed"])

        assert gguf_file == "gemma-4-E4B-it-7.5B-Q2_K_S.gguf"
        assert candidates == ["gemma-4-E4B-it-7.5B-Q2_K_S.gguf"]

    def test_prefers_exact_format_match_when_multiple_text_gguf_files_exist(self, tmp_path):
        from auto_round.eval.evaluation import select_gguf_eval_file

        (tmp_path / "model-Q4_0.gguf").touch()
        (tmp_path / "model-Q8_0.gguf").touch()
        (tmp_path / "mmproj-model.gguf").touch()

        gguf_file, candidates = select_gguf_eval_file(str(tmp_path), ["gguf:q4_0"])

        assert gguf_file == "model-Q4_0.gguf"
        assert candidates == ["model-Q4_0.gguf", "model-Q8_0.gguf"]


class TestParseVllmArgs:
    """Test parse_vllm_args function for parsing custom vllm arguments."""

    def test_parse_vllm_args_empty(self):
        """Test parsing empty vllm_args."""
        from auto_round.eval.eval_cli import parse_vllm_args

        result = parse_vllm_args(None)
        assert result == {}

        result = parse_vllm_args("")
        assert result == {}

    def test_parse_vllm_args_integers(self):
        """Test parsing integer arguments."""
        from auto_round.eval.eval_cli import parse_vllm_args

        result = parse_vllm_args("--tensor_parallel_size=2,--max_model_len=4096")
        assert result == {"tensor_parallel_size": 2, "max_model_len": 4096}
        assert isinstance(result["tensor_parallel_size"], int)
        assert isinstance(result["max_model_len"], int)

    def test_parse_vllm_args_floats(self):
        """Test parsing float arguments."""
        from auto_round.eval.eval_cli import parse_vllm_args

        result = parse_vllm_args("--gpu_memory_utilization=0.9,--swap_space=4.5")
        assert result == {"gpu_memory_utilization": 0.9, "swap_space": 4.5}
        assert isinstance(result["gpu_memory_utilization"], float)
        assert isinstance(result["swap_space"], float)

    def test_parse_vllm_args_booleans(self):
        """Test parsing boolean arguments."""
        from auto_round.eval.eval_cli import parse_vllm_args

        result = parse_vllm_args("--trust_remote_code=true,--enable_lora=false")
        assert result == {"trust_remote_code": True, "enable_lora": False}
        assert isinstance(result["trust_remote_code"], bool)
        assert isinstance(result["enable_lora"], bool)

    def test_parse_vllm_args_strings(self):
        """Test parsing string arguments."""
        from auto_round.eval.eval_cli import parse_vllm_args

        result = parse_vllm_args("--tokenizer_mode=auto,--quantization=awq")
        assert result == {"tokenizer_mode": "auto", "quantization": "awq"}
        assert isinstance(result["tokenizer_mode"], str)
        assert isinstance(result["quantization"], str)

    def test_parse_vllm_args_mixed_types(self):
        """Test parsing mixed type arguments."""
        from auto_round.eval.eval_cli import parse_vllm_args

        result = parse_vllm_args(
            "--tensor_parallel_size=2,--gpu_memory_utilization=0.9,--trust_remote_code=true,--tokenizer_mode=auto"
        )
        assert result == {
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": True,
            "tokenizer_mode": "auto",
        }

    def test_parse_vllm_args_without_double_dash(self):
        """Test parsing arguments without leading '--'."""
        from auto_round.eval.eval_cli import parse_vllm_args

        result = parse_vllm_args("tensor_parallel_size=2,max_model_len=4096")
        assert result == {"tensor_parallel_size": 2, "max_model_len": 4096}


class TestLoadGgufModelIfNeeded:
    """Test _load_gguf_model_if_needed function for GGUF model detection and loading."""

    def test_load_gguf_model_non_gguf_string_path(self):
        """Test with non-GGUF model path (string)."""
        from auto_round.eval.eval_cli import _load_gguf_model_if_needed

        model_path = "/path/to/regular/model"
        model, tokenizer, is_gguf, gguf_file = _load_gguf_model_if_needed(model_path)

        assert model == model_path
        assert tokenizer is None
        assert is_gguf is False
        assert gguf_file is None

    def test_load_gguf_model_non_string_model(self):
        """Test with model object (not a string path)."""
        from auto_round.eval.eval_cli import _load_gguf_model_if_needed

        model_obj = object()
        model, tokenizer, is_gguf, gguf_file = _load_gguf_model_if_needed(model_obj)
        assert model is model_obj
        assert tokenizer is None
        assert is_gguf is False
        assert gguf_file is None


def _make_eval_args(**overrides):
    defaults = dict(
        model="test-model",
        model_name="test-model",
        mllm=False,
        device_map="0",
        tasks="lambada_openai",
        disable_trust_remote_code=False,
        seed=42,
        eval_bs=2,
        eval_task_by_task=False,
        eval_model_dtype=None,
        limit=5,
        num_fewshot=3,
        eval_gen_kwargs="temperature=0.1,top_p=0.9",
        eval_backend="hf",
        add_bos_token=False,
        vllm_args=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fake_lm_eval_modules(simple_evaluate_impl):
    root_module = ModuleType("lm_eval")
    utils_module = ModuleType("lm_eval.utils")
    utils_module.make_table = lambda result: "table"

    evaluator_module = ModuleType("lm_eval.evaluator")
    evaluator_module.simple_evaluate = simple_evaluate_impl

    models_module = ModuleType("lm_eval.models")
    vllm_module = ModuleType("lm_eval.models.vllm_causallms")
    vllm_module.VLLM = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(kwargs=kwargs))

    vllm_vlm_module = ModuleType("lm_eval.models.vllm_vlms")
    vllm_vlm_module.VLLM_VLM = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(kwargs=kwargs))

    root_module.evaluator = evaluator_module
    root_module.utils = utils_module
    root_module.models = models_module
    models_module.vllm_causallms = vllm_module
    models_module.vllm_vlms = vllm_vlm_module

    return {
        "lm_eval": root_module,
        "lm_eval.utils": utils_module,
        "lm_eval.models": models_module,
        "lm_eval.evaluator": evaluator_module,
        "lm_eval.models.vllm_causallms": vllm_module,
        "lm_eval.models.vllm_vlms": vllm_vlm_module,
    }


class TestEvalArgumentForwarding:
    def test_run_eval_task_by_task_forwards_eval_generation_arguments(self):
        from auto_round.cli.main import run_eval

        args = _make_eval_args(eval_task_by_task=True)

        with patch("auto_round.cli.main.setup_eval_parser", return_value=args), patch(
            "auto_round.utils.is_gguf_model", return_value=False
        ), patch("auto_round.utils.is_mllm_model", return_value=False), patch(
            "auto_round.eval.eval_cli.eval_task_by_task"
        ) as mock_eval_task_by_task:
            run_eval([])

        _, kwargs = mock_eval_task_by_task.call_args
        assert kwargs["num_fewshot"] == args.num_fewshot
        assert kwargs["gen_kwargs"] == args.eval_gen_kwargs

    def test_eval_hf_batch_forwards_eval_generation_arguments(self):
        from auto_round.eval.eval_cli import eval

        args = _make_eval_args()
        result = {"results": {"lambada_openai": {}}, "versions": {}, "n-shot": {}, "higher_is_better": {}}

        with patch("auto_round.eval.eval_cli.require_version"), patch(
            "auto_round.eval.eval_cli.is_diffusion_model", return_value=False
        ), patch(
            "auto_round.eval.eval_cli._eval_init", return_value=(["lambada_openai"], "pretrained=test-model", "cpu")
        ), patch(
            "auto_round.eval.eval_cli._load_gguf_model_if_needed", return_value=(None, None, False, None)
        ), patch(
            "auto_round.eval.evaluation.simple_evaluate", return_value=result
        ) as mock_simple_evaluate, patch(
            "builtins.print"
        ), patch.dict(
            sys.modules, _fake_lm_eval_modules(MagicMock(return_value=result))
        ):
            eval(args)

        _, kwargs = mock_simple_evaluate.call_args
        assert kwargs["num_fewshot"] == args.num_fewshot
        assert kwargs["gen_kwargs"] == args.eval_gen_kwargs
        assert kwargs["fewshot_as_multiturn"] is False

    def test_eval_gguf_batch_forwards_eval_generation_arguments(self):
        from auto_round.eval.eval_cli import eval

        args = _make_eval_args()
        result = {"results": {"lambada_openai": {}}, "versions": {}, "n-shot": {}, "higher_is_better": {}}

        with patch("auto_round.eval.eval_cli.require_version"), patch(
            "auto_round.eval.eval_cli.is_diffusion_model", return_value=False
        ), patch(
            "auto_round.eval.eval_cli._eval_init", return_value=(["lambada_openai"], "pretrained=test-model", "cpu")
        ), patch(
            "auto_round.eval.eval_cli._load_gguf_model_if_needed",
            return_value=(object(), object(), True, "model.gguf"),
        ), patch(
            "auto_round.eval.evaluation.simple_evaluate_user_model", return_value=result
        ) as mock_simple_evaluate_user_model, patch(
            "builtins.print"
        ), patch.dict(
            sys.modules, _fake_lm_eval_modules(MagicMock(return_value=result))
        ):
            eval(args)

        _, kwargs = mock_simple_evaluate_user_model.call_args
        assert kwargs["num_fewshot"] == args.num_fewshot
        assert kwargs["gen_kwargs"] == args.eval_gen_kwargs

    def test_eval_with_vllm_forwards_eval_generation_arguments(self):
        from auto_round.eval.eval_cli import eval_with_vllm

        args = _make_eval_args()
        evaluator_simple_evaluate = MagicMock(
            return_value={"results": {"lambada_openai": {}}, "versions": {}, "n-shot": {}, "higher_is_better": {}}
        )

        with patch("auto_round.eval.eval_cli.get_device_and_parallelism", return_value=("cuda:0", False)), patch(
            "auto_round.eval.eval_cli.get_major_device", return_value="cuda"
        ), patch("auto_round.eval.eval_cli.get_model_dtype", return_value="float16"), patch(
            "builtins.print"
        ), patch.dict(
            sys.modules, _fake_lm_eval_modules(evaluator_simple_evaluate)
        ):
            eval_with_vllm(args)

        _, kwargs = evaluator_simple_evaluate.call_args
        assert kwargs["num_fewshot"] == args.num_fewshot
        assert kwargs["gen_kwargs"] == args.eval_gen_kwargs
        assert kwargs["fewshot_as_multiturn"] is False

    def test_run_model_evaluation_vllm_forwards_eval_generation_arguments(self, tmp_path):
        from auto_round.eval.evaluation import run_model_evaluation

        args = _make_eval_args(eval_backend="vllm")
        autoround = SimpleNamespace()

        with patch("auto_round.utils.model.detect_model_type", return_value="llm"), patch(
            "auto_round.utils.device_manager.get_device_and_parallelism", return_value=("cpu", False)
        ), patch("auto_round.eval.eval_cli.eval_with_vllm") as mock_eval_with_vllm:
            run_model_evaluation(None, None, autoround, str(tmp_path), ["auto_round"], args)

        forwarded_args = mock_eval_with_vllm.call_args.args[0]
        assert forwarded_args.num_fewshot == args.num_fewshot
        assert forwarded_args.eval_gen_kwargs == args.eval_gen_kwargs
