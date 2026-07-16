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
"""CPU-only pytest coverage for `auto_round.eval.eval_cli`."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from auto_round.eval import eval_cli


class TestParseVllmArgs:
    """Tests for `parse_vllm_args`."""

    def test_empty_string_returns_empty_dict(self):
        assert eval_cli.parse_vllm_args("") == {}

    def test_none_returns_empty_dict(self):
        assert eval_cli.parse_vllm_args(None) == {}

    def test_simple_args_are_parsed(self):
        result = eval_cli.parse_vllm_args("tensor_parallel_size=2,gpu_memory_utilization=0.9")
        assert result["tensor_parallel_size"] == 2
        assert result["gpu_memory_utilization"] == 0.9

    def test_leading_dashes_are_stripped(self):
        result = eval_cli.parse_vllm_args("--tensor_parallel_size=2")
        assert result["tensor_parallel_size"] == 2

    def test_boolean_values_are_converted(self):
        result = eval_cli.parse_vllm_args("enable_chunked_prefill=true,disable_log_requests=false")
        assert result["enable_chunked_prefill"] is True
        assert result["disable_log_requests"] is False

    def test_space_separated_args_are_normalized(self):
        result = eval_cli.parse_vllm_args("tensor_parallel_size 2")
        assert result["tensor_parallel_size"] == 2

    def test_unknown_values_are_kept_as_strings(self):
        result = eval_cli.parse_vllm_args("model=facebook/opt-125m")
        assert result["model"] == "facebook/opt-125m"


class TestEvalArgumentParser:
    """Check parser defaults, aliases, and diffusion-specific options."""

    def test_positional_model_default_is_none(self):
        parser = eval_cli.EvalArgumentParser()
        args = parser.parse_args([])
        assert args.model is None

    def test_positional_model_is_accepted(self):
        parser = eval_cli.EvalArgumentParser()
        args = parser.parse_args(["local-model-path"])
        assert args.model == "local-model-path"

    def test_model_name_default(self):
        parser = eval_cli.EvalArgumentParser()
        args = parser.parse_args([])
        assert args.model_name == "facebook/opt-125m"

    def test_model_alias_updates_model_name(self):
        parser = eval_cli.EvalArgumentParser()
        args = parser.parse_args(["--model_name", "custom-model"])
        assert args.model_name == "custom-model"

    def test_default_device_map(self):
        parser = eval_cli.EvalArgumentParser()
        args = parser.parse_args([])
        assert args.device_map == "0"

    def test_default_tasks(self):
        parser = eval_cli.EvalArgumentParser()
        args = parser.parse_args([])
        assert "mmlu" in args.tasks
        assert "hellaswag" in args.tasks

    def test_tasks_are_overridable(self):
        parser = eval_cli.EvalArgumentParser()
        args = parser.parse_args(["--tasks", "mmlu,wikitext"])
        assert args.tasks == "mmlu,wikitext"

    def test_disable_trust_remote_code_flag(self):
        parser = eval_cli.EvalArgumentParser()
        args = parser.parse_args(["--disable_trust_remote_code"])
        assert args.disable_trust_remote_code is True

    def test_diffusion_args_are_available(self):
        parser = eval_cli.EvalArgumentParser()
        args = parser.parse_args(
            [
                "--prompt",
                "a cat",
                "--metrics",
                "clip-iqa",
                "--guidance_scale",
                "10.0",
                "--num_inference_steps",
                "10",
            ]
        )
        assert args.prompt == "a cat"
        assert args.metrics == "clip-iqa"
        assert args.guidance_scale == 10.0
        assert args.num_inference_steps == 10

    def test_eval_backend_defaults_to_hf(self):
        parser = eval_cli.EvalArgumentParser()
        args = parser.parse_args([])
        assert args.eval_backend == "hf"

    def test_vllm_args_are_accepted(self):
        parser = eval_cli.EvalArgumentParser()
        args = parser.parse_args(["--vllm_args", "tensor_parallel_size=2,gpu_memory_utilization=0.9"])
        assert args.vllm_args == "tensor_parallel_size=2,gpu_memory_utilization=0.9"


class TestEvalInit:
    """Tests for `_eval_init` task normalization, device resolution, and dtype."""

    def test_cuda_visible_devices_is_set(self):
        with patch.object(eval_cli, "set_cuda_visible_devices") as mock_set, patch.object(
            eval_cli, "get_device_and_parallelism", return_value=("cpu", None)
        ), patch.object(eval_cli, "get_model_dtype", return_value="auto"):
            tasks, model_args, device_str = eval_cli._eval_init(
                "mmlu,wikitext", "/model", "0", disable_trust_remote_code=False, dtype="auto"
            )
        mock_set.assert_called_once_with("0")
        assert device_str == "cpu"

    def test_tasks_are_split_from_comma_string(self):
        with patch.object(eval_cli, "set_cuda_visible_devices"), patch.object(
            eval_cli, "get_device_and_parallelism", return_value=("cpu", None)
        ), patch.object(eval_cli, "get_model_dtype", return_value="auto"):
            tasks, _, _ = eval_cli._eval_init(
                "mmlu,wikitext", "/model", "0", disable_trust_remote_code=False, dtype="auto"
            )
        assert tasks == ["mmlu", "wikitext"]

    def test_list_tasks_are_passed_through(self):
        with patch.object(eval_cli, "set_cuda_visible_devices"), patch.object(
            eval_cli, "get_device_and_parallelism", return_value=("cpu", None)
        ), patch.object(eval_cli, "get_model_dtype", return_value="auto"):
            tasks, _, _ = eval_cli._eval_init(
                ["mmlu", "wikitext"], "/model", "0", disable_trust_remote_code=False, dtype="auto"
            )
        assert tasks == ["mmlu", "wikitext"]

    def test_parallelism_appended_to_model_args(self):
        with patch.object(eval_cli, "set_cuda_visible_devices"), patch.object(
            eval_cli, "get_device_and_parallelism", return_value=("cpu", "p")
        ), patch.object(eval_cli, "get_model_dtype", return_value="auto"):
            _, model_args, _ = eval_cli._eval_init(
                "mmlu", "/model", "0", disable_trust_remote_code=False, dtype="auto"
            )
        assert ",parallelize=True" in model_args

    def test_dtype_is_resolved_when_not_auto(self):
        with patch.object(eval_cli, "set_cuda_visible_devices"), patch.object(
            eval_cli, "get_device_and_parallelism", return_value=("cpu", None)
        ), patch.object(eval_cli, "get_model_dtype", return_value="auto"):
            _, model_args, _ = eval_cli._eval_init(
                "mmlu", "/model", "0", disable_trust_remote_code=False, dtype="bfloat16"
            )
        assert "dtype=auto" in model_args

    def test_model_args_contains_trust_remote_code(self):
        with patch.object(eval_cli, "set_cuda_visible_devices"), patch.object(
            eval_cli, "get_device_and_parallelism", return_value=("cpu", None)
        ), patch.object(eval_cli, "get_model_dtype", return_value="auto"):
            _, model_args, _ = eval_cli._eval_init(
                "mmlu", "/model", "0", disable_trust_remote_code=True, dtype="auto"
            )
        assert "trust_remote_code=False" in model_args
        assert ",parallelize=True" not in model_args


class TestEval:
    """Tests for the main `eval` entry point."""

    def test_diffusion_model_path_skips_lm_eval(self, monkeypatch):
        args = SimpleNamespace(model="diffusion-model", eval_backend="hf")
        captured = {}

        def fake_diffusion_eval(evaluation_args, pipe):
            captured["args"] = evaluation_args
            captured["pipe"] = pipe

        monkeypatch.setattr(eval_cli, "is_diffusion_model", lambda value: True)
        monkeypatch.setattr("auto_round.utils.diffusion_load_model", lambda value: (object(), object()))
        monkeypatch.setattr("auto_round.eval.evaluation.evaluate_diffusion_model", fake_diffusion_eval)
        eval_cli.eval(args)
        assert captured["args"] is args
        assert captured["pipe"] is not None

    def test_vllm_backend_delegates(self, monkeypatch):
        captured = {}

        def fake_eval_with_vllm(args):
            captured["args"] = args

        args = SimpleNamespace(model="vllm-model", eval_backend="vllm")
        monkeypatch.setattr(eval_cli, "is_diffusion_model", lambda value: False)
        monkeypatch.setattr(eval_cli, "eval_with_vllm", fake_eval_with_vllm)
        eval_cli.eval(args)
        assert captured["args"] is args

    def test_gguf_branch_uses_user_model(self, monkeypatch, capsys):
        args = SimpleNamespace(
            model="/model.gguf",
            eval_backend="hf",
            eval_bs=None,
            mllm=False,
            eval_model_dtype="auto",
            tasks="mmlu",
            device_map="cpu",
            disable_trust_remote_code=False,
            add_bos_token=False,
            limit=None,
        )

        fake_res = {"results": {"mmlu": {}}, "versions": {}, "n-shot": {}, "higher_is_better": {}}

        monkeypatch.setattr(eval_cli, "is_diffusion_model", lambda value: False)
        monkeypatch.setattr(
            eval_cli,
            "_load_gguf_model_if_needed",
            lambda *args, **kwargs: (object(), object(), True, "model.gguf"),
        )

        monkeypatch.setattr(
            "auto_round.eval.evaluation.simple_evaluate_user_model",
            lambda *args, **kwargs: fake_res,
        )

        eval_cli.eval(args)
        captured = capsys.readouterr()
        assert "evaluation running time=" in captured.out

    def test_mllm_warning_when_auto_batch_size(self, monkeypatch, capsys):
        args = SimpleNamespace(
            model="/model",
            eval_backend="hf",
            eval_bs=None,
            mllm=True,
            eval_model_dtype="auto",
            tasks="mmlu",
            device_map="cpu",
            disable_trust_remote_code=False,
            add_bos_token=False,
            limit=None,
        )

        monkeypatch.setattr(eval_cli, "is_diffusion_model", lambda value: False)
        monkeypatch.setattr(
            eval_cli,
            "_load_gguf_model_if_needed",
            lambda *args, **kwargs: (object(), object(), False, None),
        )

        captured_calls = {}

        def fake_simple_evaluate(*args, **kwargs):
            captured_calls["kwargs"] = kwargs
            return {"results": {"mmlu": {}}, "versions": {}, "n-shot": {}, "higher_is_better": {}}

        monkeypatch.setattr("auto_round.eval.evaluation.simple_evaluate", fake_simple_evaluate)

        eval_cli.eval(args)
        captured = capsys.readouterr()
        assert captured_calls["kwargs"]["batch_size"] == 16
        assert "evaluation running time=" in captured.out

    def test_non_mllm_uses_hf_model_name(self, monkeypatch, capsys):
        args = SimpleNamespace(
            model="/model",
            eval_backend="hf",
            eval_bs=8,
            mllm=False,
            eval_model_dtype="auto",
            tasks="mmlu",
            device_map="cpu",
            disable_trust_remote_code=False,
            add_bos_token=True,
            limit=None,
        )

        monkeypatch.setattr(eval_cli, "is_diffusion_model", lambda value: False)
        monkeypatch.setattr(
            eval_cli,
            "_load_gguf_model_if_needed",
            lambda *args, **kwargs: (object(), object(), False, None),
        )

        captured_calls = {}

        def fake_simple_evaluate(*args, **kwargs):
            captured_calls["kwargs"] = kwargs
            return {"results": {"mmlu": {}}, "versions": {}, "n-shot": {}, "higher_is_better": {}}

        monkeypatch.setattr("auto_round.eval.evaluation.simple_evaluate", fake_simple_evaluate)

        eval_cli.eval(args)
        captured = capsys.readouterr()
        assert captured_calls["kwargs"]["model"] == "hf"
        assert "add_bos_token=True" in captured_calls["kwargs"]["model_args"]
        assert "evaluation running time=" in captured.out


class TestEvalWithVllm:
    """Tests for the vLLM evaluation backend."""

    def test_tensor_parallel_size_from_device_map(self, monkeypatch):
        args = SimpleNamespace(
            model="model-id",
            device_map="0,1",
            eval_bs=8,
            eval_model_dtype="auto",
            tasks="mmlu",
            mllm=False,
            disable_trust_remote_code=False,
            add_bos_token=False,
            limit=None,
            vllm_args=None,
        )

        captured_kwargs = {}
        fake_res = {"results": {"mmlu": {}}, "versions": {}, "n-shot": {}, "higher_is_better": {}}

        class FakeVLLM:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        fake_vllm_causallms = type(sys)("lm_eval.models.vllm_causallms")
        fake_vllm_causallms.VLLM = FakeVLLM
        fake_vllm_vlms = type(sys)("lm_eval.models.vllm_vlms")
        fake_vllm_vlms.VLLM_VLM = type("VLLM_VLM", (), {"__init__": lambda *args, **kwargs: None})

        monkeypatch.setitem(sys.modules, "lm_eval.models.vllm_causallms", fake_vllm_causallms)
        monkeypatch.setitem(sys.modules, "lm_eval.models.vllm_vlms", fake_vllm_vlms)

        monkeypatch.setattr(eval_cli, "get_major_device", lambda: "cuda")
        monkeypatch.setattr(eval_cli, "get_device_and_parallelism", lambda device: ("cuda", False))
        monkeypatch.setattr(eval_cli, "get_model_dtype", lambda dtype, default="auto": "auto")
        monkeypatch.setattr(
            "lm_eval.evaluator.simple_evaluate",
            lambda **kwargs: fake_res,
        )
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")

        with patch("auto_round.utils.DEVICE_ENVIRON_VARIABLE_MAPPING", {"cuda": "CUDA_VISIBLE_DEVICES"}):
            eval_cli.eval_with_vllm(args)

        assert captured_kwargs["pretrained"] == "model-id"
        assert captured_kwargs["tensor_parallel_size"] == 2

    def test_mllm_uses_vllm_vlm(self, monkeypatch):
        args = SimpleNamespace(
            model="model-id",
            device_map="0",
            eval_bs=8,
            eval_model_dtype="auto",
            tasks="mmlu",
            mllm=True,
            disable_trust_remote_code=False,
            add_bos_token=False,
            limit=None,
            vllm_args=None,
        )

        captured_class = {}

        class FakeVLLM_VLM:
            def __init__(self, **kwargs):
                captured_class["kwargs"] = kwargs

        fake_vllm_causallms = type(sys)("lm_eval.models.vllm_causallms")
        fake_vllm_causallms.VLLM = object
        fake_vllm_vlms = type(sys)("lm_eval.models.vllm_vlms")
        fake_vllm_vlms.VLLM_VLM = FakeVLLM_VLM

        monkeypatch.setitem(sys.modules, "lm_eval.models.vllm_causallms", fake_vllm_causallms)
        monkeypatch.setitem(sys.modules, "lm_eval.models.vllm_vlms", fake_vllm_vlms)

        monkeypatch.setattr(eval_cli, "get_major_device", lambda: "cuda")
        monkeypatch.setattr(eval_cli, "get_device_and_parallelism", lambda device: ("cuda", False))
        monkeypatch.setattr(eval_cli, "get_model_dtype", lambda dtype, default="auto": "auto")
        monkeypatch.setattr(
            "lm_eval.evaluator.simple_evaluate",
            lambda **kwargs: {"results": {"mmlu": {}}, "versions": {}, "n-shot": {}, "higher_is_better": {}},
        )
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

        with patch("auto_round.utils.DEVICE_ENVIRON_VARIABLE_MAPPING", {"cuda": "CUDA_VISIBLE_DEVICES"}):
            eval_cli.eval_with_vllm(args)

        assert captured_class["kwargs"]["pretrained"] == "model-id"

    def test_existing_device_env_is_not_overwritten(self, monkeypatch):
        args = SimpleNamespace(
            model="model-id",
            device_map="0",
            eval_bs=8,
            eval_model_dtype="auto",
            tasks="mmlu",
            mllm=False,
            disable_trust_remote_code=False,
            add_bos_token=False,
            limit=None,
            vllm_args=None,
        )

        fake_vllm_causallms = type(sys)("lm_eval.models.vllm_causallms")
        fake_vllm_causallms.VLLM = type("VLLM", (), {"__init__": lambda *args, **kwargs: None})
        fake_vllm_vlms = type(sys)("lm_eval.models.vllm_vlms")
        fake_vllm_vlms.VLLM_VLM = type("VLLM_VLM", (), {"__init__": lambda *args, **kwargs: None})

        monkeypatch.setitem(sys.modules, "lm_eval.models.vllm_causallms", fake_vllm_causallms)
        monkeypatch.setitem(sys.modules, "lm_eval.models.vllm_vlms", fake_vllm_vlms)

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "fake-env")
        monkeypatch.setattr(eval_cli, "get_major_device", lambda: "cuda")
        monkeypatch.setattr(eval_cli, "get_device_and_parallelism", lambda device: ("cuda", False))
        monkeypatch.setattr(eval_cli, "get_model_dtype", lambda dtype, default="auto": "auto")
        monkeypatch.setattr(
            "lm_eval.evaluator.simple_evaluate",
            lambda **kwargs: {"results": {"mmlu": {}}, "versions": {}, "n-shot": {}, "higher_is_better": {}},
        )

        with patch("auto_round.utils.DEVICE_ENVIRON_VARIABLE_MAPPING", {"cuda": "CUDA_VISIBLE_DEVICES"}):
            eval_cli.eval_with_vllm(args)

        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "fake-env"


class TestEvalTaskByTask:
    """Tests for `eval_task_by_task`, focusing on CPU-safe branches."""

    def test_non_parallel_string_model_path(self, monkeypatch):
        fake_hflm = object()

        monkeypatch.setattr(eval_cli, "set_cuda_visible_devices", lambda device: None)
        monkeypatch.setattr(eval_cli, "get_device_and_parallelism", lambda device: ("cpu", None))
        monkeypatch.setattr(
            "auto_round.eval.eval_cli._load_gguf_model_if_needed",
            lambda *args, **kwargs: ("hf-causallm-model", None, False, None),
        )
        monkeypatch.setattr(
            "lm_eval.models.huggingface.HFLM",
            lambda **kwargs: fake_hflm,
        )
        monkeypatch.setattr(
            "auto_round.eval.eval_cli.dispatch_model_block_wise",
            lambda model, device_map: None,
        )

        captured = {}

        def fake_evaluate(*args, **kwargs):
            captured["tasks"] = kwargs.get("tasks", args[0] if args else None)
            captured["hflm"] = kwargs.get("hflm", args[1] if len(args) > 1 else None)

        monkeypatch.setattr(eval_cli, "_evaluate_tasks_with_retry", fake_evaluate)

        eval_cli.eval_task_by_task(
            model="hf-causallm-model",
            device="cpu",
            tasks="mmlu",
            batch_size=4,
            limit=2,
        )

        assert captured["hflm"] is fake_hflm

    def test_non_parallel_non_string_model_skips_gguf(self, monkeypatch):
        fake_model = object()
        fake_hflm = object()

        monkeypatch.setattr(eval_cli, "set_cuda_visible_devices", lambda device: None)
        monkeypatch.setattr(eval_cli, "get_device_and_parallelism", lambda device: ("cpu", None))
        monkeypatch.setattr(
            "auto_round.eval.eval_cli._load_gguf_model_if_needed",
            lambda *args, **kwargs: (fake_model, None, False, None),
        )
        monkeypatch.setattr(
            "lm_eval.models.huggingface.HFLM",
            lambda **kwargs: fake_hflm,
        )
        monkeypatch.setattr(
            "auto_round.eval.eval_cli.dispatch_model_block_wise",
            lambda model, device_map: None,
        )
        monkeypatch.setattr(eval_cli, "_evaluate_tasks_with_retry", lambda *args, **kwargs: None)

        eval_cli.eval_task_by_task(
            model=fake_model,
            device="cpu",
            tasks="mmlu",
        )


class TestEvaluateTasksWithRetry:
    """Tests for `_evaluate_tasks_with_retry` retry and aggregation behavior."""

    def test_successful_task_is_recorded(self, monkeypatch):
        fake_hflm = object()
        fake_res = {
            "results": {"mmlu": {"accuracy": 0.5}},
            "versions": {"mmlu": "1.0"},
            "n-shot": {"mmlu": 5},
            "higher_is_better": {"mmlu": True},
        }

        monkeypatch.setattr(
            "lm_eval.simple_evaluate",
            lambda **kwargs: fake_res,
        )
        monkeypatch.setattr(
            "lm_eval.utils.make_table",
            lambda res: "",
        )

        eval_cli._evaluate_tasks_with_retry(
            tasks=["mmlu"],
            hflm=fake_hflm,
            device_str="cpu",
            batch_size=8,
            limit=None,
            retry_times=3,
        )

    def test_string_tasks_are_split(self, monkeypatch):
        fake_hflm = object()
        fake_res = {
            "results": {"mmlu": {"accuracy": 0.5}},
            "versions": {"mmlu": "1.0"},
            "n-shot": {"mmlu": 5},
            "higher_is_better": {"mmlu": True},
        }
        captured = []

        def fake_simple_evaluate(**kwargs):
            captured.append(kwargs["tasks"])
            return fake_res

        monkeypatch.setattr("lm_eval.simple_evaluate", fake_simple_evaluate)
        monkeypatch.setattr("lm_eval.utils.make_table", lambda res: "")

        eval_cli._evaluate_tasks_with_retry(
            tasks="mmlu,wikitext",
            hflm=fake_hflm,
            device_str="cpu",
            batch_size=8,
            limit=None,
            retry_times=1,
        )

        assert captured == ["mmlu", "wikitext"]

    def test_oom_retry_reduces_batch_size(self, monkeypatch):
        fake_hflm = type("FakeHFLM", (), {"batch_sizes": None})()
        fake_res = {
            "results": {"mmlu": {"accuracy": 0.5}},
            "versions": {"mmlu": "1.0"},
            "n-shot": {"mmlu": 5},
            "higher_is_better": {"mmlu": True},
        }
        calls = []

        def fake_simple_evaluate(**kwargs):
            calls.append(kwargs.get("batch_size"))
            if len(calls) <= 2 and calls[-1] == 8:
                raise RuntimeError("oom")
            return fake_res

        monkeypatch.setattr("lm_eval.simple_evaluate", fake_simple_evaluate)
        monkeypatch.setattr("lm_eval.utils.make_table", lambda res: "")

        eval_cli._evaluate_tasks_with_retry(
            tasks=["mmlu"],
            hflm=fake_hflm,
            device_str="cpu",
            batch_size=8,
            limit=None,
            retry_times=1,
        )

        assert calls == [8, 1]

    def test_exhausted_retries_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr(
            "lm_eval.simple_evaluate",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("permanent failure")),
        )

        with pytest.raises(RuntimeError, match="Failed to evaluate task 'bad-task'"):
            eval_cli._evaluate_tasks_with_retry(
                tasks=["bad-task"],
                hflm=object(),
                device_str="cpu",
                batch_size=8,
                limit=None,
                retry_times=2,
            )

    def test_multiple_tasks_are_aggregated(self, monkeypatch):
        fake_hflm = object()
        res_a = {
            "results": {"mmlu": {"accuracy": 0.5}},
            "versions": {"mmlu": "1.0"},
            "n-shot": {"mmlu": 5},
            "higher_is_better": {"mmlu": True},
        }
        res_b = {
            "results": {"wikitext": {"word_perplexity": 10.0}},
            "versions": {"wikitext": "1.0"},
            "n-shot": {"wikitext": 0},
            "higher_is_better": {"wikitext": False},
        }

        monkeypatch.setattr("lm_eval.simple_evaluate", lambda **kwargs: res_a if kwargs["tasks"] == "mmlu" else res_b)
        monkeypatch.setattr("lm_eval.utils.make_table", lambda res: "")

        eval_cli._evaluate_tasks_with_retry(
            tasks=["mmlu", "wikitext"],
            hflm=fake_hflm,
            device_str="cpu",
            batch_size=8,
            limit=None,
            retry_times=1,
        )


class TestLoadGgufModelIfNeeded:
    """CPU-only tests for GGUF detection using temporary files/directories."""

    def test_gguf_file_detected_at_file_path(self, monkeypatch, tmp_path):
        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_text("fake")
        fake_model = MagicMock()

        monkeypatch.setattr(eval_cli.os.path, "isfile", lambda value: value == str(gguf_path))
        monkeypatch.setattr(
            eval_cli.os.path, "exists", lambda value: value == str(tmp_path) or value == str(gguf_path)
        )
        monkeypatch.setattr(eval_cli.os, "listdir", lambda value: ["model.gguf"] if value == str(tmp_path) else [])
        monkeypatch.setattr(eval_cli, "get_model_dtype", lambda value="auto": "auto")
        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            "transformers.AutoModelForCausalLM.from_pretrained",
            lambda *args, **kwargs: fake_model,
        )

        model, tokenizer, is_gguf, gguf_file = eval_cli._load_gguf_model_if_needed(
            str(gguf_path), eval_model_dtype="auto"
        )

        assert is_gguf is True
        assert gguf_file == "model.gguf"
        assert tokenizer is not None
        assert model is fake_model

    def test_gguf_file_detected_inside_model_dir(self, monkeypatch, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.gguf").write_text("fake")
        fake_model = MagicMock()

        monkeypatch.setattr(eval_cli.os.path, "isfile", lambda value: value == str(model_dir))
        monkeypatch.setattr(
            eval_cli.os.path, "exists", lambda value: value == str(model_dir) or value == str(model_dir / "model.gguf")
        )
        monkeypatch.setattr(eval_cli.os, "listdir", lambda value: ["model.gguf"])
        monkeypatch.setattr(eval_cli, "get_model_dtype", lambda value="auto": "auto")
        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            "transformers.AutoModelForCausalLM.from_pretrained",
            lambda *args, **kwargs: fake_model,
        )

        model, tokenizer, is_gguf, gguf_file = eval_cli._load_gguf_model_if_needed(
            str(model_dir), eval_model_dtype="auto"
        )

        assert is_gguf is True
        assert gguf_file == "model.gguf"
        assert tokenizer is not None
        assert model is fake_model
