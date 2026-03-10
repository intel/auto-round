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

import os
from unittest.mock import MagicMock, patch

import pytest


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

    def test_load_gguf_model_non_string_model(self, tiny_opt_model_path):
        """Test with model object (not a string path)."""
        from auto_round.eval.eval_cli import _load_gguf_model_if_needed

        model, tokenizer, is_gguf, gguf_file = _load_gguf_model_if_needed(tiny_opt_model_path)
        assert tokenizer is None
        assert is_gguf is False
        assert gguf_file is None
