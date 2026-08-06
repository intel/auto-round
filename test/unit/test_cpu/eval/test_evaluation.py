# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.eval.evaluation``."""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from auto_round.eval.evaluation import (
    _collect_model_floating_dtypes,
    _normalize_model_eval_dtype,
    prepare_model_for_eval,
    select_gguf_eval_file,
)


class TestCollectModelFloatingDtypes:
    """Test _collect_model_floating_dtypes."""

    def test_empty_model(self):
        model = nn.Module()
        result = _collect_model_floating_dtypes(model)
        assert result == set()

    def test_single_float32_param(self):
        model = nn.Linear(4, 4)
        model = model.to(torch.float32)
        result = _collect_model_floating_dtypes(model)
        assert torch.float32 in result

    def test_single_bfloat16_param(self):
        model = nn.Linear(4, 4)
        model = model.to(torch.bfloat16)
        result = _collect_model_floating_dtypes(model)
        assert torch.bfloat16 in result

    def test_int_buffer_ignored(self):
        """Integer buffers are not counted as floating point."""
        model = nn.Module()
        model.register_buffer("int_buffer", torch.zeros(4, dtype=torch.long))
        result = _collect_model_floating_dtypes(model)
        assert len(result) == 0

    def test_buffers_included(self):
        model = nn.Module()
        model.register_buffer("my_buffer", torch.randn(4, dtype=torch.float16))
        result = _collect_model_floating_dtypes(model)
        assert torch.float16 in result

    def test_multiple_dtypes(self):
        model = nn.Module()
        model.p1 = nn.Linear(4, 4).to(torch.float32)
        model.p2 = nn.Linear(4, 4).to(torch.float16)
        model.register_buffer("b1", torch.randn(4, dtype=torch.bfloat16))
        result = _collect_model_floating_dtypes(model)
        assert torch.float32 in result
        assert torch.float16 in result
        assert torch.bfloat16 in result


class TestNormalizeModelEvalDtype:
    """Test _normalize_model_eval_dtype."""

    def test_no_floating_point_buffers(self):
        """Model with only integer buffers returns unchanged."""
        model = nn.Module()
        model.register_buffer("int_buffer", torch.zeros(4, dtype=torch.long))
        result = _normalize_model_eval_dtype(model, "float32")
        assert result is model

    def test_auto_with_single_dtype(self):
        model = nn.Linear(4, 4).to(torch.float32)
        result = _normalize_model_eval_dtype(model, "auto")
        assert result is model

    def test_auto_with_mixed_dtypes_converts_to_bfloat16(self):
        model = nn.Module()
        model.p1 = nn.Linear(4, 4).to(torch.float32)
        model.p2 = nn.Linear(4, 4).to(torch.bfloat16)
        result = _normalize_model_eval_dtype(model, "auto")
        # Check that parameters are now bfloat16
        for p in result.parameters():
            assert p.dtype == torch.bfloat16

    def test_auto_with_mixed_no_bfloat16_uses_model_dtype(self):
        model = nn.Module()
        model.p1 = nn.Linear(4, 4).to(torch.float32)
        model.p2 = nn.Linear(4, 4).to(torch.float16)
        result = _normalize_model_eval_dtype(model, "auto")
        assert result is model

    def test_explicit_dtype_matches_no_change(self):
        model = nn.Linear(4, 4).to(torch.float32)
        result = _normalize_model_eval_dtype(model, "float32")
        assert result is model

    def test_explicit_dtype_differs_converts(self):
        model = nn.Linear(4, 4).to(torch.float32)
        result = _normalize_model_eval_dtype(model, "float16")
        # Check parameters are now float16
        for p in result.parameters():
            assert p.dtype == torch.float16


class TestSelectGgufEvalFile:
    """Test select_gguf_eval_file."""

    def test_no_gguf_format(self, tmp_path):
        (tmp_path / "model.bin").touch()
        result, candidates = select_gguf_eval_file(str(tmp_path), ["auto_gptq", "auto_awq"])
        assert result is None
        assert candidates == []

    def test_q4_gguf_file_selected(self, tmp_path):
        # Use uppercase format to match substring check
        (tmp_path / "model-Q4_0.gguf").touch()
        (tmp_path / "model-Q8_0.gguf").touch()
        result, candidates = select_gguf_eval_file(str(tmp_path), ["gguf:Q4_0"])
        assert result == "model-Q4_0.gguf"
        assert "model-Q4_0.gguf" in candidates
        assert "model-Q8_0.gguf" in candidates

    def test_q4_substring_in_filename(self, tmp_path):
        # Q4 (without underscore) matches Q4_0 file
        (tmp_path / "model-Q4_0.gguf").touch()
        (tmp_path / "model-Q8_0.gguf").touch()
        result, candidates = select_gguf_eval_file(str(tmp_path), ["gguf:Q4"])
        assert result == "model-Q4_0.gguf"

    def test_no_matching_but_single_file(self, tmp_path):
        (tmp_path / "model.gguf").touch()
        (tmp_path / "mmproj-model.gguf").touch()
        result, candidates = select_gguf_eval_file(str(tmp_path), ["gguf:Q4_0"])
        assert result == "model.gguf"
        assert "model.gguf" in candidates
        assert "mmproj-model.gguf" not in candidates

    def test_no_match_multiple_files(self, tmp_path):
        (tmp_path / "model-Q4.gguf").touch()
        (tmp_path / "model-Q8.gguf").touch()
        result, candidates = select_gguf_eval_file(str(tmp_path), ["gguf:Q4_0"])
        assert result is None
        assert "model-Q4.gguf" in candidates

    def test_mmproj_excluded(self, tmp_path):
        (tmp_path / "model.gguf").touch()
        (tmp_path / "mmproj-model.gguf").touch()
        result, candidates = select_gguf_eval_file(str(tmp_path), ["gguf"])
        assert result == "model.gguf"
        assert "mmproj-model.gguf" not in candidates

    def test_uppercase_format_matched(self, tmp_path):
        (tmp_path / "model-Q4_0.gguf").touch()
        # lowercase input in format should also work since format is uppercased
        result, candidates = select_gguf_eval_file(str(tmp_path), ["gguf:q4_0"])
        assert result == "model-Q4_0.gguf"

    def test_any_gguf_format(self, tmp_path):
        (tmp_path / "model.gguf").touch()
        result, candidates = select_gguf_eval_file(str(tmp_path), ["gguf"])
        assert result == "model.gguf"


class TestPrepareModelForEval:
    """Test prepare_model_for_eval."""

    def test_normalizes_dtype(self):
        model = nn.Linear(4, 4)
        model = model.to(torch.float32)
        result = prepare_model_for_eval(model, device_map="auto", eval_model_dtype="auto")
        assert result is not None

    def test_handles_hf_device_map(self):
        model = nn.Module()
        model.p1 = nn.Linear(4, 4)
        model.hf_device_map = {"p1": 0}
        with patch("auto_round.utils.dispatch_model_block_wise") as mock_dispatch:
            mock_dispatch.side_effect = ImportError("no accelerate")
            result = prepare_model_for_eval(model, device_map="auto", eval_model_dtype="auto")

    def test_falls_back_to_dispatch_block_wise(self):
        model = nn.Module()
        model.p1 = nn.Linear(4, 4)
        with patch("auto_round.eval.evaluation.dispatch_model_block_wise") as mock_dispatch:
            result = prepare_model_for_eval(model, device_map="auto", eval_model_dtype="auto")
            mock_dispatch.assert_called_once_with(model, "auto")
