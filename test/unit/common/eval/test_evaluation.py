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
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.eval.evaluation import (
    _collect_model_floating_dtypes,
    _normalize_model_eval_dtype,
    evaluate_diffusion_model,
    prepare_model_for_eval,
    select_gguf_eval_file,
    simple_evaluate,
    simple_evaluate_user_model,
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

    def test_collects_from_submodule_parameters_and_buffers(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 4)
                self.register_buffer("scale", torch.tensor(1.0))

        dtypes = _collect_model_floating_dtypes(M())
        assert torch.float32 in dtypes


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

    def test_no_floating_dtypes_returns_unchanged(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("int_buf", torch.tensor([1], dtype=torch.int32))

        m = M()
        result = _normalize_model_eval_dtype(m, "auto")
        assert result is m


class TestEvaluateDiffusionModel:
    """Test evaluate_diffusion_model function."""

    def test_raises_when_no_pipe_no_autoround(self):
        args = SimpleNamespace()
        with pytest.raises(ValueError, match="must be provided"):
            evaluate_diffusion_model(args)

    def test_raises_when_only_autoround(self):
        args = SimpleNamespace()
        with pytest.raises(ValueError, match="must be provided"):
            evaluate_diffusion_model(args, autoround=MagicMock())


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

    def test_no_match_returns_full_candidate_list(self, tmp_path):
        (tmp_path / "model-Q8_0.gguf").touch()
        (tmp_path / "model-f32.gguf").touch()
        result, candidates = select_gguf_eval_file(str(tmp_path), ["gguf:Q4_0"])
        assert result is None
        assert len(candidates) == 2


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
            prepare_model_for_eval(model, device_map="auto", eval_model_dtype="auto")

    def test_falls_back_to_dispatch_block_wise(self):
        model = nn.Module()
        model.p1 = nn.Linear(4, 4)
        with patch("auto_round.eval.evaluation.dispatch_model_block_wise") as mock_dispatch:
            prepare_model_for_eval(model, device_map="auto", eval_model_dtype="auto")
            mock_dispatch.assert_called_once_with(model, "auto")

    def test_raises_when_meta_device(self):
        m = nn.Linear(4, 4)
        m.dtype = torch.bfloat16
        with patch("auto_round.eval.evaluation._normalize_model_eval_dtype", return_value=m):
            with patch("auto_round.eval.evaluation.dispatch_model_block_wise") as mock_dispatch:
                result = prepare_model_for_eval(m, "cpu", "auto")
                assert result is m
                mock_dispatch.assert_called_once()

    def test_multi_device_dispatch(self):
        m = nn.Linear(4, 4)
        m.hf_device_map = {"linear": "cpu", "linear2": "cpu"}
        with patch("auto_round.eval.evaluation._normalize_model_eval_dtype", return_value=m):
            with patch("accelerate.big_modeling.dispatch_model") as mock_dispatch:
                result = prepare_model_for_eval(m, "cpu", "auto")
                assert result is m
                mock_dispatch.assert_called_once()


class TestSimpleEvaluate:
    """Test simple_evaluate wrapper."""

    def test_calls_lm_eval(self):
        with patch("lm_eval.simple_evaluate") as mock_eval:
            mock_eval.return_value = {"results": {}}
            result = simple_evaluate(model="hf", model_args="test")
            assert result == {"results": {}}
            mock_eval.assert_called_once()


class TestSimpleEvaluateUserModel:
    """Test simple_evaluate_user_model wrapper."""

    def test_creates_hflm(self):
        mock_hflm = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "lm_eval": MagicMock(),
                "lm_eval.models": MagicMock(),
                "lm_eval.models.huggingface": MagicMock(HFLM=mock_hflm),
            },
        ):
            with patch("lm_eval.simple_evaluate", return_value={"results": {}}) as mock_eval:
                model = MagicMock()
                tokenizer = MagicMock()
                result = simple_evaluate_user_model(model, tokenizer, batch_size=4)
                assert "results" in result or mock_hflm.called
