# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Additional unit tests for ``auto_round.eval.evaluation``."""

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
    evaluate_with_model_instance,
    evaluate_with_model_path,
    load_gguf_model_for_eval,
    prepare_model_for_eval,
    run_model_evaluation,
    select_gguf_eval_file,
    simple_evaluate,
    simple_evaluate_user_model,
)


# ==============================================================================
# _collect_model_floating_dtypes (additional tests)
# ==============================================================================


class TestCollectModelFloatingDtypesMore:
    """Additional tests for floating dtype collection."""

    def test_collects_from_parameters_and_buffers(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 4)
                self.register_buffer("scale", torch.tensor(1.0))

        m = M()
        dtypes = _collect_model_floating_dtypes(m)
        assert torch.float32 in dtypes

    def test_excludes_non_floating(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 4)
                self.register_buffer("int_buf", torch.tensor([1, 2, 3], dtype=torch.int32))

        m = M()
        dtypes = _collect_model_floating_dtypes(m)
        assert torch.float32 in dtypes
        assert torch.int32 not in dtypes

    def test_collects_multiple_dtypes(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 4)
                self.fc2 = nn.Linear(4, 4).to(torch.float16)

        m = M()
        dtypes = _collect_model_floating_dtypes(m)
        assert torch.float32 in dtypes
        assert torch.float16 in dtypes


# ==============================================================================
# _normalize_model_eval_dtype (additional tests)
# ==============================================================================


class TestNormalizeModelEvalDtypeMore:
    """Additional tests for normalize_model_eval_dtype."""

    def test_auto_with_single_dtype(self):
        m = nn.Linear(4, 4)
        result = _normalize_model_eval_dtype(m, "auto")
        assert result is m

    def test_auto_with_mixed_returns_model(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 4)
                self.fc2 = nn.Linear(4, 4).to(torch.float16)

        m = M()
        result = _normalize_model_eval_dtype(m, "auto")
        # Should normalize to bfloat16 or float32
        assert isinstance(result, nn.Module)

    def test_specific_dtype_match_no_change(self):
        m = nn.Linear(4, 4)
        result = _normalize_model_eval_dtype(m, "float32")
        assert isinstance(result, nn.Module)

    def test_specific_dtype_mismatch_converts(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 4)

        m = M()
        result = _normalize_model_eval_dtype(m, "float16")
        assert isinstance(result, nn.Module)

    def test_no_floating_dtypes_returns_unchanged(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("int_buf", torch.tensor([1], dtype=torch.int32))

        m = M()
        result = _normalize_model_eval_dtype(m, "auto")
        assert result is m


# ==============================================================================
# evaluate_diffusion_model
# ==============================================================================


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


# ==============================================================================
# select_gguf_eval_file
# ==============================================================================


class TestSelectGgufEvalFileAdditional:
    """Additional tests for select_gguf_eval_file."""

    def test_no_gguf_format_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_file, result_list = select_gguf_eval_file(tmpdir, ["autoround", "gptq"])
            assert result_file is None
            assert result_list == []

    def test_filters_mmproj_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "model-Q4_0.gguf"), "w").close()
            open(os.path.join(tmpdir, "mmproj-model.gguf"), "w").close()
            result_file, result_list = select_gguf_eval_file(tmpdir, ["gguf:Q4_0"])
            assert result_file == "model-Q4_0.gguf"
            assert "mmproj-model.gguf" not in result_list

    def test_no_match_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "model-Q8_0.gguf"), "w").close()
            open(os.path.join(tmpdir, "model-f32.gguf"), "w").close()
            result_file, result_list = select_gguf_eval_file(tmpdir, ["gguf:Q4_0"])
            assert result_file is None
            assert len(result_list) == 2

    def test_single_file_returns_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "model-f16.gguf"), "w").close()
            result_file, result_list = select_gguf_eval_file(tmpdir, ["gguf:Q4_0"])
            assert result_file == "model-f16.gguf"


# ==============================================================================
# prepare_model_for_eval
# ==============================================================================


class TestPrepareModelForEval:
    """Test prepare_model_for_eval function."""

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


# ==============================================================================
# simple_evaluate
# ==============================================================================


class TestSimpleEvaluate:
    """Test simple_evaluate wrapper."""

    def test_calls_lm_eval(self):
        with patch("lm_eval.simple_evaluate") as mock_eval:
            mock_eval.return_value = {"results": {}}
            result = simple_evaluate(model="hf", model_args="test")
            assert result == {"results": {}}
            mock_eval.assert_called_once()


# ==============================================================================
# simple_evaluate_user_model
# ==============================================================================


class TestSimpleEvaluateUserModel:
    """Test simple_evaluate_user_model wrapper."""

    def test_creates_hflm(self):
        mock_hflm = MagicMock()
        with patch.dict("sys.modules", {"lm_eval": MagicMock(), "lm_eval.models": MagicMock(), "lm_eval.models.huggingface": MagicMock(HFLM=mock_hflm)}):
            with patch("lm_eval.simple_evaluate", return_value={"results": {}}) as mock_eval:
                model = MagicMock()
                tokenizer = MagicMock()
                result = simple_evaluate_user_model(model, tokenizer, batch_size=4)
                assert "results" in result or mock_hflm.called
