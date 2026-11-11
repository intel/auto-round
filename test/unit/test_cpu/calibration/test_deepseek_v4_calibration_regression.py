# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Regressions for native DeepSeek V4 loading and block replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from auto_round.calibration.inputs import preprocess_block_inputs, split_inputs
from auto_round.calibration.llm import LLMCalibrator
from auto_round.calibration.utils import _update_inputs
from auto_round.context.model import ModelContext
from auto_round.parallel.execution import ParallelExecution
from auto_round.utils.model import _is_fp8_model, llm_load_model


class TestDeepseekV4Calibration(unittest.TestCase):
    def test_native_fp8_checkpoint_detected(self):
        for quant_config in ({"quant_method": "fp8"}, SimpleNamespace(quant_method="fp8")):
            config = SimpleNamespace(model_type="deepseek_v4", quantization_config=quant_config)
            with patch("transformers.AutoConfig.from_pretrained", return_value=config):
                self.assertTrue(_is_fp8_model("unused"))

    def test_native_fp8_disables_meta_even_when_forced(self):
        context = object.__new__(ModelContext)
        context.model = "unused"
        for quant_config in ({"quant_method": "fp8"}, SimpleNamespace(quant_method="fp8")):
            config = SimpleNamespace(model_type="deepseek_v4", quantization_config=quant_config)
            with patch("auto_round.context.model.envs.AR_DISK_STREAM_MODEL", True):
                self.assertFalse(context._should_use_meta_skeleton(config))
        with patch("auto_round.context.model.envs.AR_DISK_STREAM_MODEL", True):
            self.assertTrue(context._should_use_meta_skeleton(SimpleNamespace(model_type="other")))

    def test_full_loader_requests_fp8_dequantization(self):
        config = SimpleNamespace(model_type="deepseek_v4", quantization_config={"quant_method": "fp8"})
        model = torch.nn.Linear(16, 16)
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=config),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as loader,
            patch("transformers.AutoTokenizer.from_pretrained"),
            patch("auto_round.utils.model._keep_checkpoint_fp32_params"),
        ):
            loaded, _ = llm_load_model("unused", device="cpu", model_dtype=torch.float32)
        self.assertIs(loaded, model)
        self.assertTrue(loader.call_args.kwargs["quantization_config"].dequantize)

    def test_router_ids_survive_hidden_state_alias(self):
        hidden = [torch.randn(1, 4, 4, 8)]
        tokens = [torch.tensor([[1, 2, 3, 4]])]
        inputs = {"hidden_states": hidden, "input_ids": tokens}
        q_hidden = [hidden[0].clone()]
        with patch("auto_round.context.model.ModelContext", return_value=SimpleNamespace(is_diffusion=False)):
            inputs, q_inputs = _update_inputs(inputs, {"hidden_states": q_hidden, "input_ids": tokens})
        self.assertIs(q_inputs, q_hidden)
        main, others = split_inputs(inputs, "input_ids", is_diffusion=False)
        self.assertIs(main, hidden)
        self.assertIs(others["input_ids"], tokens)
        self.assertNotIn("_ar_router_input_ids", others)

    def test_router_ids_keep_integer_dtype_during_preprocessing(self):
        hidden = [torch.randn(1, 4, 4, 8)]
        tokens = [torch.tensor([[1, 2, 3, 4]])]
        inputs = {"input_ids": hidden, "_ar_router_input_ids": tokens}
        main, others = preprocess_block_inputs(
            inputs,
            model_context=SimpleNamespace(is_diffusion=False, amp=True, amp_dtype=torch.bfloat16),
            compress_context=SimpleNamespace(cache_device="cpu"),
        )
        self.assertEqual(main[0].dtype, torch.bfloat16)
        self.assertEqual(others["input_ids"].dtype, torch.int64)
        torch.testing.assert_close(others["input_ids"], tokens[0])

    def test_dictionary_position_embeddings_cached(self):
        calibrator = object.__new__(LLMCalibrator)
        calibrator.parallel = ParallelExecution()
        calibrator.inputs = {}
        calibrator.batch_dim = 0
        calibrator.batch_size = 1
        calibrator.has_variable_block_shape = False
        calibrator.shared_cache_keys = ("position_embeddings",)
        calibrator._should_stop_cache_forward = lambda name: True
        rope = {"main": (torch.ones(1, 4, 8), torch.zeros(1, 4, 8))}
        capture = calibrator._make_block_forward_func("block")
        with self.assertRaises(NotImplementedError):
            capture(None, torch.randn(1, 4, 4, 8), position_embeddings=rope)
        cached = calibrator.inputs["block"]["position_embeddings"]
        self.assertIsInstance(cached, dict)
        torch.testing.assert_close(cached["main"][0], rope["main"][0])


if __name__ == "__main__":
    unittest.main()
