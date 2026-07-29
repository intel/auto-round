# Copyright (c) 2026 Intel Corporation
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

"""CPU CI tests for AWQ (Activation-Aware Weight Quantization) algorithm.

Covers:
- Normal LLM (OPT-125m): W4A16 quantization, inference, export args verification
- INT W8A8 export via llm_compressor format with config validation
- Tiny Qwen MoE: dynamic smoothing, quantized layer checks, quant config saving
"""

import json
import os
import shutil

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...helpers import generate_prompt, get_model_path, opt_name_or_path, save_tiny_model


class TestAWQWeightClip:
    """AWQ weight-clip option (issue #1854).

    Validates the ``apply_clip`` AWQ preprocessing combined with downstream
    block quantizers, covering the two extensibility scenarios:
      - clip + RTN  (clip as the weight range, then round-to-nearest)
      - clip + SignRound (clip as initialization, then SignRound tuning)
    """

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_awq_clip_then_rtn(self, tiny_opt_model_path):
        """AWQ smooth+clip → RTN: produces a valid W4 model that can generate."""
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        ar = AutoRound(
            tiny_opt_model_path,
            alg_configs=[
                AWQConfig(
                    bits=4, group_size=128, sym=True, apply_clip=True, n_grid=2, clip_n_grid=2, clip_n_sample_token=8
                ),
                RTNConfig(disable_opt_rtn=True),
            ],
            nsamples=2,
            seqlen=8,
            batch_size=2,
        )
        model, layer_config = ar.quantize()

        assert model is not None
        assert len(layer_config) > 0
        for name, cfg in layer_config.items():
            assert cfg["bits"] == 4, f"Layer {name} expected bits=4, got {cfg['bits']}"

        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)
        output = generate_prompt(model, tokenizer, device="cpu")
        assert len(output) > 0, "Clipped model should produce non-empty output"

    def test_awq_clip_as_init_signround(self, tiny_opt_model_path):
        """clip_as_init: clip is kept on the model context and initializes SignRound's range."""
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        ar = AutoRound(
            tiny_opt_model_path,
            alg_configs=[
                AWQConfig(
                    bits=4,
                    group_size=128,
                    sym=True,
                    apply_clip=True,
                    clip_as_init=True,
                    n_grid=2,
                    clip_n_grid=2,
                    clip_n_sample_token=8,
                ),
                SignRoundConfig(iters=1),
            ],
            nsamples=2,
            seqlen=8,
            batch_size=2,
            enable_torch_compile=False,  # disable torch.compile to ensure CI pass, cannot reproduce in local env.
        )
        model, layer_config = ar.quantize()

        assert model is not None
        assert len(layer_config) > 0
        for name, cfg in layer_config.items():
            assert cfg["bits"] == 4, f"Layer {name} expected bits=4, got {cfg['bits']}"

        # The searched clip magnitudes are kept on the model context.
        clip_values = getattr(ar.model_context, "awq_clip_values", {})
        assert len(clip_values) > 0, "clip_as_init should record per-group clip values on the model context"

        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)
        output = generate_prompt(model, tokenizer, device="cpu")
        assert len(output) > 0, "clip_as_init model should produce non-empty output"


class TestAWQUseV2ScaleSearch:
    """Unit tests for AWQ's ``use_v2_scale_search`` detection and dispatch.

    The flag is True whenever the terminal block quantizer resolves to
    ``SignRoundV2Quantizer``. The per-data-type init_scale injection (int/mx/nv,
    sym only) is then handled by ``search_optimized_init_scale`` inside the QDQ
    path. Regression guard: the block-quantizer config does not expose
    ``_alg_cls``, so detection must go through the pipeline registry, not an
    ``_alg_cls`` string comparison (which always evaluated False before the fix).
    """

    @staticmethod
    def _make_compressor(block_config):
        import types

        return types.SimpleNamespace(quantize_config=block_config, alg_configs=[block_config])

    @staticmethod
    def _awq_transform():
        from auto_round.algorithms.transforms.awq.base import AWQTransform
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        return AWQTransform(AWQConfig(n_grid=1, apply_smooth=True))

    @staticmethod
    def _signroundv2_config(data_type=None):
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
        from auto_round.algorithms.registry import normalize_algorithm_config

        cfg = normalize_algorithm_config(SignRoundConfig(iters=1, enable_alg_ext=True))
        assert type(cfg).__name__ == "SignRoundV2Config"
        if data_type is not None:
            cfg.data_type = data_type
        return cfg

    # TODO weiwei monkey patch,recover better
    # def test_block_v2(self):
    #     """An RTN block quantizer must NOT be detected as V2."""
    #     from auto_round.algorithms.quantization.rtn.config import RTNConfig
    #
    #     q = self._awq_transform()
    #     compressor = self._make_compressor(RTNConfig())
    #     assert q._qdq_tool._block_quantizer_is_signroundv2(compressor) is False
    #
    #     compressor = self._make_compressor(self._signroundv2_config(data_type="mx_fp"))
    #     assert q._qdq_tool._block_quantizer_is_signroundv2(compressor) is True

    def test_use_v2_false_for_non_v2_block(self):
        """Gate is False when the block is not SignRoundV2, regardless of dtype."""
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        q = self._awq_transform()
        block = SignRoundConfig(iters=1)
        block.data_type = "mx_fp"
        compressor = self._make_compressor(block)
        assert q._qdq_tool._block_quantizer_is_signroundv2(compressor) is False

    def test_init_scale_dispatch_by_data_type(self):
        """``search_optimized_init_scale`` injects only for sym int/mx/nv."""

        from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, search_optimized_init_scale

        for dt, gs in (("mx_fp4", 32),):
            weight = torch.randn(32, 64)
            weight_reshape, _, _ = reshape_pad_tensor_by_group_size(weight, gs)
            init_scale = search_optimized_init_scale(weight_reshape, dt, 4, None)
            assert init_scale is not None, dt
            assert init_scale.shape[0] == weight_reshape.shape[0]

        # asym int and *_dq are not part of the optimized init-scale path.
        assert search_optimized_init_scale(torch.randn(4, 128), "int_asym", 4, None) is None
        assert search_optimized_init_scale(torch.randn(4, 128), "int_sym_dq", 4, None) is None
