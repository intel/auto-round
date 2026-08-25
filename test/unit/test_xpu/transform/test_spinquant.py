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

"""XPU CI tests for the SpinQuant/QuaRot rotation transform.

XPU counterpart of test/test_cuda/transform/test_spinquant.py. Covers:
- Rotation correctness: R1, R1+R2, R1+R2+R3+R4 produce valid logits on XPU
- Rotation equivalence: rotation preserves model output (cosine similarity)
- Hook lifecycle: SpinQuant-tagged hooks are registered and selectively removed
- Pipeline integration via AutoRound(rotation_config="quarot") on XPU
"""

import shutil

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.algorithms.transforms import apply_rotation
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig
from auto_round.algorithms.transforms.spinquant.preprocessor import remove_spinquant_hooks_from_model

from test.helpers import generate_prompt

DEVICE = "xpu"

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU is not available on this host",
)

PROMPT = "The capital of France is"


def _load_model(model_path, dtype=torch.float32):
    return AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(DEVICE).eval()


def _get_logits(model, tokenizer, text=PROMPT):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        return model(**inputs).logits.cpu()


class TestRotationCorrectnessXPU:
    """Rotation configurations should produce valid (non-NaN, non-Inf) logits on XPU."""

    @pytest.mark.parametrize(
        "r1,r2,r3,r4,label",
        [
            (True, False, False, False, "R1"),
            (True, True, False, False, "R1+R2"),
            (True, True, True, True, "R1+R2+R3+R4"),
        ],
    )
    def test_rotation_produces_valid_logits(self, tiny_qwen_model_path, r1, r2, r3, r4, label):
        model = _load_model(tiny_qwen_model_path)
        cfg = SpinQuantConfig(
            r1=r1,
            r2=r2,
            r3=r3,
            r4=r4,
            online_r1_rotation=True,
            trainable_rotation=False,
            trainable_smooth=False,
        )
        model = apply_rotation(model, cfg)
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_model_path)
        logits = _get_logits(model, tokenizer)

        assert not torch.isnan(logits).any(), f"{label} rotation produced NaN logits"
        assert not torch.isinf(logits).any(), f"{label} rotation produced Inf logits"
        assert logits.abs().sum() > 0, f"{label} rotation produced all-zero logits"
        del model
        torch.xpu.empty_cache()


class TestRotationEquivalenceXPU:
    """Rotation should preserve model output (functional equivalence) on XPU.

    R1 (online): activation hook + weight compensation → x·R·(R^T·W)^T = x·W^T
    R2 (offline): head rotation fused into o_proj/next-layer weights
    R3 (online): same rotation on Q and K after RoPE → (Q@R)(K@R)^T = Q@K^T
    R4 (online + offline fuse): activation rotation + down_proj compensation
    """

    @pytest.mark.parametrize(
        "r1,r2,r3,r4,label",
        [
            (True, False, False, False, "R1"),
            (False, True, False, False, "R2"),
            (False, False, True, False, "R3"),
            (False, False, False, True, "R4"),
            (True, True, True, True, "R1+R2+R3+R4"),
        ],
    )
    def test_rotation_equivalence(self, tiny_qwen_model_path, r1, r2, r3, r4, label):
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_model_path)

        baseline_model = _load_model(tiny_qwen_model_path)
        baseline_logits = _get_logits(baseline_model, tokenizer)
        del baseline_model
        torch.xpu.empty_cache()

        model = _load_model(tiny_qwen_model_path)
        cfg = SpinQuantConfig(
            r1=r1,
            r2=r2,
            r3=r3,
            r4=r4,
            online_r1_rotation=True,
            trainable_rotation=False,
            trainable_smooth=False,
        )
        model = apply_rotation(model, cfg)
        logits = _get_logits(model, tokenizer)
        del model
        torch.xpu.empty_cache()

        cos_sim = F.cosine_similarity(
            baseline_logits.flatten().unsqueeze(0).float(),
            logits.flatten().unsqueeze(0).float(),
        ).item()
        max_diff = (baseline_logits - logits).abs().max().item()
        assert (
            cos_sim > 0.9999
        ), f"{label} rotation broke model equivalence: cos_sim = {cos_sim:.6f}, max_diff = {max_diff:.4f}"


class TestHookLifecycleXPU:
    """SpinQuant hooks are properly tagged and selectively removed on XPU."""

    def test_remove_only_spinquant_hooks(self, tiny_qwen_model_path):
        model = _load_model(tiny_qwen_model_path, dtype=torch.float16)

        # Register a foreign hook
        def foreign_hook(module, input):
            return input

        first_linear = None
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                first_linear = m
                break
        handle = first_linear.register_forward_pre_hook(foreign_hook)

        cfg = SpinQuantConfig(
            r1=True,
            r2=False,
            r3=False,
            r4=False,
            online_r1_rotation=True,
            trainable_rotation=False,
            trainable_smooth=False,
        )
        model = apply_rotation(model, cfg)

        # SpinQuant-tagged hooks should exist
        tagged_hooks = 0
        for module in model.modules():
            for hook in module._forward_pre_hooks.values():
                if getattr(hook, "_spinquant_hook", False):
                    tagged_hooks += 1
        assert tagged_hooks > 0, "No SpinQuant-tagged hooks found"

        # Remove only spinquant hooks
        remove_spinquant_hooks_from_model(model)

        # Foreign hook should still exist
        assert handle.id in first_linear._forward_pre_hooks, "Foreign hook was incorrectly removed"

        # SpinQuant hooks should be gone
        for module in model.modules():
            for hook in module._forward_pre_hooks.values():
                assert not getattr(hook, "_spinquant_hook", False), "SpinQuant hook was not removed"

        handle.remove()
        del model
        torch.xpu.empty_cache()


class TestPipelineIntegrationXPU:
    """AutoRound(rotation_config='quarot') end-to-end on XPU."""

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_pipeline_quarot_string(self, tiny_qwen_model_path):
        """AutoRound(rotation_config='quarot') should work end-to-end on XPU."""
        ar = AutoRound(
            model=tiny_qwen_model_path,
            iters=0,
            seqlen=8,
            nsamples=2,
            scheme="W4A16",
            rotation_config="quarot",
            device_map=DEVICE,
        )
        _, quantized_model_path = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, dtype="auto", device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        output = generate_prompt(model, tokenizer, device=DEVICE)
        assert len(output) > 0, "Quantized model should produce non-empty output"
        del model
        torch.xpu.empty_cache()
