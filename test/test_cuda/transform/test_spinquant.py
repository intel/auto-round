"""Unit tests for SpinQuant/QuaRot rotation implementation.

Tests cover:
  - SpinQuantConfig creation and validation
  - normalize_rotation_config dispatcher (string, dict, object)
  - BaseRotation registry integration
  - Hook lifecycle (registration, tagging, selective removal)
  - Pipeline integration via AutoRound(rotation_config=...)
  - Rotation correctness: R1, R1+R2, R1+R2+R3+R4 produce valid logits
"""

import shutil

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.algorithms.transforms import apply_rotation, normalize_rotation_config
from auto_round.algorithms.transforms.base import BaseRotation, BaseRotationConfig
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig
from auto_round.algorithms.transforms.spinquant.preprocessor import (
    SpinQuantPreprocessor,
    remove_spinquant_hooks_from_model,
)

from ...helpers import generate_prompt, get_model_path

# ═══════════════════════════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpinQuantConfig:
    """Test SpinQuantConfig creation and defaults."""

    def test_default_config(self):
        cfg = SpinQuantConfig()
        assert cfg.r1 is True
        assert cfg.r2 is True
        assert cfg.r3 is False
        assert cfg.r4 is False
        assert cfg.online_r1_rotation is True
        assert cfg.trainable_rotation is False
        assert cfg.trainable_smooth is False

    def test_trainable_config(self):
        """SpinQuant mode = explicitly enable trainable."""
        cfg = SpinQuantConfig(trainable_rotation=True, trainable_smooth=True)
        assert cfg.trainable_rotation is True
        assert cfg.trainable_smooth is True

    def test_selective_rotation_levels(self):
        cfg = SpinQuantConfig(r1=True, r2=True, r3=False, r4=False)
        assert cfg.r1 is True
        assert cfg.r2 is True
        assert cfg.r3 is False
        assert cfg.r4 is False

    def test_is_base_rotation_config(self):
        """SpinQuantConfig should be a BaseRotationConfig subclass."""
        cfg = SpinQuantConfig()
        assert isinstance(cfg, BaseRotationConfig)

    def test_rotation_size(self):
        cfg = SpinQuantConfig(rotation_size=128)
        assert cfg.rotation_size == 128


# ═══════════════════════════════════════════════════════════════════════════════
# normalize_rotation_config Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeRotationConfig:
    """Test the unified config normalizer."""

    def test_string_quarot(self):
        cfg = normalize_rotation_config("quarot")
        assert isinstance(cfg, SpinQuantConfig)
        assert cfg.trainable_rotation is False
        assert cfg.trainable_smooth is False

    def test_string_spinquant(self):
        cfg = normalize_rotation_config("spinquant")
        assert isinstance(cfg, SpinQuantConfig)
        assert cfg.trainable_rotation is True
        assert cfg.trainable_smooth is True

    def test_string_hadamard(self):
        """'hadamard' should return the original RotationConfig, not SpinQuantConfig."""
        cfg = normalize_rotation_config("hadamard")
        assert not isinstance(cfg, SpinQuantConfig)
        assert isinstance(cfg, BaseRotationConfig)

    def test_dict_spinquant(self):
        cfg = normalize_rotation_config(
            {
                "algorithm": "spinquant",
                "r1": True,
                "r2": True,
                "r3": False,
                "r4": False,
            }
        )
        assert isinstance(cfg, SpinQuantConfig)
        assert cfg.r1 is True
        assert cfg.r3 is False

    def test_dict_extra_keys_ignored(self):
        """Unknown keys in dict should be silently filtered."""
        cfg = normalize_rotation_config(
            {
                "algorithm": "spinquant",
                "r1": True,
                "unknown_field": 42,
            }
        )
        assert isinstance(cfg, SpinQuantConfig)
        assert cfg.r1 is True

    def test_none_returns_none(self):
        assert normalize_rotation_config(None) is None

    def test_passthrough_config_object(self):
        original = SpinQuantConfig(r1=True, r2=False)
        result = normalize_rotation_config(original)
        assert result is original

    def test_invalid_string_raises(self):
        with pytest.raises((ValueError, KeyError)):
            normalize_rotation_config("nonexistent_algo")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            normalize_rotation_config(12345)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaseRotationRegistry:
    """Test BaseRotation registry discovery."""

    def test_spinquant_registered(self):
        assert "spinquant" in BaseRotation._REGISTRY

    def test_hadamard_registered(self):
        assert "hadamard" in BaseRotation._REGISTRY

    def test_from_config_spinquant(self):
        cfg = SpinQuantConfig(r1=True, r2=False, r3=False, r4=False, trainable_rotation=False, trainable_smooth=False)
        rotation = BaseRotation.from_config(cfg)
        assert rotation is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Hook Lifecycle Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHookLifecycle:
    """Test that SpinQuant hooks are properly tagged and selectively removed."""

    @pytest.fixture
    def model(self):
        model_name = get_model_path("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        ).cuda()
        model.eval()
        yield model
        del model
        torch.cuda.empty_cache()

    def test_online_r1_hooks_tagged(self, model):
        """Online R1 hooks should have _spinquant_hook=True."""
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

        tagged_hooks = 0
        for module in model.modules():
            for hook in module._forward_pre_hooks.values():
                if getattr(hook, "_spinquant_hook", False):
                    tagged_hooks += 1
        assert tagged_hooks > 0, "No SpinQuant-tagged hooks found"

    def test_remove_only_spinquant_hooks(self, model):
        """remove_spinquant_hooks_from_model should leave non-spinquant hooks."""
        # Register a foreign hook
        foreign_hook_called = [False]

        def foreign_hook(module, input):
            foreign_hook_called[0] = True
            return input

        first_linear = None
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                first_linear = m
                break
        handle = first_linear.register_forward_pre_hook(foreign_hook)

        # Apply spinquant rotation (adds tagged hooks)
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

        # Remove only spinquant hooks
        remove_spinquant_hooks_from_model(model)

        # Foreign hook should still exist
        assert handle.id in first_linear._forward_pre_hooks, "Foreign hook was incorrectly removed"

        # SpinQuant hooks should be gone
        for module in model.modules():
            for hook in module._forward_pre_hooks.values():
                assert not getattr(hook, "_spinquant_hook", False), "SpinQuant hook was not removed"

        handle.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# Rotation Correctness Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRotationCorrectness:
    """Test that rotation produces valid (non-NaN, non-zero) logits."""

    @staticmethod
    def _get_logits(model, tokenizer, text="The capital of France is"):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            return model(**inputs).logits

    @pytest.mark.parametrize(
        "r1,r2,r3,r4,label",
        [
            (True, False, False, False, "R1"),
            (True, True, False, False, "R1+R2"),
            (True, True, True, True, "R1+R2+R3+R4"),
        ],
    )
    def test_rotation_produces_valid_logits(self, r1, r2, r3, r4, label):
        """Rotation configurations should produce valid (non-NaN, non-Inf) logits."""
        model_name = get_model_path("Qwen/Qwen3-0.6B")
        model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
            .cuda()
            .eval()
        )

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
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logits = self._get_logits(model, tokenizer)

        assert not torch.isnan(logits).any(), f"{label} rotation produced NaN logits"
        assert not torch.isinf(logits).any(), f"{label} rotation produced Inf logits"
        assert logits.abs().sum() > 0, f"{label} rotation produced all-zero logits"
        del model
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipelineIntegration:
    """Test AutoRound(rotation_config=...) pipeline integration."""

    save_dir = "./saved_spinquant"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_pipeline_quarot_string(self):
        """AutoRound(rotation_config='quarot') should work end-to-end."""
        model_name = get_model_path("Qwen/Qwen3-0.6B")
        ar = AutoRound(
            model=model_name,
            iters=0,
            seqlen=2,
            scheme="W4A16",
            rotation_config="quarot",
        )
        compressed_model, quantized_model_path = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        generate_prompt(model, tokenizer)

    def test_pipeline_spinquant_config(self):
        """AutoRound(rotation_config=SpinQuantConfig(...)) should work."""
        model_name = get_model_path("Qwen/Qwen3-0.6B")
        cfg = SpinQuantConfig(
            r1=True,
            r2=True,
            r3=False,
            r4=False,
            trainable_rotation=False,
            trainable_smooth=False,
            online_r1_rotation=True,
        )
        ar = AutoRound(
            model=model_name,
            iters=0,
            seqlen=2,
            scheme="W4A16",
            rotation_config=cfg,
        )
        compressed_model, quantized_model_path = ar.quantize_and_save(
            output_dir=self.save_dir + "_cfg", format="auto_round"
        )
        shutil.rmtree(self.save_dir + "_cfg", ignore_errors=True)

    def test_pipeline_dict_config(self):
        """AutoRound(rotation_config={...}) should work."""
        model_name = get_model_path("Qwen/Qwen3-0.6B")
        ar = AutoRound(
            model=model_name,
            iters=0,
            seqlen=2,
            scheme="W4A16",
            rotation_config={
                "algorithm": "spinquant",
                "r1": True,
                "r2": True,
                "r3": False,
                "r4": False,
                "trainable_rotation": False,
                "trainable_smooth": False,
            },
        )
        compressed_model, quantized_model_path = ar.quantize_and_save(
            output_dir=self.save_dir + "_dict", format="auto_round"
        )
        shutil.rmtree(self.save_dir + "_dict", ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Trainable Mode Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainableValidation:
    """Test that trainable mode correctly requires a dataloader."""

    def test_trainable_without_dataloader_raises(self):
        """trainable_rotation=True without dataloader should raise ValueError."""
        model_name = get_model_path("Qwen/Qwen3-0.6B")
        model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
            .cuda()
            .eval()
        )
        cfg = SpinQuantConfig(
            r1=True,
            r2=True,
            r3=False,
            r4=False,
            trainable_rotation=True,
            trainable_smooth=True,
        )
        preprocessor = SpinQuantPreprocessor(model, cfg)
        with pytest.raises(ValueError, match="dataloader required"):
            preprocessor.preprocess(dataloader=None)
        del model
        torch.cuda.empty_cache()

    def test_default_config_no_dataloader_ok(self):
        """Default config (trainable=False) should work without dataloader."""
        model_name = get_model_path("Qwen/Qwen3-0.6B")
        model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
            .cuda()
            .eval()
        )
        cfg = SpinQuantConfig(r1=True, r2=True, r3=False, r4=False)
        preprocessor = SpinQuantPreprocessor(model, cfg)
        # Should not raise — no dataloader needed for fixed Hadamard
        preprocessor.preprocess(dataloader=None)
        del model
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Rotation Equivalence Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRotationEquivalence:
    """Test that rotation is mathematically equivalent (preserves model output).

    All rotation combinations (R1, R2, R3, R4 and their combinations) should
    preserve functional equivalence:
    - R1 (online): activation hook + weight compensation → x·R·(R^T·W)^T = x·W^T
    - R2 (offline): head rotation fused into o_proj/next-layer weights
    - R3 (online): same rotation on Q and K after RoPE → (Q@R)(K@R)^T = Q@K^T
    - R4 (online + offline fuse): activation rotation + down_proj compensation
    """

    MODEL_NAME = None
    TOKENIZER = None
    BASELINE_LOGITS = None

    @classmethod
    def _ensure_baseline(cls):
        """Compute baseline logits once, reuse across all parametrized tests."""
        if cls.BASELINE_LOGITS is not None:
            return
        cls.MODEL_NAME = get_model_path("Qwen/Qwen3-0.6B")
        cls.TOKENIZER = AutoTokenizer.from_pretrained(cls.MODEL_NAME, trust_remote_code=True)
        model = (
            AutoModelForCausalLM.from_pretrained(cls.MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
            .cuda()
            .eval()
        )
        inputs = cls.TOKENIZER("The capital of France is", return_tensors="pt").to(model.device)
        with torch.no_grad():
            cls.BASELINE_LOGITS = model(**inputs).logits.cpu()
        del model
        torch.cuda.empty_cache()

    @staticmethod
    def _get_logits(model, tokenizer, text="The capital of France is"):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            return model(**inputs).logits.cpu()

    @pytest.mark.parametrize(
        "r1,r2,r3,r4,random_r1,random_r2,random_r3,random_r4,label",
        [
            # --- Single rotations (deterministic) ---
            (True, False, False, False, False, False, False, False, "R1"),
            (False, True, False, False, False, False, False, False, "R2"),
            (False, False, True, False, False, False, False, False, "R3"),
            (False, False, False, True, False, False, False, False, "R4"),
            # --- Pairwise combinations (deterministic) ---
            (True, True, False, False, False, False, False, False, "R1+R2"),
            (True, False, True, False, False, False, False, False, "R1+R3"),
            (True, False, False, True, False, False, False, False, "R1+R4"),
            (False, False, True, True, False, False, False, False, "R3+R4"),
            # --- Full combination (deterministic) ---
            (True, True, True, True, False, False, False, False, "R1+R2+R3+R4"),
            # --- Random Hadamard variants ---
            (True, False, False, False, True, False, False, False, "R1-random"),
            (False, True, False, False, False, True, False, False, "R2-random"),
            (False, False, True, False, False, False, True, False, "R3-random"),
            (False, False, False, True, False, False, False, True, "R4-random"),
            (True, True, True, True, True, True, True, True, "R1+R2+R3+R4-all-random"),
        ],
    )
    def test_rotation_equivalence(self, r1, r2, r3, r4, random_r1, random_r2, random_r3, random_r4, label):
        """Rotation should preserve model output (functional equivalence)."""
        self._ensure_baseline()

        model = (
            AutoModelForCausalLM.from_pretrained(self.MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
            .cuda()
            .eval()
        )
        cfg = SpinQuantConfig(
            r1=r1,
            r2=r2,
            r3=r3,
            r4=r4,
            random_r1=random_r1,
            random_r2=random_r2,
            random_r3=random_r3,
            random_r4=random_r4,
            online_r1_rotation=True,
            trainable_rotation=False,
            trainable_smooth=False,
        )
        model = apply_rotation(model, cfg)
        logits = self._get_logits(model, self.TOKENIZER)
        del model
        torch.cuda.empty_cache()

        # Use cosine similarity — robust metric for rotation equivalence on real
        # models.  Online rotation hooks introduce a different float32 computation
        # path (butterfly ops vs direct matmul), causing expected numerical drift
        # that accumulates across 28 layers.  Max absolute diffs of 0.01–0.15 are
        # normal for float32; cosine similarity captures structural preservation.
        cos_sim = F.cosine_similarity(
            self.BASELINE_LOGITS.flatten().unsqueeze(0).float(),
            logits.flatten().unsqueeze(0).float(),
        ).item()
        max_diff = (self.BASELINE_LOGITS - logits).abs().max().item()
        assert cos_sim > 0.9999, (
            f"{label} rotation broke model equivalence: " f"cos_sim = {cos_sim:.6f}, max_diff = {max_diff:.4f}"
        )
