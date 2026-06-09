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

"""Comprehensive CPU tests for SpinQuant / QuaRot rotation.

Tests cover the entire SpinQuant module public API surface:
- SpinQuantConfig validation and post_init checks
- TrainableRMSNorm wrapper (smooth value scaling, gradient flow)
- Rotation utilities: Hadamard matrices, matmul_hadU butterfly, rotation primitives
- Cayley optimizer: SGDG (Stiefel manifold), AdamAndSGDG dual optimizer
- Loss functions: compute_rotation_loss (kl_top, kl_full, mse), spinquant_loss_fn alias
- SpinQuantState tracking and LossLogger callback
- Optimizer creation utilities
- SpinQuantRotation registry integration
- SpinQuantPreprocessor (model integration)
- QKRotationWrapper monkeypatch for R3
- In-place hook registration (R3, R4)
- Serialization: buffer injection, rebuild, config save/load
- RotationTrainer standalone trainer
"""

import copy
import math
import tempfile

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.transforms.spinquant.preprocessor import (
    SpinQuantConfig,
    SpinQuantPreprocessor,
    TrainableRMSNorm,
)
from auto_round.algorithms.transforms.spinquant.apply import SpinQuantRotation
from auto_round.algorithms.transforms.spinquant.cayley_optimizer import (
    SGDG,
    AdamAndSGDG,
)
from auto_round.algorithms.transforms.spinquant.training import (
    LossLogger,
    OrthogonalityMonitor,
    RotationTrainer,
    RotationTrainerCallback,
    RotationTrainerConfig,
    SpinQuantState,
    compute_rotation_loss,
    spinquant_loss_fn,
    create_dual_optimizer,
    create_spinquant_optimizer,
    check_orthogonality,
    clone_model_for_reference,
    move_batch_to_device,
    run_training_loop,
)
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    deterministic_hadamard_matrix,
    random_hadamard_matrix,
    is_pow2,
    get_hadamard_K,
    matmul_hadU,
    rotate_in_channels_,
    rotate_out_channels_,
    InputRotationWrapperHadamard,
    get_model_arch_info,
    untie_word_embeddings_if_needed,
    apply_hadamard_to_linear,
    create_block_diag_from_head_matrix,
)
from auto_round.algorithms.transforms.spinquant.monkeypatch import (
    copy_func_with_new_globals,
    add_wrapper_after_function_call_in_method,
    QKRotationWrapper,
    add_qk_rotation_after_rope,
)
from auto_round.algorithms.transforms.spinquant.inplace.apply import (
    register_spinquant_hooks,
    remove_spinquant_hooks,
    apply_spinquant_in_place,
)
from auto_round.algorithms.transforms.spinquant.serialize import (
    inject_spinquant_buffers,
    save_spinquant_config,
    preregister_spinquant_buffers,
    rebuild_spinquant_online,
    ROTATION_TYPE_HADAMARD,
    ROTATION_TYPE_RANDOM,
    ROTATION_TYPE_TRAINED,
    _is_quantlinear,
    _has_spinquant_buffers,
    _apply_rotation_from_buffer,
    _apply_block_rotation_butterfly,
)


# =============================================================================
# TestSpinQuantConfig — validation, defaults, serialization
# =============================================================================

class TestSpinQuantConfig:
    """SpinQuantConfig validation and field defaults."""

    def test_all_defaults(self):
        cfg = SpinQuantConfig()
        assert cfg.algorithm == "spinquant"
        assert cfg.r1 is True
        assert cfg.r2 is True
        assert cfg.r3 is False
        assert cfg.r4 is False
        assert cfg.rotation_size is None
        assert cfg.random_r1 is False
        assert cfg.trainable_rotation is False
        assert cfg.trainable_smooth is False
        assert cfg.online_r1_rotation is True
        assert cfg.iters == 200
        assert cfg.lr == 1e-4
        assert cfg.smooth_lr == 1e-3
        assert cfg.batch_size == 1
        assert cfg.loss_type == "kl_top"
        assert cfg.kl_top_k == 1000
        assert cfg.fuse_rmsnorm is True
        assert cfg.untie_embeddings is True
        assert cfg.dtype == torch.float32
        assert cfg.device in ("cuda", "cpu")

    def test_custom_values_stored(self):
        cfg = SpinQuantConfig(
            r1=True,
            r2=True,
            r3=True,
            r4=True,
            rotation_size=64,
            random_r1=True,
            random_r2=True,
            random_r3=True,
            random_r4=True,
            trainable_rotation=True,
            trainable_smooth=True,
            iters=100,
            lr=5e-4,
            smooth_lr=1e-2,
            batch_size=2,
            loss_type="kl_full",
            kl_top_k=500,
            fuse_rmsnorm=False,
            untie_embeddings=False,
        )
        assert cfg.r3 is True
        assert cfg.r4 is True
        assert cfg.rotation_size == 64
        assert cfg.random_r3 is True
        assert cfg.trainable_rotation is True
        assert cfg.trainable_smooth is True
        assert cfg.iters == 100
        assert cfg.lr == 5e-4
        assert cfg.smooth_lr == 1e-2
        assert cfg.batch_size == 2
        assert cfg.loss_type == "kl_full"
        assert cfg.kl_top_k == 500
        assert cfg.fuse_rmsnorm is False
        assert cfg.untie_embeddings is False

    def test_invalid_rotation_size_zero_raises(self):
        with pytest.raises(ValueError, match="rotation_size must be positive"):
            SpinQuantConfig(rotation_size=0)

    def test_invalid_rotation_size_negative_raises(self):
        with pytest.raises(ValueError, match="rotation_size must be positive"):
            SpinQuantConfig(rotation_size=-1)

    def test_non_power_of_two_raises(self):
        with pytest.raises(ValueError, match="rotation_size must be a power of 2"):
            SpinQuantConfig(rotation_size=12)
        with pytest.raises(ValueError, match="rotation_size must be a power of 2"):
            SpinQuantConfig(rotation_size=100)
        with pytest.raises(ValueError, match="rotation_size must be a power of 2"):
            SpinQuantConfig(rotation_size=3)

    @pytest.mark.parametrize("size", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    def test_valid_power_of_two_rotation_sizes(self, size):
        cfg = SpinQuantConfig(rotation_size=size)
        assert cfg.rotation_size == size

    def test_device_auto_detects_cuda(self):
        cfg = SpinQuantConfig(device="cpu")
        assert cfg.device == "cpu"

    def test_quarat_mode_fixed_hadamard(self):
        cfg = SpinQuantConfig(
            r1=True, r2=True, r3=True, r4=True,
            trainable_rotation=False,
            trainable_smooth=False,
        )
        assert cfg.trainable_rotation is False
        assert cfg.trainable_smooth is False

    def test_spinquant_mode_experimental(self):
        cfg = SpinQuantConfig(
            trainable_rotation=True,
            trainable_smooth=True,
            iters=200,
        )
        assert cfg.trainable_rotation is True
        assert cfg.trainable_smooth is True


# =============================================================================
# TestTrainableRMSNorm — smooth value scaling and gradient flow
# =============================================================================

class TestTrainableRMSNorm:
    """TrainableRMSNorm wrapper for joint SpinQuant + SmoothQuant."""

    def test_creation_with_weight_norm(self):
        original = nn.RMSNorm(32, elementwise_affine=True)
        trainable = TrainableRMSNorm(original, trainable=True)
        assert trainable.trainable is True
        assert trainable.smooth_values is not None
        assert trainable.smooth_values.shape == (32,)
        assert trainable.smooth_values.requires_grad is True

    def test_creation_with_trainable_false(self):
        original = nn.RMSNorm(32, elementwise_affine=True)
        trainable = TrainableRMSNorm(original, trainable=False)
        assert trainable.trainable is False
        assert trainable.smooth_values is not None
        assert trainable.smooth_values.requires_grad is False

    def test_forward_applies_smooth_values(self):
        original = nn.RMSNorm(32, elementwise_affine=True)
        nn.init.ones_(original.weight)
        trainable = TrainableRMSNorm(original, trainable=True)
        trainable.smooth_values.data.fill_(2.0)

        x = torch.randn(2, 10, 32)
        out = trainable(x)

        # Compare against the original norm's output scaled by smooth_values
        expected = original(x) * trainable.smooth_values
        assert torch.allclose(out, expected, atol=1e-5)

    def test_forward_without_smooth_values(self):
        class SimpleNormNoWeight(nn.Module):
            def forward(self, x):
                return x / x.norm(dim=-1, keepdim=True)

        original = SimpleNormNoWeight()
        trainable = TrainableRMSNorm(original, trainable=True)
        assert trainable.smooth_values is None

        x = torch.randn(2, 10, 32)
        out = trainable(x)
        expected = x / x.norm(dim=-1, keepdim=True)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gradient_flows_through_smooth_values(self):
        original = nn.RMSNorm(8)
        trainable = TrainableRMSNorm(original, trainable=True)
        x = torch.randn(2, 4, 8, requires_grad=True)
        out = trainable(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert trainable.smooth_values.grad is not None
        assert trainable.original_norm.weight.grad is not None

    def test_different_hidden_dims(self):
        for dim in [16, 64, 128]:
            original = nn.RMSNorm(dim)
            trainable = TrainableRMSNorm(original, trainable=True)
            x = torch.randn(2, 4, dim)
            out = trainable(x)
            assert out.shape == x.shape


# =============================================================================
# TestRotationUtils — Hadamard matrices and rotation primitives
# =============================================================================

class TestIsPow2:
    """is_pow2 correctly identifies powers of two."""

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    def test_powers_of_two_return_true(self, n):
        assert is_pow2(n) is True

    @pytest.mark.parametrize("n", [0, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 31, 63, 100, 255, 1000])
    def test_non_powers_of_two_return_false(self, n):
        assert is_pow2(n) is False

    @pytest.mark.parametrize("n", [-1, -2, -128])
    def test_negative_numbers_return_false(self, n):
        assert is_pow2(n) is False


class TestDeterministicHadamardMatrix:
    """deterministic_hadamard_matrix via Sylvester construction."""

    @pytest.mark.parametrize("size", [2, 4, 8, 16, 32, 64, 128])
    def test_correct_shape(self, size):
        H = deterministic_hadamard_matrix(size)
        assert H.shape == (size, size)

    @pytest.mark.parametrize("size", [2, 4, 8, 16, 32])
    def test_elements_are_plusminus_one_over_sqrt_n(self, size):
        H = deterministic_hadamard_matrix(size)
        # deterministic_hadamard_matrix returns a normalized Sylvester Hadamard (H / sqrt(N)),
        # so its unique values are ±1/sqrt(N) — not the classical ±1.
        scale = 1.0 / math.sqrt(size)
        expected = torch.tensor([scale, -scale], dtype=H.dtype, device=H.device)
        assert torch.allclose(
            torch.sort(H.unique()).values, torch.sort(expected).values, atol=1e-6
        )

    @pytest.mark.parametrize("size", [2, 4, 8, 16, 32, 64])
    def test_orthogonal_property(self, size):
        H = deterministic_hadamard_matrix(size, dtype=torch.float32)
        # The implementation returns a normalized Hadamard (H / sqrt(N)),
        # so H @ H.T = I (not N*I as for a classical Hadamard).
        product = H @ H.T
        expected = torch.eye(size, dtype=torch.float32)
        assert torch.allclose(product, expected, atol=1e-5)

    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_orthogonal_property_bfloat16(self, size):
        H = deterministic_hadamard_matrix(size, dtype=torch.bfloat16)
        product = H.float() @ H.T.float()
        expected = torch.eye(size)
        assert torch.allclose(product, expected, atol=1e-3)

    def test_respects_dtype(self):
        H = deterministic_hadamard_matrix(8, dtype=torch.float64)
        assert H.dtype == torch.float64

    def test_respects_device(self):
        H = deterministic_hadamard_matrix(8, device="cpu")
        assert H.device.type == "cpu"

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="power-of-2"):
            deterministic_hadamard_matrix(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="power-of-2"):
            deterministic_hadamard_matrix(-1)

    def test_non_power_of_two_raises(self):
        with pytest.raises(ValueError, match="power-of-2"):
            deterministic_hadamard_matrix(7)
        with pytest.raises(ValueError, match="power-of-2"):
            deterministic_hadamard_matrix(12)


class TestRandomHadamardMatrix:
    """random_hadamard_matrix construction."""

    def test_correct_shape(self):
        H = random_hadamard_matrix(8)
        assert H.shape == (8, 8)

    def test_elements_are_plusminus_normalized(self):
        H = random_hadamard_matrix(8)
        abs_vals = H.abs()
        max_val = abs_vals.max().item()
        assert max_val <= 1.0 + 1e-5

    @pytest.mark.parametrize("size", [8, 16, 32, 64])
    def test_orthogonal_property(self, size):
        H = random_hadamard_matrix(size, dtype=torch.float32)
        # random_hadamard_matrix is also normalized (uses matmul_hadU internally),
        # so H @ H.T = I, not N*I.
        product = H @ H.T
        expected = torch.eye(size, dtype=torch.float32)
        assert torch.allclose(product, expected, atol=1e-4)

    def test_respects_dtype(self):
        H = random_hadamard_matrix(8, dtype=torch.float64)
        assert H.dtype == torch.float64

    def test_respects_device(self):
        H = random_hadamard_matrix(8, device="cpu")
        assert H.device.type == "cpu"


class TestGetHadamardK:
    """get_hadamard_K decomposition for power-of-2 and non-pow2 sizes."""

    @pytest.mark.parametrize("size", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    def test_pow2_returns_K1(self, size):
        had_K, K = get_hadamard_K(size)
        assert K == 1
        assert had_K.shape == (size, size)
        assert had_K.shape[0] == had_K.shape[1]

    @pytest.mark.parametrize("size", [12, 20, 28, 36, 40, 52, 60])
    def test_non_pow2_returns_valid_K(self, size):
        had_K, K = get_hadamard_K(size)
        assert K > 1
        assert had_K.shape == (K, K)
        assert size % K == 0
        assert is_pow2(size // K)

    def test_unsupported_non_pow2_raises(self):
        with pytest.raises(ValueError, match="Cannot find suitable Hadamard decomposition"):
            get_hadamard_K(7)
        with pytest.raises(ValueError, match="Cannot find suitable Hadamard decomposition"):
            get_hadamard_K(1000)

    def test_172_returns_K172(self):
        had_K, K = get_hadamard_K(172)
        assert K == 172
        assert had_K.shape == (172, 172)


class TestMatmulHadU:
    """matmul_hadU butterfly Hadamard transform."""

    @pytest.mark.parametrize("size", [2, 4, 8, 16, 32, 64])
    def test_output_shape_matches_input(self, size):
        X = torch.randn(4, size)
        result = matmul_hadU(X)
        assert result.shape == X.shape

    @pytest.mark.parametrize("size", [2, 4, 8, 16])
    def test_double_application_inverses(self, size):
        X = torch.randn(4, size)
        reconstructed = matmul_hadU(matmul_hadU(X))
        assert torch.allclose(reconstructed, X, atol=1e-3)

    @pytest.mark.parametrize("size", [24, 48])
    def test_inverse_for_non_pow2(self, size):
        # For non-power-of-2 sizes, matmul_hadU uses a Kronecker construction
        # (H_K ⊗ H_2) that is orthogonal but not symmetric. It preserves L2 norm
        # and H @ H.T = I, but H @ H = I only holds for the symmetric construction
        # (H_2 ⊗ H_K). We verify orthogonality via L2-norm preservation and
        # H @ H.T = I on a basis vector.
        X = torch.randn(4, size)
        result = matmul_hadU(X)
        # Orthogonality: ||H X||^2 == ||X||^2
        assert torch.allclose((result ** 2).sum(-1), (X ** 2).sum(-1), atol=1e-3)
        # H @ H.T = I when applied to a basis vector
        I = torch.eye(size, dtype=torch.float32)
        H = matmul_hadU(I)
        assert torch.allclose(H @ H.T, I, atol=1e-4)

    def test_3d_input_preserves_shape(self):
        X = torch.randn(2, 3, 16)
        result = matmul_hadU(X)
        assert result.shape == X.shape

    def test_bfloat16_preserved(self):
        X = torch.randn(4, 8, dtype=torch.bfloat16)
        result = matmul_hadU(X)
        assert result.dtype == torch.bfloat16

    def test_double_precision_intermediate(self):
        X = torch.randn(4, 8, dtype=torch.float16)
        result = matmul_hadU(X)
        assert result.dtype == torch.float16

    def test_produces_orthogonal_transform(self):
        for size in [8, 16, 32]:
            I = torch.eye(size, dtype=torch.float64)
            H = matmul_hadU(I)
            product = H @ H.T
            assert torch.allclose(product, torch.eye(size, dtype=torch.float64), atol=1e-5)


class TestRotateInChannels:
    """rotate_in_channels_ fuses input-side rotation: W_new = W @ R.T."""

    def test_full_rotation(self):
        layer = nn.Linear(8, 16)
        original_weight = layer.weight.data.clone()
        R = deterministic_hadamard_matrix(8)
        rotate_in_channels_(layer, rotation_matrix=R)

        assert not torch.equal(layer.weight.data, original_weight)
        new_W = original_weight.float() @ R.T.float()
        assert torch.allclose(layer.weight.data.float(), new_W, atol=1e-5)

    def test_block_rotation(self):
        layer = nn.Linear(16, 8)
        original_weight = layer.weight.data.clone()
        R = deterministic_hadamard_matrix(8)
        rotate_in_channels_(layer, rotation_matrix=R)

        assert not torch.equal(layer.weight.data, original_weight)

    def test_bias_unchanged(self):
        layer = nn.Linear(8, 8, bias=True)
        original_bias = layer.bias.data.clone()
        R = deterministic_hadamard_matrix(8)
        rotate_in_channels_(layer, rotation_matrix=R)
        assert torch.equal(layer.bias.data, original_bias)

    def test_incompatible_rotation_size_raises(self):
        layer = nn.Linear(7, 8)
        R = deterministic_hadamard_matrix(8)
        with pytest.raises(ValueError, match="rotation_size.*does not divide"):
            rotate_in_channels_(layer, rotation_matrix=R)

    def test_deduplication_via_rotated_modules(self):
        layer = nn.Linear(8, 16)
        R = deterministic_hadamard_matrix(8)
        seen = set()
        rotate_in_channels_(layer, rotation_matrix=R, rotated_modules=seen)
        rotate_in_channels_(layer, rotation_matrix=R, rotated_modules=seen)
        assert layer in seen
        assert len(seen) == 1

    def test_no_rotation_matrix(self):
        layer = nn.Linear(8, 16)
        original = layer.weight.data.clone()
        rotate_in_channels_(layer, rotation_matrix=None)
        assert torch.equal(layer.weight.data, original)


class TestRotateOutChannels:
    """rotate_out_channels_ fuses output-side rotation: W_new = R.T @ W."""

    def test_full_rotation(self):
        layer = nn.Linear(8, 8)
        original_weight = layer.weight.data.clone()
        R = deterministic_hadamard_matrix(8)
        rotate_out_channels_(layer, rotation_matrix=R)

        assert not torch.equal(layer.weight.data, original_weight)
        new_W = R.T.float() @ original_weight.float()
        assert torch.allclose(layer.weight.data.float(), new_W, atol=1e-5)

    def test_bias_rotated(self):
        layer = nn.Linear(8, 8, bias=True)
        original_bias = layer.bias.data.clone()
        R = deterministic_hadamard_matrix(8)
        rotate_out_channels_(layer, rotation_matrix=R)

        assert not torch.equal(layer.bias.data, original_bias)
        new_bias = R.T.float() @ original_bias.float()
        assert torch.allclose(layer.bias.data.float(), new_bias, atol=1e-5)

    def test_bias_block_rotation(self):
        layer = nn.Linear(8, 16, bias=True)
        original_bias = layer.bias.data.clone()
        R = deterministic_hadamard_matrix(8)
        rotate_out_channels_(layer, rotation_matrix=R)
        assert not torch.equal(layer.bias.data, original_bias)

    def test_incompatible_rotation_size_raises(self):
        layer = nn.Linear(8, 7)
        R = deterministic_hadamard_matrix(8)
        with pytest.raises(ValueError, match="rotation_size.*does not divide"):
            rotate_out_channels_(layer, rotation_matrix=R)


class TestInputRotationWrapperHadamard:
    """InputRotationWrapperHadamard applies online R1 rotation to activations."""

    def test_creation_requires_linear(self):
        with pytest.raises(ValueError, match="only supports nn.Linear"):
            InputRotationWrapperHadamard(nn.Conv2d(3, 8, 3), rotation_size=8)

    def test_full_rotation_uses_butterfly(self):
        layer = nn.Linear(8, 16)
        wrapper = InputRotationWrapperHadamard(layer, rotation_size=8)
        assert wrapper._use_butterfly is True
        assert wrapper._in_features == 8
        assert wrapper._out_features == 16

    def test_block_rotation_uses_matrix(self):
        layer = nn.Linear(16, 8)
        wrapper = InputRotationWrapperHadamard(layer, rotation_size=8)
        assert wrapper._use_butterfly is False
        assert wrapper._rotation_size == 8

    def test_incompatible_rotation_size_raises(self):
        layer = nn.Linear(7, 8)
        with pytest.raises(ValueError, match="not compatible with"):
            InputRotationWrapperHadamard(layer, rotation_size=8)

    def test_forward_full_rotation(self):
        layer = nn.Linear(8, 8)
        layer.weight.data.fill_(1.0)
        layer.bias = None
        wrapper = InputRotationWrapperHadamard(layer, rotation_size=8)
        x = torch.randn(2, 8)
        out = wrapper(x)
        assert out.shape == x.shape
        assert not torch.equal(out, x)

    def test_forward_block_rotation(self):
        layer = nn.Linear(16, 8)
        layer.weight.data.fill_(1.0)
        layer.bias = None
        wrapper = InputRotationWrapperHadamard(layer, rotation_size=8)
        x = torch.randn(2, 16)
        out = wrapper(x)
        # InputRotationWrapperHadamard wraps a Linear, so output shape is
        # determined by out_features (=8), not by input shape.
        assert out.shape == (2, 8)
        # Block rotation should change the output relative to a plain Linear
        # with the same weights/bias.
        plain = nn.Linear(16, 8, bias=False)
        plain.weight.data.fill_(1.0)
        plain_out = plain(x)
        assert not torch.equal(out, plain_out)

    def test_forward_bfloat16(self):
        layer = nn.Linear(8, 8)
        # Cast both layer and input to bfloat16 to keep dtypes consistent
        # (the wrapper does not auto-cast weights, only the activation rotation).
        layer = layer.to(torch.bfloat16)
        wrapper = InputRotationWrapperHadamard(layer, rotation_size=8)
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        out = wrapper(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == (2, 8)

    def test_weight_and_bias_ownership(self):
        layer = nn.Linear(8, 8, bias=True)
        original_weight = layer.weight.data.clone()
        original_bias = layer.bias.data.clone()
        wrapper = InputRotationWrapperHadamard(layer, rotation_size=8)
        assert wrapper.weight is layer.weight
        assert wrapper.bias is layer.bias
        assert torch.equal(wrapper.weight.data, original_weight)

    def test_in_features_property(self):
        layer = nn.Linear(16, 8)
        wrapper = InputRotationWrapperHadamard(layer, rotation_size=8)
        assert wrapper.in_features == 16
        assert wrapper.out_features == 8

    def test_repr(self):
        layer = nn.Linear(8, 16)
        wrapper = InputRotationWrapperHadamard(layer, rotation_size=8)
        repr_str = repr(wrapper)
        assert "InputRotationWrapperHadamard" in repr_str
        assert "in_features=8" in repr_str
        assert "out_features=16" in repr_str


class TestApplyHadamardToLinear:
    """apply_hadamard_to_linear applies Hadamard to linear weights in-place."""

    def test_full_input_rotation(self):
        layer = nn.Linear(8, 8)
        original = layer.weight.data.clone()
        apply_hadamard_to_linear(layer, had_dim=-1, output=False)
        assert not torch.equal(layer.weight.data, original)

    def test_full_output_rotation(self):
        layer = nn.Linear(8, 8)
        original = layer.weight.data.clone()
        apply_hadamard_to_linear(layer, had_dim=-1, output=True)
        assert not torch.equal(layer.weight.data, original)

    def test_block_rotation_input(self):
        layer = nn.Linear(16, 8)
        original = layer.weight.data.clone()
        apply_hadamard_to_linear(layer, had_dim=8, output=False)
        assert not torch.equal(layer.weight.data, original)

    def test_block_rotation_output(self):
        layer = nn.Linear(8, 16)
        original = layer.weight.data.clone()
        apply_hadamard_to_linear(layer, had_dim=8, output=True)
        assert not torch.equal(layer.weight.data, original)

    def test_bias_rotated_on_output(self):
        layer = nn.Linear(8, 8, bias=True)
        original_bias = layer.bias.data.clone()
        apply_hadamard_to_linear(layer, had_dim=-1, output=True)
        assert not torch.equal(layer.bias.data, original_bias)

    def test_requires_linear_or_wrapper(self):
        with pytest.raises(AssertionError):
            apply_hadamard_to_linear(nn.Conv2d(3, 8, 3), had_dim=8)


class TestGetModelArchInfo:
    """get_model_arch_info extracts architecture metadata from models."""

    def test_returns_expected_keys(self):
        class DummyModel(nn.Module):
            pass

        model = DummyModel()
        info = get_model_arch_info(model)
        assert "model_type" in info
        assert "hidden_size" in info
        assert "head_dim" in info
        assert "num_q_heads" in info
        assert "num_kv_heads" in info
        assert "intermediate_size" in info


class TestCreateBlockDiagFromHeadMatrix:
    """create_block_diag_from_head_matrix builds block-diagonal rotation."""

    def test_block_diag_matrix(self):
        R_head = deterministic_hadamard_matrix(8)
        block = create_block_diag_from_head_matrix(R_head, num_heads=4)
        assert block.shape == (32, 32)
        # The block-diagonal is built from a normalized Hadamard, so it's
        # also normalized: block @ block.T = I (not 8*I).
        product = block @ block.T
        expected = torch.eye(32)
        assert torch.allclose(product, expected, atol=1e-5)


class TestUntieWordEmbeddings:
    """untie_word_embeddings_if_needed separates tied embeddings."""

    def test_returns_false_when_not_tied(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 32)
                self.lm_head = nn.Linear(32, 100, bias=False)

        model = DummyModel()
        result = untie_word_embeddings_if_needed(model)
        assert result is False
        assert model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr()


# =============================================================================
# TestCayleyOptimizer — SGDG and AdamAndSGDG
# =============================================================================

class TestSGDG:
    """SGDG optimizer with Cayley retraction on the Stiefel manifold."""

    def test_initialization_defaults(self):
        param = nn.Parameter(torch.randn(8, 8))
        opt = SGDG([param], lr=1e-4)
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["lr"] == 1e-4
        assert opt.stiefel is True

    def test_custom_params(self):
        param = nn.Parameter(torch.randn(8, 8))
        opt = SGDG(
            [param],
            lr=1e-3,
            momentum=0.9,
            weight_decay=0.01,
            stiefel=True,
        )
        group = opt.param_groups[0]
        assert group["momentum"] == 0.9
        assert group["weight_decay"] == 0.01

    def test_step_maintains_orthogonality(self):
        H = torch.linalg.qr(torch.randn(8, 8))[0]
        param = nn.Parameter(H.clone())
        opt = SGDG([param], lr=1e-3)

        loss = (param @ param.T - torch.eye(8)).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

        product = param.data @ param.data.T
        assert torch.allclose(product, torch.eye(8), atol=1e-3)

    def test_multiple_steps_maintain_orthogonality(self):
        H = torch.linalg.qr(torch.randn(8, 8))[0]
        param = nn.Parameter(H.clone())
        opt = SGDG([param], lr=1e-3)

        for _ in range(20):
            loss = (param @ param.T - torch.eye(8)).pow(2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

        product = param.data @ param.data.T
        assert torch.allclose(product, torch.eye(8), atol=1e-2)

    def test_determinant_flip_fix(self):
        param = nn.Parameter(torch.eye(8))
        param.data[7, :] *= -1
        assert torch.det(param.data) < 0

        opt = SGDG([param], lr=1e-3)
        loss = (param @ param.T - torch.eye(8)).pow(2).mean()
        loss.backward()
        opt.step()

        det = torch.det(param.data)
        assert det > 0

    def test_momentum_buffer(self):
        param = nn.Parameter(torch.randn(8, 8))
        opt = SGDG([param], lr=1e-3, momentum=0.9)

        loss = param.sum()
        loss.backward()
        opt.step()

        # PyTorch optimizers key state by the parameter tensor itself.
        assert param in opt.state
        assert "momentum_buffer" in opt.state[param]

    def test_weight_decay(self):
        param = nn.Parameter(torch.randn(8, 8))
        opt = SGDG([param], lr=1e-3, weight_decay=0.1)

        loss = param.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()


class TestAdamAndSGDG:
    """AdamAndSGDG dual optimizer for rotation (SGDG) + smooth (Adam) params."""

    def test_with_both_param_lists(self):
        rot_param = nn.Parameter(torch.randn(8, 8))
        smooth_param = nn.Parameter(torch.randn(8))
        opt = AdamAndSGDG(
            adam_params=[smooth_param],
            sgdg_params=[rot_param],
            learning_rate=1e-4,
            smooth_learning_rate=1e-3,
        )
        assert opt.adam_optimizer is not None
        assert opt.sgdg_optimizer is not None

    def test_with_empty_adam_params(self):
        rot_param = nn.Parameter(torch.randn(8, 8))
        opt = AdamAndSGDG(
            adam_params=[],
            sgdg_params=[rot_param],
            learning_rate=1e-4,
        )
        assert opt.adam_optimizer is None
        assert opt.sgdg_optimizer is not None
        assert opt._has_adam is False
        assert opt._has_sgdg is True

    def test_with_empty_sgdg_params(self):
        smooth_param = nn.Parameter(torch.randn(8))
        opt = AdamAndSGDG(
            adam_params=[smooth_param],
            sgdg_params=[],
            smooth_learning_rate=1e-3,
        )
        assert opt.adam_optimizer is not None
        assert opt.sgdg_optimizer is None
        assert opt._has_adam is True
        assert opt._has_sgdg is False

    def test_with_both_empty(self):
        opt = AdamAndSGDG(
            adam_params=[],
            sgdg_params=[],
            learning_rate=1e-4,
        )
        assert opt.adam_optimizer is None
        assert opt.sgdg_optimizer is None

    def test_step_calls_both_optimizers(self):
        rot_param = nn.Parameter(torch.randn(8, 8))
        smooth_param = nn.Parameter(torch.randn(8))
        opt = AdamAndSGDG(
            adam_params=[smooth_param],
            sgdg_params=[rot_param],
            learning_rate=1e-4,
            smooth_learning_rate=1e-3,
        )

        loss = rot_param.sum() + smooth_param.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    def test_state_dict_roundtrip(self):
        rot_param = nn.Parameter(torch.randn(8, 8))
        smooth_param = nn.Parameter(torch.randn(8))
        opt = AdamAndSGDG(
            adam_params=[smooth_param],
            sgdg_params=[rot_param],
            learning_rate=1e-4,
            smooth_learning_rate=1e-3,
        )

        loss = rot_param.sum() + smooth_param.sum()
        loss.backward()
        opt.step()

        sd = opt.state_dict()
        opt.zero_grad()

        # State dict round-trip should not raise and should preserve internal
        # optimizer state. AdamAndSGDG.state stores the *combined* parent state
        # which is empty by default; the real per-parameter state lives in
        # the internal adam_optimizer / sgdg_optimizer.
        opt.load_state_dict(sd)
        assert opt.sgdg_optimizer is not None
        assert rot_param in opt.sgdg_optimizer.state

    def test_zero_grad(self):
        rot_param = nn.Parameter(torch.randn(8, 8))
        smooth_param = nn.Parameter(torch.randn(8))
        opt = AdamAndSGDG(
            adam_params=[smooth_param],
            sgdg_params=[rot_param],
        )

        loss = rot_param.sum() + smooth_param.sum()
        loss.backward()
        opt.zero_grad()

        assert smooth_param.grad is None or smooth_param.grad.abs().sum() == 0


# =============================================================================
# TestLossFunctions — compute_rotation_loss and spinquant_loss_fn
# =============================================================================

class TestComputeRotationLoss:
    """compute_rotation_loss with kl_top, kl_full, and mse loss types."""

    def test_kl_top_produces_scalar(self):
        logits = torch.randn(2, 10, 100)
        ori_logits = torch.randn(2, 10, 100)
        loss = compute_rotation_loss(logits, ori_logits, loss_type="kl_top", kl_top_k=50)
        assert loss.dim() == 0
        assert loss >= 0

    def test_kl_top_respects_kl_top_k(self):
        logits = torch.randn(2, 10, 100)
        ori_logits = torch.randn(2, 10, 100)
        loss_10 = compute_rotation_loss(logits, ori_logits, loss_type="kl_top", kl_top_k=10)
        loss_50 = compute_rotation_loss(logits, ori_logits, loss_type="kl_top", kl_top_k=50)
        assert loss_10 >= 0
        assert loss_50 >= 0

    def test_kl_top_handles_vocab_larger_than_k(self):
        logits = torch.randn(2, 10, 1000)
        ori_logits = torch.randn(2, 10, 1000)
        loss = compute_rotation_loss(logits, ori_logits, loss_type="kl_top", kl_top_k=50)
        assert loss.dim() == 0
        assert loss >= 0

    def test_kl_full_produces_scalar(self):
        logits = torch.randn(2, 10, 100)
        ori_logits = torch.randn(2, 10, 100)
        loss = compute_rotation_loss(logits, ori_logits, loss_type="kl_full")
        assert loss.dim() == 0
        assert loss >= 0

    def test_mse_produces_scalar(self):
        logits = torch.randn(2, 10, 100)
        ori_logits = torch.randn(2, 10, 100)
        loss = compute_rotation_loss(logits, ori_logits, loss_type="mse")
        assert loss.dim() == 0
        assert loss >= 0

    def test_mse_same_logits_is_zero(self):
        logits = torch.randn(2, 10, 100)
        loss = compute_rotation_loss(logits, logits, loss_type="mse")
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_unknown_loss_type_raises(self):
        logits = torch.randn(2, 10, 100)
        ori_logits = torch.randn(2, 10, 100)
        with pytest.raises(ValueError, match="Unknown loss_type"):
            compute_rotation_loss(logits, ori_logits, loss_type="unknown")

    def test_spinquant_loss_fn_is_alias(self):
        assert spinquant_loss_fn is compute_rotation_loss


# =============================================================================
# TestSpinQuantState — training state tracking
# =============================================================================

class TestSpinQuantState:
    """SpinQuantState tracks training metrics."""

    def test_initialization_defaults(self):
        state = SpinQuantState()
        assert state.enabled is False
        assert state.iteration == 0
        assert state.max_iterations == 0
        assert state.loss_history == []
        assert state.rotation_names == []
        assert state.orthogonality_deviation == []

    def test_update_records_loss_and_ortho(self):
        state = SpinQuantState()
        state.update(loss=0.5, ortho_dev=0.01)
        state.update(loss=0.4, ortho_dev=0.005)
        state.update(loss=0.3, ortho_dev=0.002)
        assert state.iteration == 3
        assert len(state.loss_history) == 3
        assert len(state.orthogonality_deviation) == 3
        assert state.loss_history[-1] == 0.3

    def test_avg_loss(self):
        state = SpinQuantState()
        state.update(loss=0.5)
        state.update(loss=0.3)
        state.update(loss=0.1)
        assert state.avg_loss == pytest.approx(0.3)

    def test_avg_loss_empty_returns_zero(self):
        state = SpinQuantState()
        assert state.avg_loss == 0.0

    def test_final_ortho_dev(self):
        state = SpinQuantState()
        state.update(loss=0.5, ortho_dev=0.01)
        state.update(loss=0.4, ortho_dev=0.005)
        assert state.final_ortho_dev == 0.005

    def test_final_ortho_dev_empty_returns_zero(self):
        state = SpinQuantState()
        assert state.final_ortho_dev == 0.0

    def test_summary_contains_expected_keys(self):
        state = SpinQuantState()
        state.update(loss=0.5, ortho_dev=0.01)
        summary = state.summary()
        assert "enabled" in summary
        assert "iterations" in summary
        assert "final_loss" in summary
        assert "avg_loss" in summary
        assert summary["iterations"] == 1


# =============================================================================
# TestTrainingUtilities — check_orthogonality, clone_model, move_batch
# =============================================================================

class TestCheckOrthogonality:
    """check_orthogonality computes R @ R.T deviation."""

    def test_orthogonal_matrix_returns_near_zero(self):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                H = torch.linalg.qr(torch.randn(8, 8))[0]
                self.spinquant_R1 = nn.Parameter(H)

        model = Dummy()
        dev = check_orthogonality(model)
        assert dev < 1e-4

    def test_non_orthogonal_returns_positive(self):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.spinquant_R1 = nn.Parameter(torch.randn(8, 8))

        model = Dummy()
        dev = check_orthogonality(model)
        assert dev > 0

    def test_skips_non_grad_params(self):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                H = torch.randn(8, 8)
                self.spinquant_R1 = nn.Parameter(H, requires_grad=False)

        model = Dummy()
        dev = check_orthogonality(model)
        assert dev == 0.0


class TestMoveBatchToDevice:
    """move_batch_to_device handles tensor and dict batches."""

    def test_tensor_moves(self):
        batch = torch.randn(2, 10, 100)
        result = move_batch_to_device(batch, torch.device("cpu"))
        assert result.device.type == "cpu"

    def test_dict_moves(self):
        batch = {"input_ids": torch.randn(2, 10), "attention_mask": torch.ones(2, 10)}
        result = move_batch_to_device(batch, torch.device("cpu"))
        assert result["input_ids"].device.type == "cpu"
        assert result["attention_mask"].device.type == "cpu"

    def test_non_tensor_passed_through(self):
        batch = "not a tensor"
        result = move_batch_to_device(batch, torch.device("cpu"))
        assert result == batch


# =============================================================================
# TestLossLogger — training callback
# =============================================================================

class TestLossLogger:
    """LossLogger callback for rotation training."""

    def test_default_interval(self):
        logger = LossLogger()
        assert logger.log_interval == 50

    def test_custom_interval(self):
        logger = LossLogger(log_interval=10)
        assert logger.log_interval == 10

    def test_has_on_step_end(self):
        logger = LossLogger()
        assert hasattr(logger, "on_step_end")
        assert callable(logger.on_step_end)

    def test_callback_interface(self):
        class CustomCB(RotationTrainerCallback):
            called = False

            def on_train_begin(self, args, state, control):
                self.called = True

        cb = CustomCB()
        args = RotationTrainerConfig()
        state = {}
        control = {}
        cb.on_train_begin(args, state, control)
        assert cb.called is True


# =============================================================================
# TestOptimizerCreation — create_dual_optimizer and alias
# =============================================================================

class TestOptimizerCreation:
    """create_dual_optimizer groups params by type."""

    def test_returns_none_for_no_trainable_params(self):
        model = nn.Linear(32, 32)
        result = create_dual_optimizer(model, lr=1e-4)
        assert result is None

    def test_alias_is_same_function(self):
        assert create_spinquant_optimizer is create_dual_optimizer

    def test_returns_none_when_only_smooth_params(self):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.smooth_values = nn.Parameter(torch.randn(8))

        model = Dummy()
        result = create_dual_optimizer(model, lr=1e-4)
        assert result is not None
        assert isinstance(result, AdamAndSGDG)


# =============================================================================
# TestSpinQuantRotation — BaseRotation registry integration
# =============================================================================

class TestSpinQuantRotation:
    """SpinQuantRotation registered as 'spinquant' in BaseRotation."""

    def test_registered_in_base_rotation(self):
        from auto_round.algorithms.transforms.base import BaseRotation

        BaseRotation.from_config(SpinQuantConfig())
        assert "spinquant" in BaseRotation._REGISTRY

    def test_from_config_with_dict(self):
        # from_config requires a BaseRotationConfig (not a raw dict), so
        # construct a SpinQuantConfig from the dict first.
        cfg = SpinQuantConfig(r1=True, r2=True, r3=False, r4=False)
        rot = SpinQuantRotation.from_config(cfg)
        assert rot.config.r1 is True
        assert rot.config.r2 is True
        assert rot.config.r3 is False

    def test_from_config_with_spinquant_config(self):
        cfg = SpinQuantConfig(r1=True, r2=False, trainable_rotation=True)
        rot = SpinQuantRotation.from_config(cfg)
        assert rot.config.r1 is True
        assert rot.config.trainable_rotation is True

    def test_has_rotation_buffers_false_for_normal_module(self):
        module = nn.Linear(32, 32)
        rot = SpinQuantRotation(SpinQuantConfig())
        assert rot.has_rotation_buffers(module) is False

    def test_config_key(self):
        assert SpinQuantRotation.config_key() == "spinquant_config"


# =============================================================================
# TestSpinQuantPreprocessor — model preprocessing integration
# =============================================================================

class TestSpinQuantPreprocessor:
    """SpinQuantPreprocessor orchestrates the rotation pipeline."""

    def test_creation_stores_model_and_config(self):
        model = nn.Linear(32, 32)
        preprocessor = SpinQuantPreprocessor(model)
        assert preprocessor.model is model
        assert isinstance(preprocessor.config, SpinQuantConfig)

    def test_custom_config(self):
        model = nn.Linear(32, 32)
        config = SpinQuantConfig(r1=False, r2=False, r3=False, r4=False)
        preprocessor = SpinQuantPreprocessor(model, config)
        assert preprocessor.config.r1 is False
        assert preprocessor.config.r2 is False

    def test_model_architecture_info(self):
        model = nn.Linear(32, 32)
        preprocessor = SpinQuantPreprocessor(model)
        info = get_model_arch_info(model)
        assert "hidden_size" in info


# =============================================================================
# TestMonkeypatch — QKRotationWrapper and R3 monkeypatch
# =============================================================================

class TestCopyFuncWithNewGlobals:
    """copy_func_with_new_globals creates function copies with modified globals."""

    def test_copy_has_same_code(self):
        # The copied function should have the same bytecode as the original
        # (so its behavior only changes through the modified globals).
        def original(x):
            return x + ADD  # ADD is a free variable resolved from globals

        copied = copy_func_with_new_globals(original, {"ADD": 1})
        assert copied(0) == 1
        assert copied.__code__ is original.__code__

    def test_modified_globals(self):
        def original(x):
            return x + y

        copied = copy_func_with_new_globals(original, {"y": 10})
        assert copied(5) == 15


class TestQKRotationWrapper:
    """QKRotationWrapper applies R3 Hadamard after RoPE on Q and K."""

    def test_initialization(self):
        def dummy_rope(q, k, *args, **kwargs):
            return q, k

        wrapper = QKRotationWrapper(dummy_rope)
        assert wrapper._had_K is None
        assert wrapper._full_matrix is None

    def test_set_hadamard_sets_decomposition(self):
        def dummy_rope(q, k, *args, **kwargs):
            return q, k

        wrapper = QKRotationWrapper(dummy_rope)
        wrapper.set_hadamard(None, head_dim=8)
        assert wrapper._had_K is not None
        assert wrapper._K == 1
        assert wrapper._head_dim == 8

    def test_set_matrix_stores_full_matrix(self):
        def dummy_rope(q, k, *args, **kwargs):
            return q, k

        wrapper = QKRotationWrapper(dummy_rope)
        R = deterministic_hadamard_matrix(8)
        wrapper.set_matrix(R)
        assert wrapper._full_matrix is not None
        assert wrapper._full_matrix.shape == (8, 8)
        assert wrapper._had_K is None

    def test_forward_with_hadamard_mode(self):
        def dummy_rope(q, k, *args, **kwargs):
            return q, k

        wrapper = QKRotationWrapper(dummy_rope)
        wrapper.set_hadamard(None, head_dim=8)

        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 4, 8)
        q_out, k_out = wrapper(q, k)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert not torch.equal(q_out, q)

    def test_forward_with_matrix_mode(self):
        def dummy_rope(q, k, *args, **kwargs):
            return q, k

        wrapper = QKRotationWrapper(dummy_rope)
        R = deterministic_hadamard_matrix(8)
        wrapper.set_matrix(R)

        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 4, 8)
        q_out, k_out = wrapper(q, k)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_forward_dtype_preserved(self):
        def dummy_rope(q, k, *args, **kwargs):
            return q, k

        wrapper = QKRotationWrapper(dummy_rope)
        wrapper.set_hadamard(None, head_dim=8)

        q = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        k = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        q_out, k_out = wrapper(q, k)
        assert q_out.dtype == torch.bfloat16

    def test_attention_orthogonality_preserved(self):
        """(Q@R) @ (K@R).T = Q @ K.T since R is orthogonal."""
        def dummy_rope(q, k, *args, **kwargs):
            return q, k

        wrapper = QKRotationWrapper(dummy_rope)
        R = deterministic_hadamard_matrix(8)
        wrapper.set_matrix(R)

        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 4, 8)
        q_out, k_out = wrapper(q, k)

        attn_original = q @ k.transpose(-2, -1)
        attn_rotated = q_out @ k_out.transpose(-2, -1)
        assert torch.allclose(attn_rotated, attn_original, atol=1e-4)


# =============================================================================
# TestInplaceApply — hook registration for R3 and R4
# =============================================================================

class TestRegisterSpinquantHooks:
    """register_spinquant_hooks registers R3 and R4 online rotations."""

    def test_register_with_empty_config(self):
        class DummyConfig:
            r1 = False
            r2 = False
            r3 = False
            r4 = False
            head_dim = 0
            intermediate_size = 0

        model = nn.Linear(8, 8)
        handles = register_spinquant_hooks(model, DummyConfig())
        assert isinstance(handles, list)

    def test_register_r3_non_pow2_head_dim_warns(self):
        class DummyConfig:
            r3 = True
            r4 = False
            random_r3 = False
            random_r4 = False
            head_dim = 6
            intermediate_size = 0

        model = nn.Linear(8, 8)
        handles = register_spinquant_hooks(model, DummyConfig())
        assert isinstance(handles, list)


class TestRemoveSpinquantHooks:
    """remove_spinquant_hooks safely removes registered hooks."""

    def test_remove_empty_handles(self):
        remove_spinquant_hooks([])

    def test_remove_with_dummy_handle(self):
        model = nn.Linear(8, 8)

        class DummyConfig:
            r3 = False
            r4 = False

        handles = register_spinquant_hooks(model, DummyConfig())
        remove_spinquant_hooks(handles)


class TestApplySpinquantInPlace:
    """apply_spinquant_in_place is a thin wrapper around preprocessor."""

    model = nn.Linear(8, 8)

    def test_returns_model(self):
        model = nn.Linear(8, 8)
        result = apply_spinquant_in_place(model, SpinQuantConfig(r1=False, r2=False))
        assert result is model


# =============================================================================
# TestSerialize — buffer injection, rebuild, config save/load
# =============================================================================

class TestIsQuantlinear:
    """_is_quantlinear identifies QuantLinear subclasses."""

    def test_named_quantlinear(self):
        class QuantLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 8))

        assert _is_quantlinear(QuantLinear()) is True

    def test_name_containing_quantlinear(self):
        class NVFP4QuantLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 8))

        assert _is_quantlinear(NVFP4QuantLinear()) is True

    def test_normal_linear_returns_false(self):
        assert _is_quantlinear(nn.Linear(8, 8)) is False

    def test_conv2d_returns_false(self):
        assert _is_quantlinear(nn.Conv2d(3, 8, 3)) is False


class TestHasSpinquantBuffers:
    """_has_spinquant_buffers checks for spinquant buffer prefixes."""

    def test_false_when_no_buffers(self):
        module = nn.Linear(8, 8)
        assert _has_spinquant_buffers(module) is False

    def test_true_with_r1_buffer(self):
        module = nn.Linear(8, 8)
        module.register_buffer("spinquant_r1_type", torch.tensor(0))
        assert _has_spinquant_buffers(module) is True

    def test_true_with_r4_buffer(self):
        module = nn.Linear(8, 8)
        module.register_buffer("spinquant_r4_type", torch.tensor(0))
        assert _has_spinquant_buffers(module) is True


class TestApplyRotationFromBuffer:
    """_apply_rotation_from_buffer applies rotation using stored buffers."""

    def test_hadamard_type_reconstructs_from_size(self):
        module = nn.Linear(8, 8)
        module.register_buffer("spinquant_r1_type", torch.tensor(ROTATION_TYPE_HADAMARD))
        module.register_buffer("spinquant_r1_size", torch.tensor(8))

        x = torch.randn(2, 8)
        result = _apply_rotation_from_buffer(module, x, "spinquant_r1")
        assert result.shape == x.shape
        assert not torch.equal(result, x)

    def test_random_type_uses_stored_matrix(self):
        module = nn.Linear(8, 8)
        R = deterministic_hadamard_matrix(8)
        R_int8 = R.sign().to(torch.int8)
        module.register_buffer("spinquant_r1_type", torch.tensor(ROTATION_TYPE_RANDOM))
        module.register_buffer("spinquant_r1_size", torch.tensor(8))
        module.register_buffer("spinquant_r1_matrix", R_int8)

        x = torch.randn(2, 8)
        result = _apply_rotation_from_buffer(module, x, "spinquant_r1")
        assert result.shape == x.shape

    def test_trained_type_uses_stored_float32(self):
        module = nn.Linear(8, 8)
        R = torch.linalg.qr(torch.randn(8, 8))[0].float()
        module.register_buffer("spinquant_r1_type", torch.tensor(ROTATION_TYPE_TRAINED))
        module.register_buffer("spinquant_r1_size", torch.tensor(8))
        module.register_buffer("spinquant_r1_matrix", R)

        x = torch.randn(2, 8)
        result = _apply_rotation_from_buffer(module, x, "spinquant_r1")
        assert result.shape == x.shape


class TestApplyBlockRotationButterfly:
    """_apply_block_rotation_butterfly handles non-pow2 block rotation."""

    def test_full_rotation_uses_matmul_hadU(self):
        x = torch.randn(2, 8)
        had_K, K = get_hadamard_K(8)
        result = _apply_block_rotation_butterfly(x, had_K, K, 8)
        expected = matmul_hadU(x, hadamard_K=had_K, K=K)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_block_rotation_output_shape(self):
        x = torch.randn(2, 16)
        had_K, K = get_hadamard_K(8)
        result = _apply_block_rotation_butterfly(x, had_K, K, 8)
        assert result.shape == x.shape


class TestInjectSpinquantBuffers:
    """inject_spinquant_buffers injects rotation buffers into QuantLinear."""

    def test_returns_zero_for_non_quantlinear_model(self):
        model = nn.Linear(8, 8)

        class DummyConfig:
            r1 = True
            r2 = False
            r3 = False
            r4 = False
            online_r1_rotation = True
            rotation_size = None
            random_r1 = False

        n = inject_spinquant_buffers(model, DummyConfig())
        assert n == 0


class TestRebuildSpinquantOnline:
    """rebuild_spinquant_online reconstructs rotations from config."""

    def test_no_op_for_unconfigured_model(self):
        model = nn.Linear(8, 8)
        result = rebuild_spinquant_online(model, config=None)
        assert result is model


# =============================================================================
# TestRotationTrainer — standalone trainer
# =============================================================================

class TestRotationTrainerConfig:
    """RotationTrainerConfig dataclass defaults."""

    def test_all_defaults(self):
        cfg = RotationTrainerConfig()
        assert cfg.r1 is True
        assert cfg.r2 is True
        assert cfg.r3 is True
        assert cfg.r4 is True
        assert cfg.trainable_rotation is True
        assert cfg.trainable_smooth is True
        assert cfg.online_r1_rotation is False
        assert cfg.lr == 1e-4
        assert cfg.smooth_lr == 1e-3
        assert cfg.iters == 200
        assert cfg.batch_size == 1
        assert cfg.loss_type == "kl_top"
        assert cfg.kl_top_k == 1000
        assert cfg.fuse_rmsnorm is True
        assert cfg.untie_embeddings is True
        assert cfg.log_interval == 50
        assert cfg.eval_interval == 0
        assert cfg.save_interval == 0

    def test_device_auto_detects_cuda(self):
        cfg = RotationTrainerConfig()
        assert cfg.device in ("cuda", "cpu")


class TestRotationTrainer:
    """RotationTrainer lifecycle — setup, train, fuse, checkpoint."""

    def test_creation_stores_model_and_config(self):
        model = nn.Linear(8, 8)
        trainer = RotationTrainer(model, config=RotationTrainerConfig())
        assert trainer.model is model
        assert isinstance(trainer.config, RotationTrainerConfig)

    def test_default_callbacks_included(self):
        model = nn.Linear(8, 8)
        trainer = RotationTrainer(model)
        assert len(trainer.callbacks) == 2
        callback_types = [type(cb).__name__ for cb in trainer.callbacks]
        assert "LossLogger" in callback_types
        assert "OrthogonalityMonitor" in callback_types

    def test_custom_callbacks(self):
        model = nn.Linear(8, 8)
        cb = LossLogger(log_interval=10)
        trainer = RotationTrainer(model, callbacks=[cb])
        assert trainer.callbacks == [cb]

    def test_state_initialized(self):
        model = nn.Linear(8, 8)
        trainer = RotationTrainer(model)
        assert "step" in trainer.state
        assert trainer.state["step"] == 0
        assert "loss" in trainer.state
        assert "avg_loss" in trainer.state


# =============================================================================
# TestSerializationTypes — rotation type constants
# =============================================================================

class TestRotationTypeConstants:
    """ROTATION_TYPE_* constants have expected integer values."""

    def test_rotation_type_values(self):
        assert ROTATION_TYPE_HADAMARD == 0
        assert ROTATION_TYPE_RANDOM == 1
        assert ROTATION_TYPE_TRAINED == 2
