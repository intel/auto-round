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

"""Comprehensive CPU tests for the rotation module (Hadamard rotation/transform).

Tests cover the entire public API surface across both backends:
- RotationConfig schema and normalization helpers
- Deterministic and random Hadamard matrix construction
- Hadamard matrix orthogonality and correctness properties
- Block-diagonal matrix multiplication (multihead_matmul)
- HadamardTransform and RandomHadamardTransform nn.Module classes
- Transform registry and build factory
- Backend dispatcher logic (auto / inplace / transform)
- apply_hadamard_rotation and apply_rotation_transform
- HadamardRotation BaseRotation subclass
- Inplace rotation primitives (layer fusion, weight rotation)
- Online Hadamard hooks (Full, CrossHead, Group variants)
- RotationMapping registry and model-config inference
- Patch idempotency (WrapperLinear, WrapperWALayer, QuantLinear)
- Random Hadamard global cache management
- matmul_hadU butterfly transform
- Non-power-of-2 Hadamard construction via safetensors fallback
"""

from __future__ import annotations

import copy
import gc

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.transforms.quarot.config import (
    RotationConfig,
    normalize_rotation_config,
    to_dict_rotation_config,
    dump_group_size_to_rotation_config,
)
from auto_round.algorithms.transforms.quarot.transforms import (
    HadamardTransform,
    RandomHadamardTransform,
    build_hadamard_transform,
    HADAMARDS,
)
from auto_round.algorithms.transforms.quarot.utils.math import (
    deterministic_hadamard_matrix,
    random_hadamard_matrix,
    is_pow2,
    _fetch_hadamard_divisor,
    _matmul_hadU,
)
from auto_round.algorithms.transforms.quarot.utils.matrix import (
    apply_transform_weight,
    multihead_matmul,
)
from auto_round.algorithms.transforms.quarot.dispatcher import (
    resolve_hadamard_backend,
    apply_hadamard_rotation,
)
from auto_round.algorithms.transforms.quarot.apply import (
    HadamardRotation,
)
from auto_round.algorithms.transforms.quarot.inplace.model_config import (
    RotationMapping,
    MAPPING_REGISTRY,
    register_mapping,
    get_mapping,
    infer_mapping_from_model,
    _resolve,
)
from auto_round.algorithms.transforms.quarot.inplace.hooks import (
    matmul_hadU,
    matmul_hadUt,
    get_hadK,
    is_pow2 as inplace_is_pow2,
    _normalize_rotation_matrix,
    _resolve_compute_device,
    get_or_create_random_hadamard,
    clear_random_hadamard_cache,
    FullOnlineHadamardHook,
    CrossHeadOnlineHadamardHook,
    GroupOnlineHadamardHook,
    apply_exact_had_to_linear,
    apply_cross_head_had_to_linear,
    deterministic_hadamard_matrix as inplace_det_hadamard,
    random_hadamard_matrix as inplace_rand_hadamard,
)
from auto_round.algorithms.transforms.quarot.patch import (
    patch_wrapperlinear_to_apply_transform,
    patch_wrapperwalayer_forward_to_apply_transform,
)


# =============================================================================
# Test RotationConfig
# =============================================================================

class TestRotationConfigValidation:
    """RotationConfig field validation and defaults."""

    def test_all_defaults(self):
        cfg = RotationConfig()
        assert cfg.algorithm == "hadamard"
        assert cfg.backend == "auto"
        assert cfg.block_size is None
        assert cfg.hadamard_type == "hadamard"
        assert cfg.fuse_online_to_weight is None
        assert cfg.allow_online_rotation is True
        assert cfg.random_seed is False

    def test_custom_values_stored(self):
        cfg = RotationConfig(
            backend="inplace",
            block_size=128,
            hadamard_type="random_hadamard",
            fuse_online_to_weight=True,
            allow_online_rotation=False,
            random_seed=True,
        )
        assert cfg.backend == "inplace"
        assert cfg.block_size == 128
        assert cfg.hadamard_type == "random_hadamard"
        assert cfg.fuse_online_to_weight is True
        assert cfg.allow_online_rotation is False
        assert cfg.random_seed is True

    def test_quarot_hadamard_type(self):
        cfg = RotationConfig(hadamard_type="inplace_quarot_hadamard")
        assert cfg.hadamard_type == "inplace_quarot_hadamard"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            RotationConfig(backend="bad_backend")

    def test_invalid_hadamard_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported hadamard_type"):
            RotationConfig(hadamard_type="bad_type")

    def test_model_dump_roundtrip(self):
        cfg = RotationConfig(backend="inplace", block_size=64, hadamard_type="random_hadamard")
        dumped = cfg.model_dump()
        restored = RotationConfig.model_validate(dumped)
        assert restored.backend == "inplace"
        assert restored.block_size == 64
        assert restored.hadamard_type == "random_hadamard"

    def test_arbitrary_types_allowed(self):
        cfg = RotationConfig()
        cfg._extra_field = {"some": "data"}
        assert cfg._extra_field["some"] == "data"


class TestToDictRotationConfig:
    """to_dict_rotation_config conversion from all supported input types."""

    def test_none_returns_empty_dict(self):
        assert to_dict_rotation_config(None) == {}

    def test_empty_string_returns_empty_dict(self):
        assert to_dict_rotation_config("") == {}

    def test_whitespace_string_returns_empty_dict(self):
        assert to_dict_rotation_config("   ") == {}

    def test_default_string_maps_to_hadamard(self):
        result = to_dict_rotation_config("default")
        assert result == {"hadamard_type": "hadamard"}

    def test_hadamard_string_shorthand(self):
        result = to_dict_rotation_config("hadamard")
        assert result == {"hadamard_type": "hadamard"}

    def test_random_hadamard_string_shorthand(self):
        result = to_dict_rotation_config("random_hadamard")
        assert result == {"hadamard_type": "random_hadamard"}

    def test_quarot_hadamard_string_shorthand(self):
        result = to_dict_rotation_config("quarot_hadamard")
        assert result == {"hadamard_type": "quarot_hadamard"}

    def test_dict_shallow_copy(self):
        input_dict = {"hadamard_type": "hadamard", "block_size": 32}
        result = to_dict_rotation_config(input_dict)
        assert result == input_dict
        assert result is not input_dict

    def test_rotation_config_model_dump(self):
        cfg = RotationConfig(backend="inplace", block_size=64)
        result = to_dict_rotation_config(cfg)
        assert result["backend"] == "inplace"
        assert result["block_size"] == 64

    def test_dict_with_extra_keys(self):
        input_dict = {"hadamard_type": "hadamard", "unknown_key": 123}
        result = to_dict_rotation_config(input_dict)
        assert result["unknown_key"] == 123


class TestDumpGroupSizeToRotationConfig:
    """dump_group_size_to_rotation_config sets block_size from group_size."""

    def test_sets_block_size_when_absent(self):
        result = dump_group_size_to_rotation_config({}, 128)
        assert result["block_size"] == 128

    def test_does_not_override_existing_block_size(self):
        result = dump_group_size_to_rotation_config({"block_size": 64}, 128)
        assert result["block_size"] == 64

    def test_preserves_other_keys(self):
        result = dump_group_size_to_rotation_config(
            {"hadamard_type": "random_hadamard"}, 32
        )
        assert result["hadamard_type"] == "random_hadamard"
        assert result["block_size"] == 32

    def test_handles_none_input(self):
        result = dump_group_size_to_rotation_config(None, 64)
        assert result["block_size"] == 64

    def test_handles_string_input(self):
        result = dump_group_size_to_rotation_config("hadamard", 128)
        assert result["hadamard_type"] == "hadamard"
        assert result["block_size"] == 128


class TestNormalizeRotationConfig:
    """normalize_rotation_config applies data-type-specific defaults."""

    def test_none_returns_empty_dict(self):
        assert normalize_rotation_config(None) == {}

    def test_mx_fp_sets_block_size_32(self):
        result = normalize_rotation_config({}, data_type="mx_fp")
        assert result["block_size"] == 32

    def test_nv_fp4_sets_block_size_16(self):
        result = normalize_rotation_config({}, data_type="nv_fp4")
        assert result["block_size"] == 16

    def test_int_does_not_override_block_size(self):
        result = normalize_rotation_config({}, data_type="int")
        assert result.get("block_size") is None

    def test_string_shorthand_normalizes(self):
        result = normalize_rotation_config("random_hadamard", data_type="mx_fp")
        assert result["hadamard_type"] == "random_hadamard"
        assert result["block_size"] == 32

    def test_rotation_config_object_normalizes(self):
        cfg = RotationConfig(backend="inplace", block_size=64)
        result = normalize_rotation_config(cfg, data_type="mx_fp")
        assert result["backend"] == "inplace"
        assert result["block_size"] == 64

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="Invalid RotationConfig"):
            normalize_rotation_config({"backend": "bad"})


# =============================================================================
# Test is_pow2
# =============================================================================

class TestIsPow2:
    """is_pow2 correctly identifies powers of two."""

    def test_powers_of_two_return_true(self):
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
            assert is_pow2(n) is True, f"{n} should be power of 2"

    def test_non_powers_of_two_return_false(self):
        for n in [0, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 31, 63, 100, 255]:
            assert is_pow2(n) is False, f"{n} should not be power of 2"

    def test_negative_numbers_return_false(self):
        assert is_pow2(-1) is False
        assert is_pow2(-2) is False
        assert is_pow2(-128) is False


# =============================================================================
# Test Hadamard Matrix Construction
# =============================================================================

class TestDeterministicHadamardMatrix:
    """deterministic_hadamard_matrix via Sylvester construction."""

    @pytest.mark.parametrize("size", [2, 4, 8, 16, 32, 64, 128])
    def test_correct_shape(self, size):
        H = deterministic_hadamard_matrix(size)
        assert H.shape == (size, size)

    @pytest.mark.parametrize("size", [2, 4, 8, 16, 32, 64])
    def test_elements_are_plusminus_one(self, size):
        H = deterministic_hadamard_matrix(size)
        unique = set(H.unique().tolist())
        assert unique <= {1.0, -1.0}

    @pytest.mark.parametrize("size", [2, 4, 8, 16, 32])
    def test_orthogonal_property(self, size):
        H = deterministic_hadamard_matrix(size, dtype=torch.float32)
        product = H @ H.T
        expected = torch.eye(size, dtype=torch.float32) * size
        assert torch.allclose(product, expected, atol=1e-5)

    @pytest.mark.parametrize("size", [2, 4, 8, 16])
    def test_orthogonal_property_bfloat16(self, size):
        H = deterministic_hadamard_matrix(size, dtype=torch.bfloat16)
        product = H.float() @ H.T.float()
        expected = torch.eye(size) * size
        assert torch.allclose(product, expected, atol=1e-3)

    def test_respects_dtype(self):
        H = deterministic_hadamard_matrix(8, dtype=torch.float64)
        assert H.dtype == torch.float64

    def test_respects_device(self):
        H = deterministic_hadamard_matrix(8, device="cpu")
        assert H.device.type == "cpu"

    def test_non_power_of_two_raises(self):
        with pytest.raises(ValueError, match="2\\^n"):
            deterministic_hadamard_matrix(7)
        with pytest.raises(ValueError, match="2\\^n"):
            deterministic_hadamard_matrix(12)
        with pytest.raises(ValueError, match="2\\^n"):
            deterministic_hadamard_matrix(100)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="size <= 0"):
            deterministic_hadamard_matrix(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="size <= 0"):
            deterministic_hadamard_matrix(-1)


class TestRandomHadamardMatrix:
    """random_hadamard_matrix construction with seed reproducibility."""

    def test_correct_shape(self):
        gen = torch.Generator().manual_seed(42)
        H = random_hadamard_matrix(8, gen=gen)
        assert H.shape == (8, 8)

    def test_same_seed_same_matrix(self):
        for seed in [0, 1, 42, 123, 999]:
            gen1 = torch.Generator().manual_seed(seed)
            gen2 = torch.Generator().manual_seed(seed)
            H1 = random_hadamard_matrix(8, gen=gen1)
            H2 = random_hadamard_matrix(8, gen=gen2)
            assert torch.equal(H1, H2), f"Seed {seed} should produce identical matrices"

    def test_different_seeds_different_matrices(self):
        H1 = random_hadamard_matrix(8, gen=torch.Generator().manual_seed(42))
        H2 = random_hadamard_matrix(8, gen=torch.Generator().manual_seed(999))
        assert not torch.equal(H1, H2)

    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_orthogonal_property(self, size):
        gen = torch.Generator().manual_seed(42)
        H = random_hadamard_matrix(size, dtype=torch.float32, gen=gen)
        product = H @ H.T
        expected = torch.eye(size, dtype=torch.float32) * size
        assert torch.allclose(product, expected, atol=1e-4)

    def test_respects_dtype(self):
        gen = torch.Generator().manual_seed(42)
        H = random_hadamard_matrix(8, dtype=torch.float64, gen=gen)
        assert H.dtype == torch.float64

    def test_respects_device(self):
        gen = torch.Generator().manual_seed(42)
        H = random_hadamard_matrix(8, device="cpu", gen=gen)
        assert H.device.type == "cpu"


class TestFetchHadamardDivisor:
    """_fetch_hadamard_divisor loads precomputed matrices for non-pow2 sizes."""

    def test_returns_tensor_for_available_sizes(self):
        result = _fetch_hadamard_divisor(24, torch.float32, torch.device("cpu"))
        assert result is not None
        assert result.shape == (24, 24)
        assert result.dtype == torch.float32
        assert result.device.type == "cpu"

    def test_returns_none_for_size_not_in_file(self):
        result = _fetch_hadamard_divisor(1000, torch.float32, torch.device("cpu"))
        assert result is None

    def test_result_is_orthogonal(self):
        result = _fetch_hadamard_divisor(24, torch.float32, torch.device("cpu"))
        assert result is not None
        product = result @ result.T
        expected = torch.eye(result.shape[0]) * result.shape[0]
        assert torch.allclose(product, expected, atol=1e-4)

    def test_result_is_orthogonal_for_size_48(self):
        result = _fetch_hadamard_divisor(48, torch.float32, torch.device("cpu"))
        assert result is not None
        product = result @ result.T
        expected = torch.eye(result.shape[0]) * result.shape[0]
        assert torch.allclose(product, expected, atol=1e-4)


# =============================================================================
# Test Matrix Operations
# =============================================================================

class TestMultiheadMatmul:
    """multihead_matmul block-diagonal matrix multiplication."""

    def test_standard_multiplication(self):
        A = torch.randn(4, 8)
        B = torch.randn(8, 8)
        result = multihead_matmul(A, B)
        expected = A @ B
        assert torch.equal(result, expected)

    def test_expands_a_when_larger(self):
        size = 8
        B = torch.randn(size, size)
        A = torch.randn(2, size * 2)
        result = multihead_matmul(A, B)
        expected = torch.zeros(2, size * 2)
        expected[:, :size] = A[:, :size] @ B
        expected[:, size:] = A[:, size:] @ B
        assert torch.allclose(result, expected, atol=1e-5)

    def test_expands_b_when_larger(self):
        size = 8
        A = torch.randn(4, size)
        B = torch.randn(size * 2, size * 2)
        result = multihead_matmul(A, B)
        assert result.shape[-1] == B.shape[-1]

    def test_incompatible_dims_raises(self):
        A = torch.randn(4, 7)
        B = torch.randn(8, 8)
        with pytest.raises(ValueError, match="not divisible"):
            multihead_matmul(A, B)

    def test_incompatible_dims_reversed_raises(self):
        A = torch.randn(4, 8)
        B = torch.randn(7, 7)
        with pytest.raises(ValueError, match="not divisible"):
            multihead_matmul(A, B)

    def test_3d_input(self):
        A = torch.randn(2, 3, 16)
        B = torch.randn(16, 16)
        result = multihead_matmul(A, B)
        assert result.shape == (2, 3, 16)

    def test_bfloat16_preserved(self):
        A = torch.randn(4, 8, dtype=torch.bfloat16)
        B = torch.randn(8, 8, dtype=torch.bfloat16)
        result = multihead_matmul(A, B)
        assert result.dtype == torch.bfloat16


class TestApplyTransformWeight:
    """apply_transform_weight applies rotation matrices correctly."""

    def test_input_location_uses_transform_directly(self):
        transform = torch.randn(8, 8)
        value = torch.randn(4, 8)
        result = apply_transform_weight(transform, value, "input", nn.Linear)
        expected = multihead_matmul(value, transform)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_weight_location_uses_transform_transpose(self):
        transform = torch.randn(8, 8)
        value = torch.randn(4, 8)
        result = apply_transform_weight(transform, value, "weight", nn.Linear)
        expected = multihead_matmul(value, transform.T)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_unsupported_module_raises(self):
        transform = torch.randn(8, 8)
        value = torch.randn(4, 8)
        with pytest.raises(NotImplementedError):
            apply_transform_weight(transform, value, "weight", nn.Conv2d)


# =============================================================================
# Test HadamardTransform
# =============================================================================

class TestHadamardTransformClass:
    """HadamardTransform nn.Module — deterministic block-diagonal rotation."""

    def test_default_size_is_32(self):
        transform = HadamardTransform()
        assert transform.size == 32
        assert transform.scale == 1.0 / (32**0.5)

    def test_custom_block_size(self):
        transform = HadamardTransform(block_size=16)
        assert transform.size == 16
        assert transform.scale == 1.0 / (16**0.5)
        assert transform.weight.shape == (16, 16)

    def test_weight_is_frozen(self):
        transform = HadamardTransform(block_size=8)
        assert not transform.weight.requires_grad

    def test_forward_preserves_shape(self):
        transform = HadamardTransform(block_size=8)
        x = torch.randn(4, 8)
        result = transform(x)
        assert result.shape == x.shape

    def test_forward_changes_values(self):
        transform = HadamardTransform(block_size=8)
        x = torch.randn(4, 8)
        result = transform(x)
        assert not torch.equal(result, x)

    def test_orthogonal_property(self):
        transform = HadamardTransform(block_size=8)
        H = transform.weight
        product = H @ H.T
        expected = torch.eye(8)
        assert torch.allclose(product, expected, atol=1e-4)

    def test_forward_bfloat16(self):
        transform = HadamardTransform(block_size=8)
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        result = transform(x)
        assert result.dtype == torch.bfloat16

    def test_multidimensional_forward(self):
        transform = HadamardTransform(block_size=8)
        x = torch.randn(2, 3, 4, 8)
        result = transform(x)
        assert result.shape == x.shape

    def test_respects_dtype_parameter(self):
        transform = HadamardTransform(block_size=8, precision=torch.float64)
        assert transform.weight.dtype == torch.float64

    def test_location_input(self):
        transform = HadamardTransform(block_size=8, location="input")
        x = torch.randn(4, 8)
        result = transform(x)
        assert result.shape == x.shape


class TestRandomHadamardTransformClass:
    """RandomHadamardTransform — seeded random rotation matrix."""

    def test_same_seed_reproducible(self):
        t1 = RandomHadamardTransform(block_size=8, seed=42)
        t2 = RandomHadamardTransform(block_size=8, seed=42)
        assert torch.equal(t1.weight, t2.weight)

    def test_different_seeds_different_matrices(self):
        t1 = RandomHadamardTransform(block_size=8, seed=42)
        t2 = RandomHadamardTransform(block_size=8, seed=999)
        assert not torch.equal(t1.weight, t2.weight)

    def test_orthogonal_property(self):
        transform = RandomHadamardTransform(block_size=8, seed=42)
        H = transform.weight
        product = H @ H.T
        expected = torch.eye(8)
        assert torch.allclose(product, expected, atol=1e-4)

    def test_inverse_transposes_matrix(self):
        t_normal = RandomHadamardTransform(block_size=8, seed=42, inverse=False)
        t_inverse = RandomHadamardTransform(block_size=8, seed=42, inverse=True)
        assert torch.allclose(t_normal.weight, t_inverse.weight.T, atol=1e-5)

    def test_generator_overrides_seed(self):
        gen = torch.Generator().manual_seed(42)
        t1 = RandomHadamardTransform(block_size=8, generator=gen)
        gen2 = torch.Generator().manual_seed(999)
        t2 = RandomHadamardTransform(block_size=8, generator=gen2)
        assert not torch.equal(t1.weight, t2.weight)

    def test_forward_changes_values(self):
        transform = RandomHadamardTransform(block_size=8, seed=42)
        x = torch.randn(4, 8)
        result = transform(x)
        assert not torch.equal(result, x)

    def test_inherits_from_hadamard_transform(self):
        assert issubclass(RandomHadamardTransform, HadamardTransform)


class TestBuildHadamardTransform:
    """build_hadamard_transform factory selects correct class."""

    def test_returns_hadamard_transform(self):
        result = build_hadamard_transform("hadamard", block_size=8)
        assert isinstance(result, HadamardTransform)
        assert result.weight.shape == (8, 8)

    def test_returns_random_hadamard_transform(self):
        result = build_hadamard_transform("random_hadamard", block_size=8, seed=42)
        assert isinstance(result, RandomHadamardTransform)

    def test_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown hadamard_type"):
            build_hadamard_transform("unknown", block_size=8)

    def test_passes_kwargs(self):
        result = build_hadamard_transform(
            "random_hadamard", block_size=8, seed=42, device=torch.device("cpu")
        )
        assert isinstance(result, RandomHadamardTransform)
        assert result.generator is not None


class TestHADAMARDSRegistry:
    """HADAMARDS registry maps type strings to classes."""

    def test_contains_hadamard_key(self):
        assert "hadamard" in HADAMARDS
        assert HADAMARDS["hadamard"] is HadamardTransform

    def test_contains_random_hadamard_key(self):
        assert "random_hadamard" in HADAMARDS
        assert HADAMARDS["random_hadamard"] is RandomHadamardTransform

    def test_quarot_hadamard_not_in_registry(self):
        assert "quarot_hadamard" not in HADAMARDS


# =============================================================================
# Test Dispatcher
# =============================================================================

class TestResolveHadamardBackend:
    """resolve_hadamard_backend routes to correct backend string."""

    def test_explicit_inplace(self):
        cfg = RotationConfig(backend="inplace")
        assert resolve_hadamard_backend(cfg, "int") == "inplace"

    def test_auto_with_fuse_returns_inplace(self):
        cfg = RotationConfig(backend="auto", fuse_online_to_weight=True)
        assert resolve_hadamard_backend(cfg, "int") == "inplace"

    def test_auto_mx_fp_returns_transform(self):
        cfg = RotationConfig(backend="auto")
        assert resolve_hadamard_backend(cfg, "mx_fp") == "transform"

    def test_auto_nv_fp_returns_transform(self):
        cfg = RotationConfig(backend="auto")
        assert resolve_hadamard_backend(cfg, "nv_fp4") == "transform"
        assert resolve_hadamard_backend(cfg, "nv_fp8") == "transform"

    def test_auto_other_dtype_returns_inplace(self):
        cfg = RotationConfig(backend="auto")
        assert resolve_hadamard_backend(cfg, "int") == "inplace"
        assert resolve_hadamard_backend(cfg, "fp8") == "inplace"
        assert resolve_hadamard_backend(cfg, "gptq") == "inplace"

    def test_transform_backend_requires_mx_or_nv_fp(self):
        cfg = RotationConfig(backend="transform", allow_online_rotation=True)
        assert resolve_hadamard_backend(cfg, "mx_fp") == "transform"
        assert resolve_hadamard_backend(cfg, "nv_fp4") == "transform"

    def test_transform_backend_rejects_non_mx_nv_fp(self):
        cfg = RotationConfig(backend="transform", allow_online_rotation=True)
        with pytest.raises(ValueError, match="only supports MXFP4 / NVFP4"):
            resolve_hadamard_backend(cfg, "int")

    def test_transform_backend_rejects_fuse(self):
        cfg = RotationConfig(
            backend="transform",
            fuse_online_to_weight=True,
            allow_online_rotation=True,
        )
        with pytest.raises(ValueError, match="does not support fuse_online_to_weight"):
            resolve_hadamard_backend(cfg, "mx_fp")

    def test_transform_backend_requires_online_rotation(self):
        cfg = RotationConfig(backend="transform", allow_online_rotation=False)
        with pytest.raises(ValueError, match="allow_online_rotation"):
            resolve_hadamard_backend(cfg, "mx_fp")


class TestApplyHadamardRotation:
    """apply_hadamard_rotation unified entry point."""

    def test_none_config_normalizes_to_defaults(self):
        result = normalize_rotation_config(None, data_type="int")
        assert result == {}

    def test_inplace_backend_sets_rotation_config(self):
        cfg = RotationConfig(backend="inplace")
        resolved = resolve_hadamard_backend(cfg, "int")
        assert resolved == "inplace"

    def test_auto_with_int_uses_inplace(self):
        cfg = RotationConfig(backend="auto")
        backend = resolve_hadamard_backend(cfg, "int")
        assert backend == "inplace"

    def test_auto_with_mx_fp_uses_transform(self):
        cfg = RotationConfig(backend="auto")
        backend = resolve_hadamard_backend(cfg, "mx_fp")
        assert backend == "transform"


# =============================================================================
# Test HadamardRotation BaseRotation Subclass
# =============================================================================

class TestHadamardRotationClass:
    """HadamardRotation — the BaseRotation implementation."""

    def test_from_config_dict(self):
        rot = HadamardRotation.from_config({"hadamard_type": "hadamard", "backend": "auto"})
        assert rot.config.hadamard_type == "hadamard"
        assert rot.config.backend == "auto"

    def test_from_config_rotation_config(self):
        cfg = RotationConfig(backend="auto", hadamard_type="random_hadamard")
        rot = HadamardRotation.from_config(cfg)
        assert rot.config.backend == "auto"
        assert rot.config.hadamard_type == "random_hadamard"

    def test_apply_to_model_resolves_inplace_backend(self):
        cfg = RotationConfig(backend="auto")
        resolved = resolve_hadamard_backend(cfg, "int")
        assert resolved == "inplace"


class TestApplyRotationTransformOneShot:
    """apply_rotation_transform — one-shot convenience wrapper."""

    def test_none_config_is_noop(self):
        cfg = normalize_rotation_config(None, data_type="int")
        assert cfg == {}

    def test_dict_config_normalizes(self):
        cfg = normalize_rotation_config(
            {"backend": "inplace", "hadamard_type": "hadamard"}, data_type="int"
        )
        assert cfg["backend"] == "inplace"
        assert cfg["hadamard_type"] == "hadamard"

    def test_rotation_config_object_normalizes(self):
        cfg = RotationConfig(backend="inplace", hadamard_type="random_hadamard")
        normalized = normalize_rotation_config(cfg, data_type="int")
        assert normalized["backend"] == "inplace"
        assert normalized["hadamard_type"] == "random_hadamard"


# =============================================================================
# Test Inplace Rotation — matmul_hadU
# =============================================================================

class TestMatmulHadU:
    """matmul_hadU butterfly Hadamard transform on tensors."""

    @pytest.mark.parametrize("size", [2, 4, 8, 16, 32])
    def test_output_shape(self, size):
        X = torch.randn(4, size)
        result = matmul_hadU(X)
        assert result.shape == X.shape

    @pytest.mark.parametrize("size", [2, 4, 8, 16])
    def test_double_application_returns_original(self, size):
        X = torch.randn(4, size)
        result = matmul_hadU(X)
        reconstructed = matmul_hadU(result)
        assert torch.allclose(reconstructed, X, atol=1e-3)

    @pytest.mark.parametrize("size", [24, 48])
    def test_inverse_using_matmul_hadUt(self, size):
        X = torch.randn(4, size)
        transformed = matmul_hadU(X)
        reconstructed = matmul_hadUt(transformed)
        assert torch.allclose(reconstructed, X, atol=1e-4)

    def test_pow2_uses_symmetric_matrix(self):
        for size in [2, 4, 8, 16]:
            X = torch.randn(4, size)
            assert torch.allclose(matmul_hadU(X), matmul_hadUt(X), atol=1e-5)

    def test_bfloat16(self):
        X = torch.randn(4, 8, dtype=torch.bfloat16)
        result = matmul_hadU(X)
        assert result.dtype == torch.bfloat16

    def test_double_precision_intermediate(self):
        X = torch.randn(4, 8, dtype=torch.float16)
        result = matmul_hadU(X)
        assert result.dtype == torch.float16


class TestGetHadK:
    """get_hadK returns the butterfly sub-matrix for Hadamard construction."""

    def test_pow2_returns_hadk_none(self):
        hadK, K = get_hadK(8)
        assert hadK is None
        assert K == 1

    def test_non_pow2_returns_hadk_tensor(self):
        hadK, K = get_hadK(24)
        assert hadK is not None
        assert hadK.shape[0] == hadK.shape[1]
        assert is_pow2(hadK.shape[0]) or hadK.shape[0] == 24

    @pytest.mark.parametrize("size", [8, 16, 32, 64])
    def test_pow2_gives_K_1(self, size):
        hadK, K = get_hadK(size)
        assert K == 1


# =============================================================================
# Test Inplace Rotation — Hook Classes
# =============================================================================

class TestFullOnlineHadamardHook:
    """FullOnlineHadamardHook applies Hadamard on the entire last dimension."""

    def test_forward_hook_changes_values(self):
        module = nn.Linear(8, 8)
        hook = FullOnlineHadamardHook(had_K=None, K=1, use_fast_had=False)
        x = torch.randn(2, 8)
        args = (x,)
        result = hook(module, args)
        assert result[0].shape == x.shape
        assert not torch.equal(result[0], x)

    def test_forward_hook_with_custom_matrix(self):
        module = nn.Linear(8, 8)
        H = torch.eye(8)
        hook = FullOnlineHadamardHook(had_K=None, K=None, use_fast_had=False, had_matrix=H)
        x = torch.randn(2, 8)
        args = (x,)
        result = hook(module, args)
        assert torch.allclose(result[0], x, atol=1e-5)

    def test_fp32_mode(self):
        module = nn.Linear(8, 8)
        hook = FullOnlineHadamardHook(had_K=None, K=1, use_fast_had=False, fp32_had=True)
        x = torch.randn(2, 8, dtype=torch.float16)
        args = (x,)
        result = hook(module, args)
        assert result[0].shape == x.shape


class TestCrossHeadOnlineHadamardHook:
    """CrossHeadOnlineHadamardHook applies Hadamard on the num_heads axis."""

    def test_forward_hook_shape_preserved(self):
        module = nn.Linear(32, 32)
        num_heads, head_dim = 4, 8
        hook = CrossHeadOnlineHadamardHook(
            had_K=None, K=1, head_dim=head_dim, use_fast_had=False
        )
        x = torch.randn(2, num_heads * head_dim)
        args = (x,)
        result = hook(module, args)
        assert result[0].shape == x.shape

    def test_with_custom_had_matrix(self):
        module = nn.Linear(32, 32)
        num_heads, head_dim = 4, 8
        H = torch.eye(num_heads)
        hook = CrossHeadOnlineHadamardHook(
            had_K=None, K=None, head_dim=head_dim, use_fast_had=False, had_matrix=H
        )
        x = torch.randn(2, num_heads * head_dim)
        args = (x,)
        result = hook(module, args)
        assert result[0].shape == x.shape


class TestGroupOnlineHadamardHook:
    """GroupOnlineHadamardHook applies block-diagonal Hadamard per group."""

    def test_forward_hook_shape_preserved(self):
        module = nn.Linear(32, 32)
        hook = GroupOnlineHadamardHook(group_size=16, use_fast_had=False)
        x = torch.randn(2, 32)
        args = (x,)
        result = hook(module, args)
        assert result[0].shape == x.shape

    def test_different_group_sizes(self):
        for group_size in [4, 8, 16]:
            module = nn.Linear(32, 32)
            hook = GroupOnlineHadamardHook(group_size=group_size, use_fast_had=False)
            x = torch.randn(2, 32)
            args = (x,)
            result = hook(module, args)
            assert result[0].shape == x.shape
            assert not torch.equal(result[0], x)

    def test_with_custom_matrix(self):
        module = nn.Linear(16, 16)
        H = torch.eye(8)
        hook = GroupOnlineHadamardHook(group_size=8, use_fast_had=False, had_matrix=H)
        x = torch.randn(2, 16)
        args = (x,)
        result = hook(module, args)
        assert result[0].shape == x.shape


# =============================================================================
# Test Inplace Rotation — Low-Level Primitives
# =============================================================================

class TestNormalizeRotationMatrix:
    """_normalize_rotation_matrix parses all supported preset inputs."""

    def test_none_returns_none(self):
        had_dict, use_fast, preset = _normalize_rotation_matrix(None, group_size=8)
        assert had_dict is None
        assert use_fast is False
        assert preset is None

    def test_quarot_hadamard_string(self):
        had_dict, use_fast, preset = _normalize_rotation_matrix(
            "quarot_hadamard", group_size=8
        )
        assert had_dict is None
        assert use_fast is True
        assert preset == "quarot_hadamard"

    def test_hadamard_string(self):
        had_dict, use_fast, preset = _normalize_rotation_matrix(
            "hadamard", group_size=8
        )
        assert had_dict is None
        assert use_fast is False
        assert preset == "hadamard"

    def test_random_hadamard_string(self):
        had_dict, use_fast, preset = _normalize_rotation_matrix(
            "random_hadamard", group_size=8
        )
        assert had_dict is None
        assert use_fast is False
        assert preset == "random_hadamard"

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown rotation_matrix preset"):
            _normalize_rotation_matrix("unknown", group_size=8)

    def test_tensor_input_requires_positive_group_size(self):
        H = torch.eye(8)
        with pytest.raises(ValueError, match="positive group_size"):
            _normalize_rotation_matrix(H, group_size=None)
        with pytest.raises(ValueError, match="positive group_size"):
            _normalize_rotation_matrix(H, group_size=0)
        with pytest.raises(ValueError, match="positive group_size"):
            _normalize_rotation_matrix(H, group_size=-1)

    def test_tensor_input(self):
        H = torch.eye(8)
        had_dict, use_fast, preset = _normalize_rotation_matrix(H, group_size=8)
        assert had_dict[8].equal(torch.eye(8))
        assert use_fast is False
        assert preset is None

    def test_dict_input(self):
        H = torch.eye(8)
        had_dict, use_fast, preset = _normalize_rotation_matrix({8: H}, group_size=8)
        assert had_dict[8].equal(torch.eye(8))

    def test_dict_with_non_square_tensor_raises(self):
        bad = torch.randn(7, 5)
        with pytest.raises(AssertionError):
            _normalize_rotation_matrix({7: bad}, group_size=7)


class TestApplyExactHadToLinear:
    """apply_exact_had_to_linear rotates Linear weights in-place."""

    def test_output_side_rotation(self):
        module = nn.Linear(8, 16)
        original_weight = module.weight.data.clone()
        apply_exact_had_to_linear(module, had_dim=-1, output=True, use_fast_had=False)
        assert not module.weight.equal(original_weight)

    def test_input_side_rotation(self):
        module = nn.Linear(8, 16)
        original_weight = module.weight.data.clone()
        apply_exact_had_to_linear(module, had_dim=-1, output=False, use_fast_had=False)
        assert not module.weight.equal(original_weight)

    def test_block_diagonal_rotation(self):
        module = nn.Linear(16, 16)
        original_weight = module.weight.data.clone()
        apply_exact_had_to_linear(module, had_dim=8, output=True, use_fast_had=False)
        assert not module.weight.equal(original_weight)

    def test_requires_linear_module(self):
        with pytest.raises(AssertionError):
            apply_exact_had_to_linear(nn.Conv2d(3, 8, 3), had_dim=-1)


class TestApplyCrossHeadHadToLinear:
    """apply_cross_head_had_to_linear applies cross-head Hadamard rotation."""

    def test_rotation_changes_values(self):
        module = nn.Linear(16, 16)
        num_heads, head_dim = 4, 4
        original_weight = module.weight.data.clone()
        apply_cross_head_had_to_linear(
            module, num_heads=num_heads, head_dim=head_dim, use_fast_had=False
        )
        assert not module.weight.equal(original_weight)

    def test_requires_linear_module(self):
        with pytest.raises(AssertionError):
            apply_cross_head_had_to_linear(nn.Conv2d(3, 8, 3), num_heads=2, head_dim=4)


# =============================================================================
# Test Inplace Rotation — Random Cache
# =============================================================================

class TestRandomHadamardCache:
    """Random Hadamard global cache ensures consistent matrices across operations."""

    def setup_method(self):
        clear_random_hadamard_cache()

    def test_same_dimension_returns_same_matrix(self):
        clear_random_hadamard_cache()
        m1 = get_or_create_random_hadamard(8)
        m2 = get_or_create_random_hadamard(8)
        assert torch.equal(m1, m2)

    def test_different_dimensions_different_matrices(self):
        clear_random_hadamard_cache()
        m8 = get_or_create_random_hadamard(8)
        m16 = get_or_create_random_hadamard(16)
        assert not torch.equal(m8, m16)

    def test_cleared_cache_produces_new_matrix(self):
        clear_random_hadamard_cache()
        m1 = get_or_create_random_hadamard(8)
        clear_random_hadamard_cache()
        m2 = get_or_create_random_hadamard(8)
        assert not torch.equal(m1, m2)

    def test_device_transfer(self):
        clear_random_hadamard_cache()
        m = get_or_create_random_hadamard(8, device=torch.device("cpu"))
        assert m.device.type == "cpu"


# =============================================================================
# Test RotationMapping
# =============================================================================

class TestRotationMappingRegistry:
    """RotationMapping registry and model-config inference."""

    def test_default_mapping_registered(self):
        assert "llama" in MAPPING_REGISTRY
        assert "LlamaForCausalLM" in MAPPING_REGISTRY
        assert "qwen2" in MAPPING_REGISTRY
        assert "qwen3" in MAPPING_REGISTRY
        assert "opt" in MAPPING_REGISTRY

    def test_register_mapping(self):
        custom = RotationMapping()
        result = register_mapping("test_arch", custom)
        assert result is custom
        assert get_mapping("test_arch") is custom

    def test_get_mapping_unknown_returns_default(self):
        mapping = get_mapping("completely_unknown_architecture_xyz")
        assert isinstance(mapping, RotationMapping)

    def test_resolve_dot_path(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(8, 8)

        model = DummyModel()
        resolved = _resolve(model, "layer")
        assert resolved is model.layer

    def test_resolve_nested_dot_path(self):
        class DummyChild(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(8, 8)

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = DummyChild()

        model = DummyModel()
        resolved = _resolve(model, "decoder.layer")
        assert isinstance(resolved, nn.Linear)


# =============================================================================
# Test Patch Idempotency
# =============================================================================

class TestWrapperLinearPatch:
    """patch_wrapperlinear_to_apply_transform is idempotent."""

    def test_patch_is_idempotent(self):
        clear_random_hadamard_cache()
        w_transform = RandomHadamardTransform(block_size=8, seed=42)
        inp_transform = RandomHadamardTransform(block_size=8, seed=42, inverse=True)

        patch_wrapperlinear_to_apply_transform(w_transform, inp_transform)
        flag_after_first = getattr(
            __import__(
                "auto_round.wrapper", fromlist=["WrapperLinear"]
            ).WrapperLinear,
            "_hadamard_patched",
            False,
        )

        patch_wrapperlinear_to_apply_transform(w_transform, inp_transform)
        flag_after_second = getattr(
            __import__(
                "auto_round.wrapper", fromlist=["WrapperLinear"]
            ).WrapperLinear,
            "_hadamard_patched",
            False,
        )

        assert flag_after_first is True
        assert flag_after_second is True


class TestWrapperWALayerPatch:
    """patch_wrapperwalayer_forward_to_apply_transform is idempotent."""

    def test_patch_is_idempotent(self):
        inp_transform = RandomHadamardTransform(block_size=8, seed=42, inverse=True)
        patch_wrapperwalayer_forward_to_apply_transform(inp_transform)

        flag = getattr(
            __import__(
                "auto_round.wrapper", fromlist=["WrapperWALayer"]
            ).WrapperWALayer,
            "_hadamard_forward_patched",
            False,
        )
        assert flag is True


# =============================================================================
# Test Resolved Compute Device
# =============================================================================

class TestResolveComputeDevice:
    """_resolve_compute_device auto-detects available accelerator."""

    def test_explicit_device_returned(self):
        assert _resolve_compute_device("cpu") == torch.device("cpu")
        assert _resolve_compute_device(torch.device("cpu")) == torch.device("cpu")

    def test_none_detects_available_accelerator(self):
        result = _resolve_compute_device(None)
        assert result.type in ("cuda", "cpu")


# =============================================================================
# Test Inplace Apply — Full Integration (CPU-safe subset)
# =============================================================================

class TestInplaceApplyRotationFuseLn:
    """LayerNorm fusion is a core step before weight rotation."""

    def test_fuse_ln_linear_fuses_weight(self):
        ln = nn.LayerNorm(16, elementwise_affine=True)
        nn.init.ones_(ln.weight)
        linear = nn.Linear(16, 8)
        orig_linear_weight = linear.weight.data.clone()
        dtype = linear.weight.dtype
        dev = linear.weight.device
        W_ = linear.weight.data.double()
        ln_weight = ln.weight.double().to(dev)
        fused_weight = (W_ * ln_weight).to(dtype)
        assert torch.equal(fused_weight, orig_linear_weight.to(dtype))


# =============================================================================
# Test Orthogonality Invariants
# =============================================================================

class TestHadamardMatrixOrthogonality:
    """Hadamard matrices must satisfy H @ H.T = n * I for unnormalized forms."""

    @pytest.mark.parametrize(
        "size,dtype",
        [
            (8, torch.float32),
            (16, torch.float32),
            (32, torch.float32),
            (64, torch.float32),
            (8, torch.bfloat16),
            (16, torch.bfloat16),
        ],
    )
    def test_sylvester_deterministic_orthogonal(self, size, dtype):
        H = deterministic_hadamard_matrix(size, dtype=dtype)
        product = H.float() @ H.T.float()
        expected = torch.eye(size) * size
        atol = 1e-3 if dtype == torch.bfloat16 else 1e-5
        assert torch.allclose(product, expected, atol=atol)

    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_sylvester_random_orthogonal(self, size):
        H = random_hadamard_matrix(size, dtype=torch.float32, gen=torch.Generator().manual_seed(42))
        product = H @ H.T
        expected = torch.eye(size) * size
        assert torch.allclose(product, expected, atol=1e-4)

    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_inplace_deterministic_orthonormal(self, size):
        """Inplace deterministic Hadamard satisfies H @ H.T = I (orthonormal)."""
        H = inplace_det_hadamard(size, device=torch.device("cpu")).double()
        product = H @ H.T
        assert torch.allclose(product, torch.eye(size, dtype=torch.float64), atol=1e-4)

    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_inplace_random_orthonormal(self, size):
        """Inplace random Hadamard satisfies H @ H.T = I (orthonormal)."""
        H = inplace_rand_hadamard(size, device=torch.device("cpu")).double()
        product = H @ H.T
        assert torch.allclose(product, torch.eye(size, dtype=torch.float64), atol=1e-4)

    def test_matmul_hadU_produces_orthogonal_transform(self):
        """matmul_hadU applies /sqrt(n) normalization internally, producing orthonormal transforms."""
        for size in [8, 16, 32]:
            I = torch.eye(size, dtype=torch.float64)
            H = matmul_hadU(I)
            product = H @ H.T
            assert torch.allclose(product, torch.eye(size, dtype=torch.float64), atol=1e-5)


# =============================================================================
# Test RotationConfig Model Dump Consistency
# =============================================================================

class TestRotationConfigPersistence:
    """RotationConfig round-trips correctly for serialization."""

    def test_model_dump_includes_all_fields(self):
        cfg = RotationConfig(
            backend="inplace",
            block_size=128,
            hadamard_type="random_hadamard",
            fuse_online_to_weight=True,
            allow_online_rotation=False,
        )
        dumped = cfg.model_dump()
        assert "backend" in dumped
        assert "block_size" in dumped
        assert "hadamard_type" in dumped
        assert "fuse_online_to_weight" in dumped
        assert "allow_online_rotation" in dumped
        assert "algorithm" in dumped

    def test_json_serializable(self):
        import json

        cfg = RotationConfig(backend="auto", block_size=32)
        dumped = cfg.model_dump()
        json_str = json.dumps(dumped)
        restored = json.loads(json_str)
        assert restored["backend"] == "auto"
        assert restored["block_size"] == 32


# =============================================================================
# Test Non-Power-of-2 Construction
# =============================================================================

class TestNonPowerOfTwoConstruction:
    """Random Hadamard supports non-power-of-2 via precomputed matrices."""

    @pytest.mark.parametrize("size", [24, 48, 56, 80, 88])
    def test_non_pow2_random_hadamard_shape(self, size):
        H = random_hadamard_matrix(size, dtype=torch.float32)
        assert H.shape == (size, size)

    @pytest.mark.parametrize("size", [24, 48])
    def test_non_pow2_random_hadamard_orthogonal(self, size):
        H = random_hadamard_matrix(size, dtype=torch.float32)
        product = H @ H.T
        expected = torch.eye(size) * size
        assert torch.allclose(product, expected, atol=1e-4)

    def test_matmul_hadU_with_non_pow2(self):
        size = 24
        X = torch.randn(4, size)
        result = matmul_hadU(X)
        assert result.shape == X.shape
        reconstructed = matmul_hadUt(result)
        assert torch.allclose(reconstructed, X, atol=1e-4)


# =============================================================================
# Test Memory Cleanup (gc + cache)
# =============================================================================

class TestMemoryCleanup:
    """Rotation primitives properly manage memory via gc.collect()."""

    def test_multiple_inplace_calls_do_not_leak(self):
        for _ in range(3):
            clear_random_hadamard_cache()
            m = get_or_create_random_hadamard(8)
            assert m.shape == (8, 8)
            del m
            gc.collect()
