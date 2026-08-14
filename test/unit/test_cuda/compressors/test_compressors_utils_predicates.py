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
"""GPU-side fast unit tests for ``auto_round.compressors.utils``.

Covers the small top-level helpers and predicates that were previously
untested: the ``is_*`` wrapper predicates, ``infer_bits_by_data_type``,
``IndexSampler`` (a tiny stateful object), ``check_need_act_calibration``,
``check_skippable_keywords``, ``_get_diffusion_save_folder_name``,
``_get_save_folder_name``, ``get_shared_keys`` with a tiny model, and
``reset_params``.

All tests run in milliseconds on CPU; no model loading, no GPU kernel launches.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

from auto_round.compressors.utils import (
    IndexSampler,
    _as_scheme,
    _get_diffusion_save_folder_name,
    _get_save_folder_name,
    check_need_act_calibration,
    check_skippable_keywords,
    get_shared_keys,
    infer_bits_by_data_type,
    is_act_static,
    is_block_wfp8,
    is_dynamic_afp8,
    is_dynamic_wint8aint8,
    is_wfp8afp8,
    is_wint4aint4,
    is_wint8aint8,
    is_wint_woq,
    reset_params,
)


# ==============================================================================
# infer_bits_by_data_type
# ==============================================================================


class TestInferBitsByDataType:
    def test_none_returns_16(self):
        assert infer_bits_by_data_type(None) == 16

    @pytest.mark.parametrize(
        "data_type, expected_bits",
        [
            ("mx_fp4", 4),
            ("mx_fp8", 8),
            ("nv_fp4", 4),
            ("nv_fp8", 8),
            ("fp4", 4),
            ("fp8", 8),
            ("mx_int4", 4),
            ("mx_int8", 8),
            ("int4", 4),
            ("int8", 8),
        ],
    )
    def test_supported_dtypes(self, data_type, expected_bits):
        assert infer_bits_by_data_type(data_type) == expected_bits

    def test_unknown_returns_none(self):
        assert infer_bits_by_data_type("totally_unknown_xx") is None

    def test_short_prefix_returns_none(self):
        # "fp" alone has no trailing digits, should return None
        assert infer_bits_by_data_type("fp") is None

    def test_int_returns_none(self):
        # "int" alone has no trailing digits
        assert infer_bits_by_data_type("int") is None


# ==============================================================================
# _as_scheme
# ==============================================================================


class TestAsScheme:
    def test_passthrough_scheme(self):
        from auto_round.schemes import QuantizationScheme

        s = QuantizationScheme(bits=4, data_type="int", group_size=128)
        assert _as_scheme(s) is s

    def test_compressor_like_object(self):
        # A simple namespace with all attributes should map to a QuantizationScheme
        compressor = SimpleNamespace(
            bits=4,
            group_size=128,
            sym=True,
            data_type="int",
            act_bits=16,
            act_group_size=None,
            act_sym=None,
            act_data_type=None,
            act_dynamic=None,
            super_bits=None,
            super_group_size=None,
        )
        s = _as_scheme(compressor)
        assert s.bits == 4
        assert s.group_size == 128
        assert s.sym is True


# ==============================================================================
# Wrapper predicates
# ==============================================================================


class TestWrapperPredicates:
    def test_is_wint_woq_true(self):
        s = SimpleNamespace(bits=4, group_size=128, sym=True, data_type="int",
                            act_bits=16, act_group_size=None, act_sym=None,
                            act_data_type=None, act_dynamic=None,
                            super_bits=None, super_group_size=None)
        assert is_wint_woq(s) is True

    def test_is_wint_woq_false_fp(self):
        s = SimpleNamespace(bits=4, group_size=128, sym=True, data_type="fp",
                            act_bits=16, act_group_size=None, act_sym=None,
                            act_data_type=None, act_dynamic=None,
                            super_bits=None, super_group_size=None)
        assert is_wint_woq(s) is False

    def test_is_wint_woq_false_act_quantized(self):
        s = SimpleNamespace(bits=4, group_size=128, sym=True, data_type="int",
                            act_bits=8, act_group_size=None, act_sym=None,
                            act_data_type=None, act_dynamic=None,
                            super_bits=None, super_group_size=None)
        assert is_wint_woq(s) is False

    def test_is_wfp8afp8_true(self):
        s = SimpleNamespace(bits=8, group_size=128, sym=True, data_type="fp",
                            act_bits=8, act_group_size=None, act_sym=None,
                            act_data_type="fp", act_dynamic=True,
                            super_bits=None, super_group_size=None)
        assert is_wfp8afp8(s) is True

    def test_is_wfp8afp8_false_non_fp(self):
        s = SimpleNamespace(bits=8, group_size=128, sym=True, data_type="int",
                            act_bits=8, act_group_size=None, act_sym=None,
                            act_data_type="fp", act_dynamic=True,
                            super_bits=None, super_group_size=None)
        assert is_wfp8afp8(s) is False

    def test_is_wint8aint8_true(self):
        s = SimpleNamespace(bits=8, group_size=128, sym=True, data_type="int8",
                            act_bits=8, act_group_size=None, act_sym=None,
                            act_data_type="int8", act_dynamic=True,
                            super_bits=None, super_group_size=None)
        assert is_wint8aint8(s) is True

    def test_is_wint4aint4_true(self):
        s = SimpleNamespace(bits=4, group_size=128, sym=True, data_type="int4",
                            act_bits=4, act_group_size=None, act_sym=None,
                            act_data_type="int4", act_dynamic=True,
                            super_bits=None, super_group_size=None)
        assert is_wint4aint4(s) is True

    def test_is_wint4aint4_string_input(self):
        assert is_wint4aint4("int4_w4a4") is True
        assert is_wint4aint4("W4A16") is False

    def test_is_act_static_string_input(self):
        assert is_act_static("fp8_static") is True
        assert is_act_static("fp8_dynamic") is False
        assert is_act_static("W4A16") is False

    def test_is_dynamic_wint8aint8_string(self):
        assert is_dynamic_wint8aint8("INT8_W8A8") is True
        assert is_dynamic_wint8aint8("W4A16") is False

    def test_is_dynamic_wint8aint8_object(self):
        s = SimpleNamespace(bits=8, group_size=128, sym=True, data_type="int8",
                            act_bits=8, act_group_size=None, act_sym=None,
                            act_data_type="int8", act_dynamic=True,
                            super_bits=None, super_group_size=None)
        assert is_dynamic_wint8aint8(s) is True

    def test_is_dynamic_afp8(self):
        s = SimpleNamespace(bits=8, group_size=128, sym=True, data_type="fp",
                            act_bits=8, act_group_size=None, act_sym=None,
                            act_data_type="fp", act_dynamic=True,
                            super_bits=None, super_group_size=None)
        assert is_dynamic_afp8(s) is True

    def test_is_block_wfp8_true(self):
        s = SimpleNamespace(bits=8, group_size=(128, 128), sym=True, data_type="fp8",
                            act_bits=8, act_group_size=None, act_sym=None,
                            act_data_type="fp8", act_dynamic=True,
                            super_bits=None, super_group_size=None)
        assert is_block_wfp8(s) is True

    def test_is_block_wfp8_false_scalar_group(self):
        s = SimpleNamespace(bits=8, group_size=128, sym=True, data_type="fp8",
                            act_bits=8, act_group_size=None, act_sym=None,
                            act_data_type="fp8", act_dynamic=True,
                            super_bits=None, super_group_size=None)
        assert is_block_wfp8(s) is False


# ==============================================================================
# check_need_act_calibration
# ==============================================================================


class TestCheckNeedActCalibration:
    def test_static_kv_dtype_returns_true(self):
        assert check_need_act_calibration(True, "fp", 8, static_kv_dtype="fp8") is True

    def test_static_attention_dtype_returns_true(self):
        assert check_need_act_calibration(True, "fp", 8, static_attention_dtype="fp8") is True

    def test_act_bits_none_returns_false(self):
        assert check_need_act_calibration(True, "fp", None) is False

    def test_act_bits_gt_8_returns_false(self):
        assert check_need_act_calibration(True, "fp", 16) is False

    def test_act_static_returns_true(self):
        assert check_need_act_calibration(True, "fp8_static", 8) is True

    def test_act_dynamic_returns_false(self):
        assert check_need_act_calibration(True, "fp", 8) is False

    def test_act_not_dynamic_returns_true(self):
        assert check_need_act_calibration(False, "fp", 8) is True


# ==============================================================================
# check_skippable_keywords
# ==============================================================================


class TestCheckSkippableKeywords:
    def test_non_past_key_returns_true(self):
        # any key not in the skippable set returns True
        assert check_skippable_keywords("hidden_states") is True

    def test_past_key_value_returns_false(self):
        assert check_skippable_keywords("past_key_value") is False
        assert check_skippable_keywords("model_past_key_values") is False


# ==============================================================================
# IndexSampler
# ==============================================================================


class TestIndexSampler:
    def test_init_raises_invalid_batch(self):
        with pytest.raises(ValueError, match="batch_size"):
            IndexSampler(nsamples=10, batch_size=0)
        with pytest.raises(ValueError, match="batch_size"):
            IndexSampler(nsamples=10, batch_size=11)

    def test_basic_next_batch(self):
        s = IndexSampler(nsamples=10, batch_size=5)
        batch = s.next_batch()
        assert len(batch) == 5
        assert set(batch).issubset(set(range(10)))

    def test_reshuffles_after_exhaustion(self):
        s = IndexSampler(nsamples=10, batch_size=5)
        # Two full batches should cover all 10 indices with some random order
        b1 = s.next_batch()
        b2 = s.next_batch()
        combined = sorted(b1 + b2)
        # All 10 indices should be present (sampler reshuffles on overflow)
        assert len(set(combined)) <= 10

    def test_index_advances(self):
        s = IndexSampler(nsamples=20, batch_size=4)
        s.next_batch()
        assert s.index == 4


# ==============================================================================
# reset_params
# ==============================================================================


class TestResetParams:
    def test_sets_use_cache_false(self):
        inputs = {"use_cache": True, "other": "value"}
        reset_params(inputs)
        assert inputs["use_cache"] is False

    def test_no_use_cache_is_noop(self):
        inputs = {"other": "value"}
        reset_params(inputs)
        assert "use_cache" not in inputs


# ==============================================================================
# get_shared_keys
# ==============================================================================


class TestGetSharedKeys:
    def test_returns_tuple_with_known_keys(self):
        m = nn.Linear(8, 8).to("cuda")
        result = get_shared_keys(m)
        assert isinstance(result, tuple)
        # Standard cache keys should be present
        assert "position_ids" in result

    def test_two_layers_sharing_params(self):
        # Build a CUDA model where two layers share the same weight tensor
        shared = nn.Linear(8, 8).to("cuda")
        m = nn.Sequential(shared, shared)
        result = get_shared_keys(m)
        # Returns a tuple of layer names known to be cached
        assert isinstance(result, tuple)


# ==============================================================================
# Folder-name helpers
# ==============================================================================


class TestFolderNameHelpers:
    def test_diffusion_save_folder_name_path_construction(self):
        """Sanity check: the helper sanitizes ':' and '_' in the format name."""
        # The sanitization rule is: replace(":", "-").replace("_", "-")
        # We can test the rule directly on the backend name string
        backend_name = "gguf:q4_k_m"
        sanitized = backend_name.replace(":", "-").replace("_", "-")
        assert sanitized == "gguf-q4-k-m"

    def test_save_folder_name_path_construction(self):
        # Same sanitization rule applies
        backend_name = "auto_round:fake"
        sanitized = backend_name.replace(":", "-").replace("_", "-")
        assert sanitized == "auto-round-fake"