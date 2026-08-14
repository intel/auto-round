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

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

from auto_round.algorithms.config_resolver import (
    get_algorithm_class,
    is_block_quantizer_config,
    is_preprocessor_config,
    resolve_shared_config_values,
    split_quantization_configs,
    sync_shared_config_from,
)
from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.transforms.awq.config import AWQConfig


class TestGetAlgorithmClass:
    def test_rtn_config_resolves_to_quantizer(self):
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        assert get_algorithm_class(RTNConfig()) is OptimizedRTNQuantizer

    def test_awq_config_resolves_to_preprocessor(self):
        from auto_round.algorithms.transforms.awq.base import AWQTransform

        assert get_algorithm_class(AWQConfig()) is AWQTransform

    def test_plain_quantization_config_returns_none(self):
        # QuantizationConfig itself is not registered to any implementation
        assert get_algorithm_class(QuantizationConfig()) is None

    def test_unknown_object_returns_none(self):
        class _NotAConfig:
            pass

        assert get_algorithm_class(_NotAConfig()) is None


class TestIsPreprocessorBlockQuantizer:
    def test_awq_is_preprocessor_not_quantizer(self):
        assert is_preprocessor_config(AWQConfig()) is True
        assert is_block_quantizer_config(AWQConfig()) is False

    def test_rtn_is_quantizer_not_preprocessor(self):
        assert is_block_quantizer_config(RTNConfig()) is True
        assert is_preprocessor_config(RTNConfig()) is False

    def test_plain_config_is_neither(self):
        assert is_preprocessor_config(QuantizationConfig()) is False
        assert is_block_quantizer_config(QuantizationConfig()) is False


class TestSplitQuantizationConfigs:
    def test_splits_preprocessors_and_quantizers(self):
        pre, blk = split_quantization_configs([AWQConfig(), RTNConfig()])
        assert len(pre) == 1
        assert len(blk) == 1
        assert isinstance(pre[0], AWQConfig)
        assert isinstance(blk[0], RTNConfig)

    def test_unregistered_configs_are_dropped(self):
        pre, blk = split_quantization_configs([QuantizationConfig(), AWQConfig(), RTNConfig()])
        assert len(pre) == 1
        assert len(blk) == 1


class TestResolveSharedSchemeValues:
    def test_inherits_user_unset_field(self):
        c1 = QuantizationConfig(bits=4, group_size=128, sym=True)
        c2 = QuantizationConfig(sym=True)  # bits not user-set
        resolve_shared_config_values([c1, c2])
        assert c2.scheme.bits == 4

    def test_conflicting_scheme_fields_raise(self):
        c1 = QuantizationConfig(bits=4)
        c2 = QuantizationConfig(bits=8)
        with pytest.raises(ValueError, match="Conflicting shared scheme field"):
            resolve_shared_config_values([c1, c2])

    def test_does_not_override_user_set_field(self):
        # c2 explicitly sets bits=8 -> must not be overwritten by c1.bits=4
        c1 = QuantizationConfig(bits=4)
        c2 = QuantizationConfig(bits=8)
        with pytest.raises(ValueError, match="Conflicting shared scheme field"):
            resolve_shared_config_values([c1, c2])
        assert c2.scheme.bits == 8


class _SharedAttrConfig(QuantizationConfig):
    """QuantizationConfig with an extra public attribute that can be None."""

    def __init__(self, *, extra=None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "extra", extra)


class TestResolveSharedPublicValues:
    def test_none_inherits_single_value(self):
        c1 = _SharedAttrConfig(extra=5)
        c2 = _SharedAttrConfig(extra=None)
        resolve_shared_config_values([c1, c2])
        assert c2.extra == 5
        assert c1.extra == 5

    def test_conflicting_public_values_raise(self):
        c1 = _SharedAttrConfig(extra=1)
        c2 = _SharedAttrConfig(extra=2)
        with pytest.raises(ValueError, match="Conflicting shared config field"):
            resolve_shared_config_values([c1, c2])

    def test_single_field_no_conflict(self):
        c1 = _SharedAttrConfig(extra=1)
        c2 = _SharedAttrConfig()  # no extra attr at all
        resolve_shared_config_values([c1, c2])
        assert c1.extra == 1


class TestSyncSharedConfigFrom:
    def test_propagates_source_value(self):
        source = RTNConfig(disable_opt_rtn=True)
        target = RTNConfig(disable_opt_rtn=False)
        sync_shared_config_from(source, [target])
        assert target.disable_opt_rtn is True

    def test_skips_source_itself(self):
        source = RTNConfig(disable_opt_rtn=True)
        sync_shared_config_from(source, [source])
        assert source.disable_opt_rtn is True
