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


from contextlib import contextmanager
from typing import Any

from auto_round.algorithms.registry import resolve_pipeline_member
from auto_round.schemes import QuantizationScheme


class BaseAlgorithm:
    pass


class BasePipelineMember:
    """Shared interface for all members of a quantization pipeline."""

    model_context = None
    compress_context = None
    _scheme_context_fields = set(QuantizationScheme.get_attributes())
    bits: int | None
    group_size: int | tuple | None
    sym: bool | None
    data_type: str | None
    act_bits: int | None
    act_group_size: int | None
    act_sym: bool | None
    act_data_type: str | None
    act_dynamic: bool | None
    super_bits: int | None
    super_group_size: int | None
    super_sym: bool | None
    scale_dtype: str | None

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.scheme = getattr(config, "scheme", None)

    @classmethod
    def from_config(cls, config: Any) -> "BasePipelineMember":
        """Instantiate the registered implementation class for ``config``."""
        alg_cls = resolve_pipeline_member(config)
        if cls is alg_cls:
            return cls(config)
        return alg_cls(config)

    def bind(self, compressor: Any) -> None:
        """Wire shared context from the owning compressor."""
        self.model_context = compressor.model_context
        self.compress_context = compressor.compress_context
        self.scheme = getattr(compressor, "scheme_context", None)

    def prepare_run(self, compressor: Any) -> None:
        """Model-level preparation called once before block iteration starts."""
        return

    def get_act_calib_policy(self, ctx: Any) -> Any:
        """Return the activation calibration policy for this block."""
        from auto_round.algorithms.pipeline import ActCalibPolicy, CalibTiming, InputSource

        return ActCalibPolicy(when=CalibTiming.SKIP, source=InputSource.FP_CACHE)

    @contextmanager
    def block_forward_hooks(self, ctx: Any) -> Any:
        """Register algorithm-specific forward hooks for the reference forward."""
        yield []

    def finalize_run(self, compressor: Any) -> None:
        """Model-level teardown called once after all blocks are processed."""
        return


def _make_scheme_property(name):
    def getter(self):
        scheme = getattr(self, "scheme", None)
        return getattr(scheme, name, None) if scheme is not None else None

    def setter(self, value):
        scheme = getattr(self, "scheme", None)
        if scheme is None:
            raise AttributeError(f"{type(self).__name__} has no bound scheme")
        setattr(scheme, name, value)

    return property(getter, setter)


for _scheme_field in QuantizationScheme.get_attributes():
    setattr(BasePipelineMember, _scheme_field, _make_scheme_property(_scheme_field))
