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
from typing import ClassVar

from auto_round.algorithms.config import AlgorithmConfig
from auto_round.schemes import QuantizationScheme


class QuantizationConfig(AlgorithmConfig):
    """Algorithm-local calibration settings; AutoRound owns the quantization scheme."""

    need_calib = True
    stage_local_fields: ClassVar[frozenset[str]] = frozenset({"nsamples"})

    def __init__(self, *, nsamples: int | None = None, **kwargs) -> None:
        self.reject_scheme_arguments(kwargs)
        if kwargs:
            unknown_args = ", ".join(repr(arg) for arg in kwargs)
            raise TypeError(f"Unexpected quantization config argument(s): {unknown_args}")
        self.nsamples = nsamples
        self.check_config()

    @staticmethod
    def reject_scheme_arguments(kwargs) -> None:
        rejected = [
            key
            for key in kwargs
            if key == "scheme" or key in QuantizationScheme.get_attributes() or key.startswith(("act_", "super_"))
        ]
        if rejected:
            names = ", ".join(repr(key) for key in rejected)
            raise TypeError(
                f"Quantization scheme argument(s) {names} belong to AutoRound; pass them to AutoRound instead."
            )

    def check_config(self) -> None:
        """Validate the algorithm-local calibration sample override."""
        if self.nsamples is not None and (
            isinstance(self.nsamples, bool) or not isinstance(self.nsamples, int) or self.nsamples <= 0
        ):
            raise ValueError("`nsamples` must be a positive integer or None")
