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


from typing import Any

from auto_round.algorithms.registry import resolve_pipeline_member
from auto_round.schemes import QuantizationScheme

# TODO later wenhuach may be deleted
class BasePipelineMember:
    """Shared interface for all members of a quantization pipeline."""

    model_context = None
    compress_context = None

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
        self.compressor = compressor
        self.model_context = compressor.model_context
        self.compress_context = compressor.compress_context
        self.scheme = getattr(compressor, "scheme_context", None)

    def prepare_run(self, compressor: Any) -> None:
        """Model-level preparation called once before block iteration starts."""
        return


    def finalize_run(self, compressor: Any) -> None:
        """Model-level teardown called once after all blocks are processed."""
        return

