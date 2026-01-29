# Copyright (c) 2025 Intel Corporation
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

from auto_round.modelling.replace_modules import (
    ReplacementModuleBase,
    apply_replacements,
    materialize_model_,
    release_original_module_,
)
from auto_round.modelling.moe_experts_impl import (
    linear_loop_experts_forward,
    register_linear_loop_experts,
    prepare_model_for_moe_quantization,
    is_linear_loop_available,
)

__all__ = [
    "ReplacementModuleBase",
    "apply_replacements",
    "materialize_model_",
    "release_original_module_",
    # Transformers-native MOE integration (transformers 5.0+)
    "linear_loop_experts_forward",
    "register_linear_loop_experts",
    "prepare_model_for_moe_quantization",
    "is_linear_loop_available",
]
