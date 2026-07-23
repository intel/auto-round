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

from dataclasses import fields

from auto_round.planning import CompressionPlan
from auto_round.schemes import QuantizationScheme
from auto_round.utils.model import get_module


def apply_plan_to_model(model, plan: CompressionPlan) -> None:
    """Apply a resolved plan at the single explicit module-attribute write boundary."""
    scheme_keys = tuple(field.name for field in fields(QuantizationScheme)) + ("scale_dtype",)
    for module_name, module in model.named_modules():
        for key in scheme_keys:
            if module_name == "" and key == "rotation_config":
                continue
            if hasattr(module, key):
                delattr(module, key)

    for layer_name, config in plan.layer_config.items():
        module = get_module(model, layer_name)
        if module is None:
            continue
        for name, value in config.items():
            setattr(module, name, value)
