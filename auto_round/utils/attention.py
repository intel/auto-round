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

"""Attention projection naming shared by transforms and scale synchronization."""

from torch.nn import Module


def get_fused_attention_projection_names(module: Module) -> tuple[str, ...]:
    """Recognize QKV names; callers must validate shared-input requirements."""
    if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj")):
        return ("q_proj", "k_proj", "v_proj")
    if not getattr(module, "is_cross_attention", False) and all(
        hasattr(module, name) for name in ("to_q", "to_k", "to_v")
    ):
        return ("to_q", "to_k", "to_v")
    return ()
