# Copyright (c) 2023 Intel Corporation
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
__all__ = ["generate_rrq_residual"]


def __getattr__(name):
    if name == "generate_rrq_residual":
        from auto_round.export.export_to_autoround.export_to_rrq import generate_rrq_residual

        return generate_rrq_residual
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
