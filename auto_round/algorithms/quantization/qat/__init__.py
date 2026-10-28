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
"""QAT (quantization-aware training) algorithm package.

QAT tunes the whole network jointly against the ground-truth next-token
cross-entropy -- i.e. the pure hard-label special case of QAD with no teacher
model and no offline teacher cache. It reuses QAD's whole-model tuning loop via
a thin subclass.
"""

from auto_round.algorithms.quantization.qat.config import QATConfig

__all__ = ["QATConfig"]

