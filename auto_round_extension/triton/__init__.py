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


def patch_triton_autotuner_if_needed():
    """Patch Autotuner if it's missing _cache_lock (compatibility fix for older triton versions)."""
    try:
        import threading
        from triton.runtime.autotuner import Autotuner

        if not hasattr(Autotuner, "_cache_lock"):
            Autotuner._cache_lock = threading.RLock()
        if not hasattr(Autotuner, "_cache_futures"):
            Autotuner._cache_futures = {}
        if not hasattr(Autotuner, "_cache"):
            Autotuner._cache = {}
        if not hasattr(Autotuner, "cache"):
            Autotuner.cache = {}
    except (ImportError, AttributeError):
        pass


patch_triton_autotuner_if_needed()
