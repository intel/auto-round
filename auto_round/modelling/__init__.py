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


import importlib
import pkgutil
from auto_round.utils import logger

for module_info in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
    module_name = module_info.name
    # Skip private modules
    if module_name.split(".")[-1].startswith("_"):
        continue
    try:
        mod = importlib.import_module(module_name)
        # Re-export symbols
        if hasattr(mod, "__all__"):
            for name in mod.__all__:
                globals()[name] = getattr(mod, name)
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning("Optional module %s not available: %s", module_name, e)
