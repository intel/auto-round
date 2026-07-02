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

import logging
import re
import subprocess
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def is_oneapi_ge_2026() -> bool:
    try:
        output = subprocess.check_output(
            ["icpx", "--version"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    match = re.search(r"Compiler\s+(\d{4})\.", output)
    return bool(match and int(match.group(1)) >= 2026)


B70_DEVICE_ID = "0xe223"
B70_IDENTIFIERS = (B70_DEVICE_ID, "b70")


@lru_cache(maxsize=None)
def is_b70(device: int = 0) -> bool:
    try:
        pro = torch.xpu.get_device_properties(device)
    except Exception:
        return False

    return any(identifier in pro.name.lower() for identifier in B70_IDENTIFIERS) or hex(pro.device_id) == B70_DEVICE_ID


@lru_cache(maxsize=None)
def fallback_compute_type_if_needed(compute_dtype: str, device: int = 0) -> str:
    if compute_dtype.lower() == "int8" and is_b70(device) and not is_oneapi_ge_2026():
        logger.warning("XMX int8 is not supported on B70 with oneAPI < 2026. Falling back to fp16.")
        return "fp16"

    return compute_dtype
