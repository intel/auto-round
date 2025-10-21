# Copyright (c) 2024 Intel Corporation
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

from typing import List

from auto_round.schemes import QuantizationScheme

REQUIRED_CONFIG_KEYS = {key for key in QuantizationScheme.__dataclass_fields__.keys()}  # pylint: disable=no-member


def check_neq_config(config: dict, **expected) -> List[str]:
    """
    Compare a config dict against expected values.
    Ensures all required keys are present in both config and expected.

    Returns:
        List[str]: [keys] for mismatched values.
    """
    # 1. Check missing from expected
    missing_expected = [k for k in REQUIRED_CONFIG_KEYS if k not in expected]
    if missing_expected:
        raise ValueError(f"Missing expected values for keys: {missing_expected}")

    # 2. Check missing from layer config
    missing_config = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing_config:
        raise ValueError(f"Missing config values for keys: {missing_config}")

    # 3. Collect mismatches
    return [key for key in REQUIRED_CONFIG_KEYS if config[key] != expected[key] and config[key] is not None]
