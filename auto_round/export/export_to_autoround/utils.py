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


def check_neq_config(config, data_type, bits, act_bits, group_size, sym):
    """
    Checks if the provided configuration parameters are not equal to the values in the config dictionary.

    Args:
        config (dict): A dictionary containing the configuration parameters.
        data_type (str): The expected data type.
        bits (int): The expected number of bits.
        group_size (int): The expected group size.
        sym (bool): The expected symmetry flag.

    Returns:
        list: A list of strings indicating which configuration parameters do not match.
    """
    expected_config = {
        "data_type": data_type,
        "bits": bits,
        "group_size": group_size,
        "sym": sym,
        "act_bits": act_bits,
    }
    return [key for key, expected_value in expected_config.items() if config.get(key) != expected_value]
