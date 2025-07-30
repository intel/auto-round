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


quantization_config = {
    "config_groups": {
        "group_0": {
            "input_activations": {
                "actorder": None,
                "block_structure": None,
                "dynamic": True,
                "group_size": None,
                "num_bits": 8,
                "observer": None,
                "observer_kwargs": {},
                "strategy": "token",
                "symmetric": True,
                "type": "int",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": None,
                "num_bits": 8,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "channel",
                "symmetric": True,
                "type": "int",
            },
        }
    },
    "format": "int-quantized",
    "global_compression_ratio": 1.5,  # not exactly correct, but a placeholder
    "ignore": [],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "provider": "auto-round",  # This is the key to indicate the model provider
}
