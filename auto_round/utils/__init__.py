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

"""Utility package for AutoRound.

Provides device management, model utilities, common helpers, weight handling,
and dataset patching for AutoRound quantization workflows.
"""

from auto_round.utils.device import *
from auto_round.utils.common import *
from auto_round.utils.model import *
from auto_round.utils.weight_handler import (
    convert_module_to_hp_if_necessary,
    detect_weight_type,
    is_quantized_input_module,
)
from auto_round.utils.missing_tensors import copy_missing_tensors_from_source

import transformers
from packaging.version import Version

DATASET_PATCHED = False
# tmp batch for transformers v5.0
if Version(transformers.__version__) >= Version("5.0.0") and not DATASET_PATCHED:
    import datasets

    datasets.original_load_dataset = datasets.load_dataset

    def patch_load_dataset(*args, **kwargs):
        """Patch datasets.load_dataset to remap legacy dataset names to updated paths.

        Replaces known renamed dataset identifiers (e.g., ``openbookqa`` →
        ``allenai/openbookqa``) in the positional arguments and keyword arguments
        ``path`` and ``name`` before forwarding to the original loader.

        Args:
            *args: Positional arguments forwarded to the original load_dataset.
            **kwargs: Keyword arguments forwarded to the original load_dataset.

        Returns:
            The dataset returned by the original datasets.load_dataset call.
        """
        for dataset_name, replace_name in [("openbookqa", "allenai/openbookqa")]:
            if len(args) > 0 and dataset_name in args[0]:
                args = (replace_name,) + args[1:]
            if "path" in kwargs and kwargs["path"] is not None:
                if dataset_name in kwargs["path"] and replace_name not in kwargs["path"]:
                    kwargs["path"] = kwargs["path"].replace(dataset_name, replace_name)
            if "name" in kwargs and kwargs["name"] is not None:
                if dataset_name in kwargs["name"] and replace_name not in kwargs["name"]:
                    kwargs["name"] = kwargs["name"].replace(dataset_name, replace_name)
        return datasets.original_load_dataset(*args, **kwargs)

    datasets.load_dataset = patch_load_dataset
    DATASET_PATCHED = True
