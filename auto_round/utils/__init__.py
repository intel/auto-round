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

from auto_round.utils.device import *
from auto_round.utils.common import *
from auto_round.utils.model import *

import transformers
from packaging.version import Version

# tmp batch for transformers v5.0
if Version(transformers.__version__) >= Version("5.0.0"):
    import datasets

    datasets.original_load_dataset = datasets.load_dataset

    def patch_load_dataset(*args, **kwargs):
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
