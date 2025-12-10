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

import os

import requests

from auto_round.utils import LazyImport

PIL = LazyImport("PIL")
Image = LazyImport("PIL.Image")

VISUAL_KEYS = [
    "thinker",
    "visual",
    "audio",
    "talker",
    "token2wav",
    "multi_modal_projector",
    "vision_tower",
    "multimodal_projector",
    "vision_model",
    "model.connector",
]


def _extract_data_dir(dir_path: str):
    if os.path.isdir(dir_path):
        return dir_path
    elif "=" in dir_path:
        result = {}
        dir_path = dir_path.split(",")
        for _path in dir_path:
            k, v = _path.split("=")
            if k in ["image", "video", "audio"]:
                result[k] = v
        return result
    else:
        raise TypeError("incorrect input of extra_data_dir, please use auto_round --help for more details.")


def fetch_image(path_or_url):
    if os.path.isfile(path_or_url):
        image_obj = Image.open(path_or_url)
    elif path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        image_obj = Image.open(requests.get(path_or_url, stream=True).raw)
    else:
        raise TypeError(f"{path_or_url} neither a path or url.")

    return image_obj
