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

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoConfig


def load_mllm(pretrained_model_name_or_path, **kwargs):
    """Load MLLMs.
    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model.
        kwargs: Variable number of keyword arguments.
    """
    trust_remote_code = True
    if "trust_remote_code" in kwargs:
        trust_remote_code = kwargs.pop("trust_remote_code")
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
    tokenizer.processor = processor
    model_type = config.model_type

    if "qwen2_vl" in model_type:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)
    elif "mllama" in model_type:
        from transformers import MllamaForConditionalGeneration
        model = MllamaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code, attn_implementation="eager", **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)

    if "cogvlm2" in pretrained_model_name_or_path:
        model.config.model_type = "cogvlm2"

    return model, tokenizer, processor


def _extract_data_dir(dir_path):
    if os.path.isdir(dir_path):
        return dir_path
    else:
        result = {}
        dir_path = dir_path.split(",")
        for _path in dir_path:
            k, v = _path.split('=')
            if k in ['image', 'video', 'audio']:
                result[k] = v
        return result


