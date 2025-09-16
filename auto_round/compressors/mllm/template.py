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

import json
import os
from dataclasses import dataclass
from enum import Enum, unique
from typing import Dict, List, Optional

from auto_round.logger import logger
from .processor import PROCESSORS, BasicProcessor

TEMPLATES: Dict[str, "Template"] = {}


def fill_content(target, **kwargs):
    for name, value in kwargs.items():
        target = target.replace("{{" + name + "}}", value, 1)
    return target


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


@dataclass
class Template:
    model_type: str
    format_user: str
    format_assistant: str
    format_system: str
    format_function: str
    format_observation: str
    format_separator: str
    default_system: str
    replace_tokens: List[tuple]
    extra_encode: bool
    default_dataset: str
    processor: "BasicProcessor"

    def _encode(self, sources):
        """Encodes formatted inputs to pairs of token ids."""
        if self.extra_encode:
            element = ""
            for i, source in enumerate(sources):
                if i == 0:
                    element += fill_content(self.format_system, content=self.default_system)
                # if i > 0 and i % 2 ==0:
                #     element += fill_content(self.format_separator)

                if source["role"] == Role.USER.value:
                    element += fill_content(self.format_user, content=source["content"])
                elif source["role"] == Role.ASSISTANT.value:
                    element += fill_content(self.format_assistant, content=source["content"])
                elif source["role"] == Role.OBSERVATION.value:
                    element += fill_content(self.format_observation, content=source["content"])
                elif source["role"] == Role.FUNCTION.value:
                    element += fill_content(self.format_function, content=source["content"])
            return element
        else:
            return sources


def _register_template(
    model_type: str,
    format_user: Optional[str] = None,
    format_assistant: Optional[str] = None,
    format_system: Optional[str] = None,
    format_function: Optional[str] = None,
    format_observation: Optional[str] = None,
    format_separator: Optional[str] = None,
    default_system: str = "",
    replace_tokens: List[tuple] = None,
    extra_encode: Optional[bool] = False,
    default_dataset: Optional[bool] = "NeelNanda/pile-10k",
    processor: "BasicProcessor" = PROCESSORS["basic"],
):
    """Registers a chat template."""
    template_class = Template
    default_format_user = "{{content}}"
    default_format_assistant = "{{content}}"
    default_format_system = "{{content}}"
    default_format_function = ""
    default_format_observation = ""
    default_format_separator = "\n"
    TEMPLATES[model_type] = template_class(
        model_type=model_type,
        format_user=format_user or default_format_user,
        format_assistant=format_assistant or default_format_assistant,
        format_system=format_system or default_format_system,
        format_function=format_function or default_format_function,
        format_observation=format_observation or default_format_observation,
        format_separator=format_separator or default_format_separator,
        default_system=default_system,
        replace_tokens=replace_tokens,
        extra_encode=extra_encode,
        default_dataset=default_dataset,
        processor=processor(),
    )
    return TEMPLATES[model_type]


_register_template("qwen2_vl", default_dataset="NeelNanda/pile-10k", processor=PROCESSORS["qwen2_vl"])
_register_template("qwen2_5_vl", default_dataset="NeelNanda/pile-10k", processor=PROCESSORS["qwen2_vl"])
_register_template("mllama", default_dataset="liuhaotian/llava", processor=PROCESSORS["hf"])
_register_template("deepseek_vl_v2", default_dataset="NeelNanda/pile-10k", processor=PROCESSORS["deepseek_v2"])
_register_template("mistral3", default_dataset="NeelNanda/pile-10k", processor=PROCESSORS["hf"])
_register_template("mistral3_2", default_dataset="liuhaotian/llava", processor=PROCESSORS["mistral3_2"])


def load_template(path: str):
    """Load template information from a json file."""
    with open(path, "r") as file:
        data = json.load(file)
        if "model_type" not in data:
            data["model_type"] = "user_define"
        if "replace_tokens" in data and data["replace_tokens"] is not None:
            if len(data["replace_tokens"]) % 2 != 0:
                raise ValueError(
                    "the format of replace_tokens should be " "[old_tag1, replace_tag1, old_tag2, replace_tag2]"
                )
            temp = []
            for i in range(0, len(data["replace_tokens"]), 2):
                temp.append((data["replace_tokens"][i], data["replace_tokens"][i + 1]))
            data["replace_tokens"] = temp
        if "processor" in data:
            if data["processor"] not in PROCESSORS.keys():
                raise ValueError(
                    f"{data['processor']} is not supported, current support: " "{','.join(PROCESSORS.keys())}"
                )
            data["processor"] = PROCESSORS[data["processor"]]
        template = _register_template(**data)
        return template


def _load_preset_template():
    dir_path = os.path.join(os.path.dirname(__file__), "templates")
    for file_name in os.listdir(dir_path):
        load_template(os.path.join(dir_path, file_name))


_load_preset_template()


def get_template(
    template_or_path: str, model=None, tokenizer=None, processor=None, image_processor=None, use_rtn=False, quiet=False
):
    """Get template by template name or from a json file.

    Args:
        template_or_path (str): Template name or a path of the template json file.

    Returns:
        The Template.

    """
    if os.path.isfile(template_or_path):
        template = load_template(template_or_path)
    else:
        if template_or_path in TEMPLATES:
            template = TEMPLATES[template_or_path]
        else:
            if not quiet:
                logger.warning(f"Unable to recognize {template_or_path}, using default template instead.")
            template = TEMPLATES["default"]
            template.model_type = template_or_path

    template.processor.post_init(
        model=model, tokenizer=tokenizer, processor=processor, image_processor=image_processor, use_rtn=use_rtn
    )

    return template
