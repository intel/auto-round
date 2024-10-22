import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, List, Union, Sequence

from .plugin import BasicPlugin, PLUGINS

TEMPLATES: Dict[str, "Template"] = {}

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
    plugin: "BasicPlugin"


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
    plugin: "BasicPlugin" = PLUGINS["basic"]
):
    template_class = Template
    default_format_user = "{{content}}"
    default_format_assistant = "{{content}}"
    default_format_system = "{{content}}"
    default_format_function = ""
    default_format_observation = ""
    default_format_separator = "\n"
    default_replace_tokens = []
    TEMPLATES[model_type] = template_class(
        model_type = model_type,
        format_user = format_user or default_format_user,
        format_assistant = format_assistant or default_format_assistant,
        format_system = format_system or default_format_system, 
        format_function = format_function or default_format_function,
        format_observation = format_observation or default_format_observation,
        format_separator = format_separator or default_format_separator,
        default_system = default_system,
        replace_tokens = replace_tokens or default_replace_tokens,
        plugin = plugin
    )
    return TEMPLATES[model_type]


def load_template(path: str):
    data = json.load(open(path, "r"))
    if "model_type" not in data:
        data["model_type"] = "user_define"
    if "replace_tokens" in data:
        assert len(data["replace_tokens"]) % 2 == 0, \
            "the format of replace_tokens should be [old_tag1, replace_tag1, old_tag2, replace_tag2]"
        temp = []
        for i in range(0, len(data["replace_tokens"]), 2):
           temp.append((data["replace_tokens"][i], data["replace_tokens"][i+1])) 
        data["replace_tokens"] = temp
    if "plugin" in data:
        assert data["plugin"] in PLUGINS.keys(), \
            "{} is not supported, current support: {}".format(data["plugin"], ",".join(PLUGINS.keys()))
        data["plugin"] = PLUGINS[data["plugin"]]
    template = _register_template(
        **data
    )
    return template


def _load_preset_template():
    dir_path = os.path.join(os.path.dirname(__file__), 'templates')
    for file_name in os.listdir(dir_path):
        load_template(os.path.join(dir_path, file_name))

_load_preset_template()