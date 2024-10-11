import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, List, Union, Sequence
from enum import Enum, unique

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROLE_MAPPING = {"human": "user", "gpt": "assistant"}

@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"

@dataclass
class Template:
    format_user: str
    format_assistant: str
    format_system: str
    format_function: str
    format_observation: str
    format_separator: str
    default_system: str
    role_mapping: dict
    replace_tokens: List[tuple]



TEMPLATES: Dict[str, "Template"] = {}

def _register_template(
    name: str,
    format_user: Optional[str] = None,
    format_assistant: Optional[str] = None,
    format_system: Optional[str] = None,
    format_function: Optional[str] = None,
    format_observation: Optional[str] = None,
    format_separator: Optional[str] = None,
    default_system: str = "",
    role_mapping: dict = None,
    replace_tokens: List[tuple] = None
):
    template_class = Template
    default_format_user = "{{content}}"
    default_format_assistant = "{{content}}"
    default_format_system = "{{content}}"
    default_format_function = ""
    default_format_observation = ""
    default_format_separator = "\n"
    default_replace_tokens = []
    TEMPLATES[name] = template_class(
        format_user = format_user or default_format_user,
        format_assistant = format_assistant or default_format_assistant,
        format_system = format_system or default_format_system, 
        format_function = format_function or default_format_function,
        format_observation = format_observation or default_format_observation,
        format_separator = format_separator or default_format_separator,
        default_system = default_system,
        role_mapping = role_mapping or ROLE_MAPPING,
        replace_tokens = replace_tokens or default_replace_tokens
    )

_register_template(
    name="qwen2_vl",
    format_user="<|im_start|>user\n{{content}}<|im_end|>\n",
    format_assistant="<|im_start|>assistant\n{{content}}<|im_end|>\n",
    format_system="<|im_start|>system\n{{content}}<|im_end|>\n",
    format_observation="<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n",
    format_separator="\n",
    default_system="You are a helpful assistant.",
    role_mapping=ROLE_MAPPING,
    replace_tokens=[("<image>", "<|vision_start|><|image_pad|><|vision_end|>")]
)

_register_template(
    name="mllama",
    format_user="<|start_header_id|>user<|end_header_id|>\n{{content}}<|eot_id|>",
    format_system="<|begin_of_text|><|start_header_id|>system\n{{content}}<|end_header_id|>\n",
    # format_observation="<|start_header_id|>tool\n{{content}}<|end_header_id|>\n<|im_start|>assistant\n",
    default_system="You are a helpful assistant.",
    role_mapping=ROLE_MAPPING,
    replace_tokens=[("<image>", "<|image|>")]
)


def fill_content(target, **kwargs):
    for name, value in kwargs.items():
        target = target.replace("{{" + name + "}}", value, 1)
    return target


def vlm_encode(sources, template: "Template"):
    element = ""
    for i, source in enumerate(sources):
        if i == 0:
            element += fill_content(template.format_system, content=template.default_system)
        # if i > 0 and i % 2 ==0:
        #     element += fill_content(template.format_separator)
        
        if source['role'] == Role.USER.value:
            element += fill_content(template.format_user, content=source["content"])
        elif source['role'] == Role.ASSISTANT.value:
            element += fill_content(template.format_assistant, content=source["content"])
        elif source['role'] == Role.OBSERVATION.value:
            element += fill_content(template.format_observation, content=source["content"])
        elif source['role'] == Role.FUNCTION.value:
            element += fill_content(template.format_function, content=source["content"])
    return element

            

class LlavaDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, model_type, tokenzier, question_path, image_path, max_len) -> None:
        super().__init__()
        assert model_type in TEMPLATES, f"{model_type} is not supported"
        self.model_type = model_type
        self.template = TEMPLATES[model_type]
        self.tokenizer = tokenzier
        self.questions = json.load(open(question_path, "r"))
        self.image_path = image_path
        self.max_len = max_len
    

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        text = self.questions[i]["conversations"]
        text = self.covert_conversations(text)
        text = vlm_encode(text, self.template)

        token_length = len(self.tokenizer(text).input_ids)
        if token_length < self.max_len:
            text += self.tokenizer.pad_token * (self.max_len - token_length)
        else:
            text = self.tokenizer.decode(self.tokenizer(text).input_ids[:self.max_len])

        image = Image.open(os.path.join(self.image_path, os.path.basename(self.questions[i]["image"]))) 
        ret = self.tokenizer.processor(
            text=text,
                images=image,
                padding=True,
                truncation=True,
                return_tensors="pt",
                # videos = None
        )
        for key in ret:
            if 'qwen' in self.model_type and key == 'pixel_values':
                continue
            ret[key] = ret[key][0]
        return ret
    
    def covert_conversations(self, data):
        new_data = []
        for d in data:
            content = d["value"]
            for old, new in self.template.replace_tokens:
                content = content.replace(old, new)
            new_data.append({
                "role": self.template.role_mapping.get(d["from"], d["from"]),
                "content": content
            })
        return new_data


if __name__ == "__main__":
    # import json
    # json_path = "/data4/zww/llava_v1_5_mix665k.json"
    # json_data = json.load(open(json_path, 'r')) 
    # breakpoint()
    print(TEMPLATES)