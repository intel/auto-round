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
"""
Support Matrix
| Model                 | calibration dataset | quant nontext module |
|-----------------------|---------------------|----------------------|
| Qwen2-VL              | pile/llava          | -                    |
| Llama-3.2-Vision      | llava               | ✔                    |
| Phi3-Vision           | pile/llava          | ✔                    |
| Llava-v1.5            | pile/llava          | X                    |
| CogVLM2               | pile/llava          | ✔                    |
| gemma-3               | pile/llava          | -                    |
| granite-vision-3.2    | pile/llava          | -                    |
| Mistral-Small-3.1     | pile/llava          | X                    |
| Aria                  | pile/llava          | -                    |

✔ means support, - means support but cannot infer or not test infert yet, X means not support.
"""
import os
from datetime import datetime, timedelta

import torch
from transformers.data.data_collator import default_data_collator

from .utils import fetch_image

PROCESSORS = {}


def register_processor(name):
    def register(processor):
        PROCESSORS[name] = processor
        return processor

    return register


@register_processor("basic")
class BasicProcessor:
    def __init__(self):
        pass

    def post_init(self, model, tokenizer, processor=None, image_processor=None, use_rtn=False, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        if image_processor is not None:
            self.image_processor = image_processor
        else:
            self.image_processor = self.default_image_processor
        self.use_rtn = use_rtn
        self.check_image_processor()

    def get_input(self, text, images, squeeze=True, **kwargs):
        raise NotImplementedError

    @staticmethod
    def data_collator(batch):
        return default_data_collator(batch)

    @staticmethod
    def default_image_processor(image_path_or_url):
        return fetch_image(image_path_or_url)

    @staticmethod
    def squeeze_result(ret):
        for key in ret:
            ret[key] = ret[key][0]
        return ret

    def check_image_processor(self):
        if not self.use_rtn and self.image_processor is None:
            raise ValueError("image processor should not be None.")


@register_processor("hf")
class HFProcessor(BasicProcessor):
    # evaluation on: Qwen2-VL, mllama, Mistral-Small
    IMAGE_TOKEN = "<image>"

    def __init__(self):
        self.process_func = self._process_v1

    def post_init(self, model, tokenizer, processor=None, image_processor=None, use_rtn=False, **kwargs):
        assert tokenizer is not None, "tokenizer should not be None"
        assert processor is not None, "processor should not be None"
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        if image_processor is not None:
            self.image_processor = image_processor
        else:
            self.image_processor = self.default_image_processor
        self.use_rtn = use_rtn
        self.check_image_processor()

    def _process_v1(self, messages, image):
        """support models: Qwen2-VL, gemma-3, granite-vision-3.2, Aria"""
        conversation = []
        for content in messages:
            conversation.append(
                {
                    "role": content["role"],
                    "content": [{"text": content["content"].replace(self.IMAGE_TOKEN, ""), "type": "text"}],
                }
            )
            if self.IMAGE_TOKEN in content["content"]:
                conversation[-1]["content"].append({"image": image, "type": "image"})
        ret = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True
        )
        return ret

    def _process_v2(self, messages, image):
        """support model: Mistral-Small-3.1, phi3_v"""
        conversation = []
        for content in messages:
            if content["role"] == "user":
                conversation.append(
                    {
                        "role": content["role"],
                        "content": [{"text": content["content"].replace(self.IMAGE_TOKEN, ""), "type": "text"}],
                    }
                )
                if self.IMAGE_TOKEN in content["content"]:
                    conversation[-1]["content"].append({"image": image, "type": "image"})
            else:
                conversation.append({"role": content["role"], "content": content["content"]})
        if hasattr(self.processor, "chat_template"):
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False, return_dict=False
            )
        else:
            continue_final_message = messages[-1]["role"] == "assistant"
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
            )
        if image is not None:
            image = self.default_image_processor(image)
            # image = self.image_processor(image)
        ret = self.processor(text=text, images=image, return_tensors="pt")
        return ret

    def get_input(
        self,
        text,
        images,
        return_tensors="pt",
        squeeze=True,
        max_length=None,
        truncation=False,
        truncation_strategy="text",
        **kwargs
    ):

        if isinstance(text, list):
            try:
                ret = self.process_func(text, images)
            except Exception:
                self.process_func = self._process_v2
                ret = self.process_func(text, images)
        else:
            text = self.tokenizer.decode(self.tokenizer(text).input_ids[:max_length])

            if images is not None:
                images = self.image_processor(images)
            ret = self.processor(text=text, images=images, return_tensors="pt", add_special_tokens=False)

        if squeeze:
            ret = self.squeeze_result(ret)
        return ret


@register_processor("qwen2_vl")
class Qwen2VLProcessor(HFProcessor):
    @staticmethod
    def squeeze_result(ret):
        for key in ret:
            if key == "pixel_values":
                continue
            ret[key] = ret[key][0]
        return ret


@register_processor("cogvlm2")
class CogVLM2Processor(BasicProcessor):
    def get_input(self, text, images, truncation=False, squeeze=True, max_length=None, **kwargs):

        if images is not None:
            images = self.image_processor(images)

        padding_len = 2303
        max_length = 0 if max_length is None else max_length
        max_length += padding_len
        padding = False
        input_data = self.model.build_conversation_input_ids(
            self.tokenizer, query=text, history=None, images=[images], template_version="base"
        )

        def pad_to_len(unpadded_tensor, pad_to_length, pad_value=0):
            current_length = len(unpadded_tensor)
            if current_length >= pad_to_length:
                if truncation:
                    return unpadded_tensor[:pad_to_length]
                else:
                    return unpadded_tensor
            if padding:
                return torch.cat(
                    (
                        unpadded_tensor,
                        torch.full(
                            [pad_to_length - current_length],
                            fill_value=pad_value,
                            dtype=unpadded_tensor.dtype,
                            device=unpadded_tensor.device,
                        ),
                    ),
                    dim=0,
                )
            else:
                return unpadded_tensor

        input_data["input_ids"] = pad_to_len(
            input_data["input_ids"],
            max_length,
            pad_value=128002,
        )
        input_data["attention_mask"] = pad_to_len(input_data["attention_mask"], max_length, pad_value=0)
        input_data["token_type_ids"] = pad_to_len(input_data["token_type_ids"], max_length, pad_value=0)
        if input_data["labels"]:
            input_data["labels"] = pad_to_len(input_data["labels"], max_length, pad_value=-100)
        return input_data

    @staticmethod
    def data_collator(batch):
        batched_data = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], list):
                batched_data[key] = [batch_item[key] for batch_item in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                batched_data[key] = torch.stack([item[key] for item in batch])
            # else:
            #     raise ValueError("Unsupported datatype in custom collate_fn")
        return batched_data

    @staticmethod
    def default_image_processor(image_path_or_url):
        return fetch_image(image_path_or_url).convert("RGB")


from auto_round.utils import LazyImport

llava_train = LazyImport("llava.train.train")


@register_processor("llava")
class LlavaProcessor(BasicProcessor):
    def post_init(self, model, tokenizer, image_processor=None, use_rtn=False, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.collator_func = llava_train.DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        self.check_image_processor()

    def get_input(
        self, text, images, max_length=None, squeeze=True, truncation=False, truncation_strategy="text", **kwargs
    ):

        if images is not None:
            images = fetch_image(images).convert("RGB")
            images = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"][0]

        class DataArgs:
            is_multimodal = True
            mm_use_im_start_end = False

        if truncation is True and truncation_strategy == "text":
            text = self.tokenizer.decode(self.tokenizer(text).input_ids[:max_length])

        input_data = llava_train.preprocess_multimodal([text], DataArgs())
        ret = llava_train.preprocess(input_data, self.tokenizer, has_image=(images is not None))

        if truncation is True and truncation_strategy == "token":
            seqlen = ret["input_ids"].shape[-1]
            for key in ret:
                if ret[key].shape[-1] == seqlen:
                    ret[key] = ret[key][:, :max_length]
        if squeeze:
            ret = self.squeeze_result(ret)
        ret["image"] = images

        return ret

    def data_collator(self, batch):
        return self.collator_func(batch)


@register_processor("deepseek_v2")
class DeepSeekV2Processor(BasicProcessor):
    IMAGE_TOKEN = "<image>"

    def get_input(
        self, text, images, max_length=None, squeeze=True, truncation=False, truncation_strategy="text", **kwargs
    ):

        messages = []
        for content in text:
            if content["role"] == "user":
                messages.append({"role": content["role"], "content": content["content"]})
                if self.IMAGE_TOKEN in content["content"]:
                    messages[-1]["images"] = [images]
            else:
                messages.append({"role": content["role"], "content": content["content"]})

        if images is not None:
            pil_image = [self.image_processor(images)]
        else:
            pil_image = None

        prepare_inputs = self.processor(
            conversations=messages, images=pil_image, force_batchify=True, system_prompt=""
        ).to(self.model.device)
        prepare_inputs = prepare_inputs.to(self.model.device)
        prepare_inputs = self.squeeze_result(dict(prepare_inputs))
        return prepare_inputs


@register_processor("mistral3_2")
class Mistral3Processor(BasicProcessor):
    IMAGE_TOKEN = "<image>"

    @staticmethod
    def load_system_prompt(repo_id_or_path: str, filename: str) -> str:
        from huggingface_hub import hf_hub_download

        if os.path.isdir(repo_id_or_path):
            file_path = os.path.join(repo_id_or_path, filename)
        else:
            file_path = hf_hub_download(repo_id=repo_id_or_path, filename=filename)
        with open(file_path, "r") as file:
            system_prompt = file.read()
        today = datetime.today().strftime("%Y-%m-%d")
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        model_name = repo_id_or_path.split("/")[-1]
        return system_prompt.format(name=model_name, today=today, yesterday=yesterday)

    def get_input(
        self,
        text,
        images,
        return_tensors="pt",
        squeeze=True,
        max_length=None,
        truncation=False,
        truncation_strategy="text",
        **kwargs
    ):
        from mistral_common.protocol.instruct.request import ChatCompletionRequest  # pylint: disable=E0401

        SYSTEM_PROMPT = self.load_system_prompt(self.model.name_or_path, "SYSTEM_PROMPT.txt")

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        for content in text:
            conversation.append(
                {
                    "role": content["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": content["content"].replace(self.IMAGE_TOKEN, ""),
                        },
                    ],
                },
            )
            if self.IMAGE_TOKEN in content["content"]:
                conversation[-1]["content"].append({"type": "image_url", "image_url": {"url": images}})
        tokenized = self.tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=conversation, continue_final_message=True)
        )
        input_ids = torch.tensor([tokenized.tokens])
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.tensor(tokenized.images[0], dtype=torch.bfloat16).unsqueeze(0)
        image_sizes = torch.tensor([pixel_values.shape[-2:]])

        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        }
        if squeeze:
            ret = self.squeeze_result(ret)
        return ret
