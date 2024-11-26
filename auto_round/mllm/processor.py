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

import torch
from transformers.data.data_collator import default_data_collator

from .utils import fetch_image

PROCESSORS = {}


def regist_processor(name):
    def register(processor):
        PROCESSORS[name] = processor
        return processor

    return register


@regist_processor("basic")
class BasicProcessor:
    def __init__(self):
        pass
    
    def post_init(self, model, tokenizer, processor=None, image_processor=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        if image_processor is not None:
            self.image_processor = image_processor
        else:
            self.image_processor = self.default_image_processor

    def get_input(
            self,
            text,
            images,
            return_tensors="pt",
            squeeze=True,
            max_length=None,
            truncation=False,
            truncation_strategy="text",
            **kwargs):

        if isinstance(text, list):
            for message in text:
                if not ("role" in message and "content" in message):
                    raise ValueError(
                        "When passing chat dicts as input, each dict must have a 'role' and 'content' key.")
            continue_final_message = text[-1]["role"] == "assistant"

            # add_generation_prompt=True will add <|im_start|>assistant to the end
            try:
                text = self.tokenizer.apply_chat_template(
                    text, tokenize=False, add_generation_prompt=not continue_final_message,
                    continue_final_message=continue_final_message, )
            except:
                raise NotImplementedError("current not support for non-instructed model.")

            if text == '':
                raise NotImplementedError("current not support for non-instructed model.")

        if images is not None:
            images = self.image_processor(images)

        if truncation is True and truncation_strategy == "text":
            text = self.tokenizer.decode(self.tokenizer(text).input_ids[:max_length])

        ret = self.processor(
            text=text,
            images=images,
            return_tensors=return_tensors,
            # videos = None
        )
        if truncation is True and truncation_strategy == "token":
            seqlen = ret['input_ids'].shape[-1]
            for key in ret:
                shape_ = ret[key].shape
                if len(shape_) == 2 and shape_[-1] == seqlen:
                    ret[key] = ret[key][:, :max_length]
                elif len(shape_) == 4 and shape_[1] == seqlen:
                    ret[key] = ret[key][:, :max_length]

        if squeeze:
            ret = self.squeeze_result(ret)
        return ret

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


@regist_processor("qwen2_vl")
class Qwen2VLProcessor(BasicProcessor):
    @staticmethod
    def squeeze_result(ret):
        for key in ret:
            if key == "pixel_values":
                continue
            ret[key] = ret[key][0]
        return ret


@regist_processor("cogvlm2")
class CogVLM2Processor(BasicProcessor):
    def get_input(
            self, text, images, truncation=False,
            squeeze=True, max_length=None, **kwargs):

        if images is not None:
            images = self.image_processor(images)
        
        padding_len = 2303
        max_length = 0 if max_length is None else max_length
        max_length += padding_len
        padding = False
        input_data = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=text,
            history=None,
            images=[images],
            template_version='base'
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
                    (unpadded_tensor,
                     torch.full([pad_to_length - current_length],
                                fill_value=pad_value,
                                dtype=unpadded_tensor.dtype,
                                device=unpadded_tensor.device)), dim=0)
            else:
                return unpadded_tensor

        input_data['input_ids'] = pad_to_len(
            input_data['input_ids'],
            max_length,
            pad_value=128002,
        )
        input_data['attention_mask'] = pad_to_len(
            input_data['attention_mask'],
            max_length,
            pad_value=0
        )
        input_data['token_type_ids'] = pad_to_len(
            input_data['token_type_ids'],
            max_length,
            pad_value=0
        )
        if input_data['labels']:
            input_data['labels'] = pad_to_len(
                input_data['labels'],
                max_length,
                pad_value=-100
            )
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
        return fetch_image(image_path_or_url).convert('RGB')


from ..utils import LazyImport

llava_train = LazyImport("llava.train.train")


@regist_processor("llava")
class LlavaProcessor(BasicProcessor):
    def post_init(self, model, tokenizer, image_processor=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        assert image_processor is not None, "for llava model, image_processor should not be None"
        self.image_processor = image_processor
        self.collator_func = llava_train.DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

    def get_input(
            self, text, images, max_length=None,
            squeeze=True, truncation=False, truncation_strategy="text", **kwargs):

        if images is not None:
            images = fetch_image(images).convert('RGB')
            images = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values'][0]

        class DataArgs:
            is_multimodal = True
            mm_use_im_start_end = False

        if truncation is True and truncation_strategy == "text":
            text = self.tokenizer.decode(self.tokenizer(text).input_ids[:max_length])

        input_data = llava_train.preprocess_multimodal([text], DataArgs())
        ret = llava_train.preprocess(input_data, self.tokenizer, has_image=(images is not None))

        if truncation is True and truncation_strategy == "token":
            seqlen = ret['input_ids'].shape[-1]
            for key in ret:
                if ret[key].shape[-1] == seqlen:
                    ret[key] = ret[key][:, :max_length]
        if squeeze:
            ret = self.squeeze_result(ret)
        ret['image'] = images

        return ret

    def data_collator(self, batch):
        return self.collator_func(batch)
