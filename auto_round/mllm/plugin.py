import torch
from transformers.data.data_collator import default_data_collator

from PIL import Image

PLUGINS = {}

def regist_plugin(name):
    def register(plugin):
        PLUGINS[name] = plugin
        return plugin
    return register

@regist_plugin("basic")
class BasicPlugin:
    def __init__(self):
        pass

    def get_input(
            model,
            tokenizer,
            text,
            images,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=None,
            **kwargs):

        if max_length:
            token_length = len(tokenizer(text).input_ids)
            if token_length < max_length:
                if tokenizer.pad_token:
                    text += tokenizer.pad_token * (max_length - token_length)
            else:
                text = tokenizer.decode(tokenizer(text).input_ids[:max_length])

        ret = tokenizer.processor(
            text=text,
            images=images,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            # videos = None
        )
        for key in ret:
            ret[key] = ret[key][0]
        return ret

    @staticmethod
    def data_collator(batch):
        return default_data_collator(batch)

    @staticmethod 
    def image_processor(image_path):
        return Image.open(image_path)

@regist_plugin("qwen2_vl")
class Qwen2VLPlugin(BasicPlugin):
    def get_input(
            model,
            tokenizer,
            text,
            images,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=None,
            **kwargs):

        if max_length:
            token_length = len(tokenizer(text).input_ids)
            if token_length < max_length:
                if tokenizer.pad_token:
                    text += tokenizer.pad_token * (max_length - token_length)
            else:
                text = tokenizer.decode(tokenizer(text).input_ids[:max_length])

        ret = tokenizer.processor(
            text=text,
            images=images,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            # videos = None
        )
        for key in ret:
            if key == "pixel_values":
                continue
            ret[key] = ret[key][0]
        return ret


@regist_plugin("cogvlm2")
class CogVLM2Plugin(BasicPlugin):
    def get_input(
            model, tokenizer, text, images, max_length=2048, 
            padding=True, truncation=True, **kwargs):
        breakpoint()
        padding_len = 2303
        max_length += padding_len
        input_data = model.build_conversation_input_ids(
                tokenizer,
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
        input_data = {
            'input_ids': input_data['input_ids'].unsqueeze(0),
            'token_type_ids': input_data['token_type_ids'].unsqueeze(0),
            'attention_mask': input_data['attention_mask'].unsqueeze(0),
            'images': [[input_data['images'][0]]] if input_data['images'] is not None else None,
        }
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
    def image_processor(image_path):
        return Image.open(image_path).convert('RGB')