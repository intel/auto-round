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
import random

import torch
from torch.utils.data import DataLoader

from .utils import is_local_path, logger

CALIB_DATASETS = {}


def register_dataset(name):
    """Class decorator to register a DATASET subclass to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        name: A string. Define the dataset type.

    Returns:
        cls: The class of register.
    """

    def register(dataset):
        CALIB_DATASETS[name] = dataset
        return dataset

    return register


def get_tokenizer_function(tokenizer, seqlen, apply_template=False):
    """Returns a default tokenizer function.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    apply_template: Whether to apply chat template in tokenization.

    Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length of
    seqlen to the "text" field of examples.
    """

    def default_tokenizer_function(examples, apply_template=apply_template):
        if not apply_template:
            example = tokenizer(examples["text"], truncation=True, max_length=seqlen)
        else:
            from jinja2 import Template  # pylint: disable=E0401
            chat_template = tokenizer.chat_template if tokenizer.chat_template is not None \
                else tokenizer.default_chat_template
            template = Template(chat_template)
            rendered_messages = []
            for text in examples["text"]:
                message = [{"role": "user", "content": text}]
                rendered_message = template.render(messages=message, add_generation_prompt=True, \
                                                   bos_token=tokenizer.bos_token)
                rendered_messages.append(rendered_message)
            example = tokenizer(rendered_messages, truncation=True, max_length=seqlen)
        return example

    return default_tokenizer_function


@register_dataset("NeelNanda/pile-10k")
def get_pile_dataset(tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", split=None, seed=42, apply_template=False):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    split = "train"
    tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_template=apply_template)

    calib_dataset = load_dataset(dataset_name, split=split)
    calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset("madao33/new-title-chinese")
def get_new_chinese_title_dataset(
        tokenizer,
        seqlen,
        dataset_name="madao33/new-title-chinese",
        split=None,
        seed=42,
        apply_template=False
):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """

    def get_tokenizer_function(tokenizer, seqlen, apply_template=apply_template):
        """Returns a default tokenizer function.

        Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.
        apply_template: Whether to apply chat template in tokenization.

        Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
        of seqlen to the "text" field of examples.
        """

        def default_tokenizer_function(examples, apply_template=apply_template):
            if not apply_template:
                example = tokenizer(examples["content"], truncation=True, max_length=seqlen)
            else:
                from jinja2 import Template
                chat_template = tokenizer.chat_template if tokenizer.chat_template is not None \
                    else tokenizer.default_chat_template
                template = Template(chat_template)
                rendered_messages = []
                for text in examples["text"]:
                    message = [{"role": "user", "content": text}]
                    rendered_message = template.render(messages=message, add_generation_prompt=True, \
                                                       bos_token=tokenizer.bos_token)
                    rendered_messages.append(rendered_message)
                example = tokenizer(rendered_messages, truncation=True, max_length=seqlen)
            return example

        return default_tokenizer_function

    split = "train"
    from datasets import load_dataset

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_template=apply_template)

    calib_dataset = load_dataset(dataset_name, split=split)
    calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset("mbpp")
def get_mbpp_dataset(tokenizer, seqlen, dataset_name="mbpp", split=None, seed=42, apply_template=False):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_template=apply_template)

    samples = []
    splits = split
    if splits is None:
        splits = ["train", "validation", "test"]
    if isinstance(splits, str):
        splits = splits.split("+")

    for split in splits:
        dataset = load_dataset(dataset_name, split=split)
        for data in dataset:
            samples.append({"text": data["text"] + data["code"]})
    random.Random(seed).shuffle(samples)
    import datasets

    calib_dataset = datasets.Dataset.from_list(samples)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset("local")
def get_local_dataset(tokenizer, seqlen, dataset_name="./tmp.json", split=None, seed=42, apply_template=False):
    """Returns a dataloader for a custom dataset and split.
    We allow the input of a json or text file containing a processed text sample each line.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name or path of the dataset, which is a jsonl file.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for a custom dataset and split, using the provided tokenizer and sequence length.
    """
    tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_template=apply_template)

    def load_local_data(data_path):
        if data_path.endswith(".json"):
            with open(data_path, "r") as f:
                data = json.load(f)
            return data
        elif data_path.endswith(".txt"):
            with open(data_path) as f:
                data = [line for line in f]
            return data
        else:
            logger.error("invalid local file type,for now only support json ")

    samples = []
    dataset = load_local_data(dataset_name)
    if isinstance(dataset, dict):
        new_dataset = []
        for key in dataset.keys():
            new_dataset.append(dataset[key])
        dataset = new_dataset
    for data in dataset:
        text = data
        if isinstance(text, str):
            pass
        elif isinstance(data, dict) and len(data.keys()) == 1:
            for item in data.items():
                text = item[1]
        elif isinstance(data, dict) and "text" in data.keys():
            text = data["text"]
        elif isinstance(data, dict) and "input_ids" in data.keys():
            text = data["input_ids"]
        assert isinstance(text, str), "data must be string"
        text = text.rstrip()
        text = text.rstrip("\n")
        samples.append({"text": text})
    random.Random(seed).shuffle(samples)
    import datasets

    calib_dataset = datasets.Dataset.from_list(samples)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)
    return calib_dataset


def get_dataloader(
        tokenizer,
        seqlen,
        dataset_name="NeelNanda/pile-10k",
        seed=42,
        bs=8,
        nsamples=512,
):
    """Generate a DataLoader for calibration using specified parameters.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for tokenization.
        seqlen (int): The exact sequence length. samples < seqlen will be dropped,
                      samples longer than seqlen will be truncated
        dataset_name (str, optional): The name of the dataset or datasets separated by commas.
                                     Defaults to "NeelNanda/pile-10k".
        split (str, optional): The data split to use. Defaults to None.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        bs (int, optional): The batch size. Defaults to 4.
        nsamples (int, optional): The total number of samples to include. Defaults to 512.
        apply_template: Whether to apply chat template in tokenization.

    Returns:
        DataLoader: The DataLoader for the calibrated dataset.
    """
    data = Dataset(model_name="/models/Mixtral-8x7B-Instruct-v0.1",
                   dataset_path="/data6/xinhe/auto-round/mixtral_15k_calibration_v4.pkl", pad_inputs=True)
    return data

    dataset_names = dataset_name.split(",")

    def filter_func(example):
        if isinstance(example["input_ids"], list):
            example["input_ids"] = torch.tensor(example["input_ids"])
        if example["input_ids"].shape[-1] < seqlen:
            return False
        input_ids = example["input_ids"][:seqlen]
        input_ids_list = input_ids.tolist()
        if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
            return False
        return True

    def concat_dataset_element(dataset):
        input_ids, concat_input_ids = [eg['input_ids'] for eg in dataset], []
        attention_mask_list, attention_mask = [], torch.ones([1, seqlen]).to(torch.int64)
        buffer_input_id = torch.Tensor().to(torch.int64)
        bos_token_id, eos_token_id = tokenizer.bos_token_id, tokenizer.eos_token_id
        os_cnt, have_bos, have_eos = 0, False, False

        for input_id in input_ids:
            if input_id[0] == bos_token_id:
                input_id = input_id[1:]
                os_cnt, have_bos = os_cnt + 1, True
            if input_id[-1] == eos_token_id:
                input_id = input_id[:-1]
                os_cnt, have_eos = os_cnt + 1, True

            if buffer_input_id.shape[-1] + input_id.shape[-1] + os_cnt > seqlen:
                idx_keep = seqlen - buffer_input_id.shape[-1] - os_cnt
                input_id_to_append = [buffer_input_id, input_id[:idx_keep]]
                if have_bos:
                    input_id_to_append = [torch.tensor([bos_token_id])] + input_id_to_append
                if have_eos:
                    input_id_to_append.append(torch.tensor([eos_token_id]))

                concat_input_ids.append(torch.cat(input_id_to_append).to(torch.int64))
                attention_mask_list.append(attention_mask)
                buffer_input_id = input_id[idx_keep:]
            else:
                buffer_input_id = torch.cat([buffer_input_id, input_id])

            if buffer_input_id.shape[-1] + os_cnt == seqlen:
                input_id_to_append = [buffer_input_id]
                if have_bos:
                    input_id_to_append = [torch.tensor([bos_token_id])] + input_id_to_append
                if have_eos:
                    input_id_to_append.append(torch.tensor([eos_token_id]))
                concat_input_ids.append(torch.cat(input_id_to_append).to(torch.int64))
                attention_mask_list.append(attention_mask)
                buffer_input_id = torch.Tensor().to(torch.int64)
        data = [{'input_ids': a, 'attention_mask': b} for a, b in zip(concat_input_ids, attention_mask_list)]
        import datasets
        dataset_new = datasets.Dataset.from_list(data)
        return dataset_new

    datasets, data_lens = [], {}
    for name in dataset_names:
        split = None
        do_concat = False
        apply_template = False
        if ":" in name:
            split_list = name.split(":")
            name, split_list = name.split(":")[0], name.split(":")[1:]
            for ele in split_list:
                key, values = ele.split('=')[0], ele.split('=')[1:]
                if key == "split":
                    split = values[0].split('+')
                if key == "num":
                    data_lens[name] = int(values[0])
                if key == "concat":
                    do_concat = False if (len(values) > 0 and values[0].lower() == 'false') else True
                if key == "apply_template":
                    apply_template = False if (len(values) > 0 and values[0].lower() == 'false') else True
        if is_local_path(name):
            get_dataset = CALIB_DATASETS.get("local")
        else:
            get_dataset = CALIB_DATASETS.get(name)
        dataset = get_dataset(
            tokenizer,
            seqlen,
            seed=seed,
            split=split,
            dataset_name=name,
            apply_template=apply_template,
        )
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        if do_concat:
            dataset = concat_dataset_element(dataset)
        dataset = dataset.filter(filter_func)
        if name in data_lens:
            dataset = dataset.select(range(data_lens[name]))
        datasets.append(dataset)
    indices = range(len(datasets))
    res = sorted(zip(indices, datasets), key=lambda x: len(x[1]))
    indices = [item[0] for item in res]
    datasets = [item[1] for item in res]
    dataset_names = [dataset_names[index] for index in indices]
    cnt = 0 if not data_lens else sum(data_lens.values())
    dataset_cnt_info = {}
    if cnt > nsamples:
        cnt = 0

    for i in range(len(datasets)):
        name = dataset_names[i].split(':')[0]
        if name not in data_lens:
            target_cnt = (nsamples - cnt) // (len(datasets) - len(data_lens)) if data_lens \
                else (nsamples - cnt) // (len(datasets) - i)
            target_cnt = min(target_cnt, len(datasets[i]))
            cnt += target_cnt
        else:
            target_cnt = data_lens[name]
        datasets[i] = datasets[i].select(range(target_cnt))
        dataset_cnt_info[name] = target_cnt
    if len(datasets) > 1:
        from datasets import concatenate_datasets

        dataset_final = concatenate_datasets(datasets)
        dataset_final = dataset_final.shuffle(seed=seed)
        logger.info(dataset_cnt_info)
    else:
        dataset_final = datasets[0]

    @torch.no_grad()
    def collate_batch(batch):
        input_ids_new = []
        attention_mask_new = []
        for text in batch:
            input_ids, attention_mask = text["input_ids"], text["attention_mask"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            if isinstance(attention_mask, list):
                attention_mask = torch.tensor(attention_mask)
            input_ids = input_ids[:seqlen]
            input_ids_list = input_ids.tolist()
            if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
                continue
            attention_mask = attention_mask[:seqlen]
            attention_mask_new.append(attention_mask)
            input_ids_new.append(input_ids)
        if len(input_ids_new) == 0:
            return None
        input_ids_new = torch.vstack(input_ids_new)
        attention_mask_new = torch.vstack(attention_mask_new)
        res = {"input_ids": input_ids_new, "attention_mask": attention_mask_new}
        return res

    calib_dataloader = DataLoader(dataset_final, batch_size=bs, shuffle=False, collate_fn=collate_batch)
    return calib_dataloader


import os
from transformers import AutoTokenizer
import pandas as pd
import torch
import numpy as np
import pickle

import torch.nn.functional as F

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("MoE-Dataset")


class Dataset():
    def __init__(self, model_name=None, total_sample_count=15000, perf_count_override=None, dataset_path=None,
                 device="cpu", pad_inputs=False):
        self.model_name = model_name or "mixtral/Mixtral-8x7B-Instruct-v0.1"
        self.dataset_path = dataset_path
        self.max_length = 2048
        self.device = device
        self.pad_inputs = pad_inputs

        self.load_tokenizer()
        self.load_processed_dataset()
        self.total_sample_count = min(len(self.input_ids), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.max_length,
            padding_side="right",
            use_fast=False, )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_processed_dataset(self):
        if not os.path.isfile(self.dataset_path):
            log.warn(
                "Processed pickle file {} not found. Please check that the path is correct".format(self.dataset_path))

        processed_data = pd.read_pickle(self.dataset_path)
        input_tokens = processed_data['tok_input']

        log.info("mlperf seq 2048,right padding")

        self.input_ids = []
        self.input_lens = []
        self.attention_masks = []
        self.dataset = []
        for i, ids in enumerate(input_tokens):
            input_ids = torch.tensor(ids, dtype=torch.int32).view(1, -1).to(self.device)
            input_len = input_ids.shape[-1]
            if input_len > self.max_length:
                input_ids = input_ids[:, :self.max_length]
                input_len = input_ids.shape[-1]

            attn_mask = torch.ones(input_len).view(1, input_len)
            if self.pad_inputs:
                pad_size = self.max_length - input_len
                pad_tup = (0, pad_size)
                input_ids = F.pad(input_ids, pad=pad_tup, value=self.tokenizer.pad_token_id)
                attn_mask = F.pad(attn_mask, pad=pad_tup)

            attn_mask[:, -1] = 0

            self.input_ids.append(input_ids)
            self.input_lens.append(input_ids.size(1))
            self.dataset.append({'input_ids': input_ids, 'attention_mask': attn_mask})
            # self.attention_masks.append(None)

        print("Finished loading dataset")

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def postProcess(self, out_tokens, input_seq_lens=None, query_id_list=None, sample_index_list=None, save=False):
        """ Postprocesses output prediction """
        # TODO: Create response object in postProcess(?)
        """
        preds = []
        for i in range(out_tokens.shape[0]):
            #pred = out_tokens[i].reshape(-1).cpu().numpy() # Slice up to original input length as below?

            input_len = input_seq_lens[i] if input_seq_lens else 0
            pred = out_tokens[i, input_len:].reshape(-1).cpu().numpy()
            preds.append(pred)
        """
        output_seqs = []

        for i, out_token in enumerate(out_tokens):
            output_seq = np.array(out_token).reshape(1, -1)
            output_seqs.append(output_seq.reshape(-1))

            if save:
                query_id = [query_id_list[i]]
                # Save outputs
                if not os.path.exists("run_outputs"):
                    os.makedirs("run_outputs")
                fname = "q" + "_".join([str(i) for i in query_id])
                fname = f"run_outputs/{fname}.pkl"
                with open(fname, mode='wb') as f:
                    d = {"query_ids": query_id,
                         "outputs": output_seq}
                    print(f"Saving outputs to {fname}")
                    pickle.dump(d, f)

        return output_seqs

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesToRam(self, sample_list):
        pass

    def __del__(self):
        pass