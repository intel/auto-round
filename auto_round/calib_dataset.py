import json
import random
import torch
from utils import logger
CALIB_DATASETS = {}


def register_dataset(name):
    """Class decorator to register a DATASET subclass to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        cls (class): The subclass of register.
        name: A string. Define the dataset type.

    Returns:
        cls: The class of register.
    """

    def register(dataset):
        CALIB_DATASETS[name] = dataset
        return dataset

    return register


def get_tokenizer_function(tokenizer, seqlen):
    """Returns a default tokenizer function.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.

    Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length of
    seqlen to the "text" field of examples.
    """

    def default_tokenizer_function(examples):
        example = tokenizer(examples["text"], truncation=True, max_length=seqlen)
        return example

    return default_tokenizer_function


@register_dataset("NeelNanda/pile-10k")
def get_dataloader(tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", split="train", seed=42, bs=4):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    bs: The batch size for the dataloader.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen)

    @torch.no_grad()
    def collate_batch(batch):
        input_ids_new = []
        for text in batch:
            input_ids = text["input_ids"]
            if input_ids.shape[0] < seqlen:
                continue
            input_ids = input_ids[:seqlen]
            input_ids_list = input_ids.tolist()
            if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
                continue
            input_ids_new.append(input_ids)
        if len(input_ids_new) == 0:
            return None
        tmp = torch.vstack(input_ids_new)
        res = {"input_ids": tmp}
        return res

    calib_dataset = load_dataset(dataset_name, split=split)
    calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)
    calib_dataset.set_format(type="torch", columns=["input_ids"])
    calib_dataloader = DataLoader(calib_dataset, batch_size=bs, shuffle=False, collate_fn=collate_batch)
    return calib_dataloader


@register_dataset("mbpp")
def get_mbpp_dataloader(
    tokenizer, seqlen, dataset_name="mbpp", split=["train", "validation", "test"], seed=42, bs=4
):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    bs: The batch size for the dataloader.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    def get_mbpp_tokenizer_function(tokenizer, seqlen):
        """Returns a default tokenizer function.

        Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.

        Returns: A default tokenizer function that applies the provided tokenizer with truncation and
        a maximum length of seqlen to the "text" field of examples.
        """

        def default_tokenizer_function(examples):
            example = tokenizer(examples, truncation=True, max_length=seqlen, return_tensors="pt")
            # example = tokenizer(examples, return_tensors="pt")
            return example

        return default_tokenizer_function

    tokenizer_function = get_mbpp_tokenizer_function(tokenizer, seqlen)

    @torch.no_grad()
    def collate_batch(batch):
        input_ids_new = []
        attention_mask_new = []
        for text in batch:
            token_text = tokenizer_function(text)
            input_ids, attention_mask = token_text["input_ids"], token_text["attention_mask"]
            if input_ids.shape[1] < seqlen:
                continue
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

    samples = []
    splits = split
    for split in splits:
        dataset = load_dataset(dataset_name, split=split)
        for data in dataset:
            samples.append(data["text"] + data["code"])
    random.Random(seed).shuffle(samples)

    calib_dataloader = DataLoader(samples, batch_size=bs, shuffle=False, collate_fn=collate_batch)
    return calib_dataloader


@register_dataset("local")
def get_custom_dataloader(
    tokenizer, seqlen, dataset_name="./tmp.json", split=None, seed=42, bs=4
):
    """Returns a dataloader for a custom dataset and split.
    We allow the input of a jsonl file containing a processed text sample each line.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name or path of the dataset, which is a jsonl file.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    bs: The batch size for the dataloader.

    Returns:
    A dataloader for a custom dataset and split, using the provided tokenizer and sequence length.
    """
    from torch.utils.data import DataLoader

    def get_custom_tokenizer_function(tokenizer, seqlen):
        """Returns a default tokenizer function.

        Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.

        Returns: A default tokenizer function that applies the provided tokenizer with truncation and
        a maximum length of seqlen to the "text" field of examples.
        """

        def default_tokenizer_function(examples):
            example = tokenizer(examples, truncation=True, max_length=seqlen, return_tensors="pt")
            return example

        return default_tokenizer_function

    tokenizer_function = get_custom_tokenizer_function(tokenizer, seqlen)

    @torch.no_grad()
    def collate_batch(batch):
        input_ids_new = []
        attention_mask_new = []
        for text in batch:
            token_text = tokenizer_function(text)
            input_ids, attention_mask = token_text["input_ids"], token_text["attention_mask"]
            if input_ids.shape[1] < seqlen:
                continue
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

    def load_local_data(data_path):
        data = []
        if data_path.endswith("json"):
            with open(data_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        else:
            logger.error(f"invalid local file type,for now only support json ")

    samples = []
    dataset = load_local_data(dataset_name)
    for data in dataset:
        samples.append(data["text"])
    random.Random(seed).shuffle(samples)

    calib_dataloader = DataLoader(samples, batch_size=bs, shuffle=False, collate_fn=collate_batch)
    return calib_dataloader
