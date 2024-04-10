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
def get_pile_dataset(tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", split=None, seed=42, bs=4):
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

    split = "train"
    tokenizer_function = get_tokenizer_function(tokenizer, seqlen)

    calib_dataset = load_dataset(dataset_name, split=split)
    calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset("madao33/new-title-chinese")
def get_new_chinese_title_dataset(
    tokenizer, seqlen, dataset_name="madao33/new-title-chinese", split=None, seed=42, bs=4
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

    def get_tokenizer_function(tokenizer, seqlen):
        """Returns a default tokenizer function.

        Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.

        Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
        of seqlen to the "text" field of examples.
        """

        def default_tokenizer_function(examples):
            example = tokenizer(examples["content"], truncation=True, max_length=seqlen)
            return example

        return default_tokenizer_function

    split = "train"
    from datasets import load_dataset

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen)

    calib_dataset = load_dataset(dataset_name, split=split)
    calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset("mbpp")
def get_mbpp_dataset(tokenizer, seqlen, dataset_name="mbpp", split=None, seed=42, bs=4):
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

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen)

    samples = []
    splits = split
    if splits is None:
        splits = ["train", "validation", "test"]
    if isinstance(splits, str):
        splits = [split]

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
def get_local_dataset(tokenizer, seqlen, dataset_name="./tmp.json", split=None, seed=42, bs=4):
    """Returns a dataloader for a custom dataset and split.
    We allow the input of a json or text file containing a processed text sample each line.

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
    tokenizer_function = get_tokenizer_function(tokenizer, seqlen)

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


def get_dataloader(tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", split=None, seed=42, bs=4, n_samples=512):
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
        n_samples (int, optional): The total number of samples to include. Defaults to 512.

    Returns:
        DataLoader: The DataLoader for the calibrated dataset.
    """

    dataset_names = dataset_name.split(",")

    # if len(dataset_names) == 1 and dataset_names[0] == "NeelNanda/pile-10k":  ##to guarantee the reproducibility.
    #     get_dataloader = CALIB_DATASETS.get("legacy-NeelNanda/pile-10k")
    #     dataloader = get_dataloader(
    #         tokenizer,
    #         seqlen,
    #         seed=seed,
    #         bs=bs,
    #         split=split,
    #         dataset_name=dataset_name,
    #     )
    #     return dataloader
    def filter_func(example):
        if example["input_ids"].shape[-1] < seqlen:
            return False
        input_ids = example["input_ids"][:seqlen]
        input_ids_list = input_ids.tolist()
        if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
            return False
        return True

    datasets = []
    for name in dataset_names:
        if is_local_path(name):
            get_dataset = CALIB_DATASETS.get("local")
        else:
            get_dataset = CALIB_DATASETS.get(name)
        dataset = get_dataset(
            tokenizer,
            seqlen,
            seed=seed,
            bs=bs,
            split=split,
            dataset_name=name,
        )
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataset = dataset.filter(filter_func)

        datasets.append(dataset)
    cnt = 0
    dataset_cnt_info = {}
    short_ds_len = [len(dataset) for dataset in datasets if len(dataset) < (n_samples // len(datasets))]
    for i in range(len(datasets)):
        if len(datasets[i]) > (n_samples // len(datasets)):
            target_cnt = (n_samples - cnt - sum(short_ds_len)) // (len(datasets) - i - len(short_ds_len))
            target_cnt = min(target_cnt, len(datasets[i]))
        else:
            target_cnt = len(datasets[i])
            short_ds_len = short_ds_len[1:]
        datasets[i] = datasets[i].select(range(target_cnt))  
        dataset_cnt_info[dataset_names[i]] = target_cnt
        cnt += target_cnt
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


# ##we keep this for reproduce issue, do not delete, keep it for a while
# @register_dataset("legacy-NeelNanda/pile-10k")
# def get_pile_dataloader(tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", split="train", seed=42, bs=4):
#     """Returns a dataloader for the specified dataset and split.
#
#     Args:
#     tokenizer: The tokenizer to be used for tokenization.
#     seqlen: The maximum sequence length.
#     data_name: The name of the dataset.
#     split: The data split to be used (e.g., "train", "test").
#     seed: The random seed for shuffling the dataset.
#     bs: The batch size for the dataloader.
#
#     Returns:
#     A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
#     """
#     from datasets import load_dataset
#     from torch.utils.data import DataLoader
#
#     tokenizer_function = get_tokenizer_function(tokenizer, seqlen)
#
#     @torch.no_grad()
#     def collate_batch(batch):
#         input_ids_new = []
#         for text in batch:
#             input_ids = text["input_ids"]
#             if input_ids.shape[0] < seqlen:
#                 continue
#             input_ids = input_ids[:seqlen]
#             input_ids_list = input_ids.tolist()
#             if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
#                 continue
#             input_ids_new.append(input_ids)
#         if len(input_ids_new) == 0:
#             return None
#         tmp = torch.vstack(input_ids_new)
#         res = {"input_ids": tmp}
#         return res
#
#     calib_dataset = load_dataset(dataset_name, split=split)
#     calib_dataset = calib_dataset.shuffle(seed=seed)
#     calib_dataset = calib_dataset.map(tokenizer_function, batched=True)
#     calib_dataset.set_format(type="torch", columns=["input_ids"])
#     calib_dataloader = DataLoader(calib_dataset, batch_size=bs, shuffle=False, collate_fn=collate_batch)
#     return calib_dataloader
