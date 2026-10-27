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

import itertools
import json
import logging
import multiprocessing
import os
import queue
import random
import ssl
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version

logging.getLogger("datasets").setLevel(logging.WARNING)

import torch
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    IterableDataset,
    IterableDatasetDict,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from packaging.version import Version
from torch.utils.data import DataLoader

from . import envs
from .utils import is_local_path, logger
from .utils.dataset_utils import (
    CalibDataset,
    auto_detect_text_field,
    auto_detect_text_field_from_sample,
    extract_text_from_sample,
    normalize_dataset_spec,
    split_unescaped,
    unescape_spec_value,
)

CALIB_DATASETS = {}
_GITHUB_CODE_CLEAN_MAX_DATASETS_VERSION = Version("3.6.0")
_FINEWEB_EDU_HF_DATASET = "HuggingFaceFW/fineweb-edu"
_FINEWEB_EDU_MODELSCOPE_DATASET = "AI-ModelScope/fineweb-edu"
_FINEWEB_EDU_CONFIG = "sample-10BT"
_FINEWEB_EDU_MIN_CANDIDATES = 10000
_DATASET_RESULT_SUCCESS = "success"
_DATASET_RESULT_ERROR = "error"


def _get_dataset_network_error(error):
    """Return the network failure in an exception chain, if present."""
    visited = set()
    current = error
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        error_name = type(current).__name__.lower()
        error_module = type(current).__module__.lower()
        error_message = str(current).lower()
        if isinstance(current, (ConnectionError, TimeoutError, ssl.SSLError)):
            return current
        if any(term in error_name for term in ("connection", "http", "proxy", "ssl", "timeout")) and any(
            package in error_module
            for package in ("datasets", "huggingface_hub", "httpcore", "httpx", "requests", "urllib3")
        ):
            return current
        if any(
            marker in error_message
            for marker in (
                "cannot send a request, as the client has been closed",
                "connection timed out",
                "max retries exceeded",
                "proxy error",
                "ssl error",
            )
        ):
            return current
        current = current.__cause__ or current.__context__
    return None


def _fallback_to_fineweb_edu(error, tokenizer, seqlen, dataset_name, seed, nsamples, trust_remote_code=False):
    """Switch to ModelScope FineWeb-Edu after a calibration dataset network failure."""
    network_error = _get_dataset_network_error(error)
    if network_error is None:
        raise error
    try:
        version("modelscope")
    except PackageNotFoundError:
        raise RuntimeError(
            f"Failed to load calibration dataset {dataset_name!r} because of an HTTP/proxy network issue: "
            f"{network_error}. Install ModelScope with `pip install modelscope` to enable automatic fallback "
            "to FineWeb-Edu."
        ) from error

    if dataset_name == _FINEWEB_EDU_MODELSCOPE_DATASET:
        raise error

    logger.warning(
        "Failed to load calibration dataset %r because of an HTTP/proxy network issue: %s. "
        "Automatically switching to the ModelScope FineWeb-Edu mirror (%s).",
        dataset_name,
        network_error,
        _FINEWEB_EDU_MODELSCOPE_DATASET,
    )
    return _get_dataset_impl(tokenizer, seqlen, _FINEWEB_EDU_MODELSCOPE_DATASET, seed, nsamples, trust_remote_code)


def _preprocess_dataset_in_subprocess(
    result_queue, tokenizer, seqlen, dataset_name, seed, nsamples, output_path, trust_remote_code=False
):
    """Run dataset preprocessing and report network failures to the parent process."""
    # The parent's OpenMP thread pool does not survive fork; keep torch single-threaded here.
    torch.set_num_threads(1)
    try:
        dataset = _get_dataset_impl(tokenizer, seqlen, dataset_name, seed, nsamples, trust_remote_code)
        dataset.save_to_disk(output_path)
    except Exception as error:
        network_error = _get_dataset_network_error(error)
        result_queue.put((_DATASET_RESULT_ERROR, str(network_error) if network_error is not None else None))
        raise
    result_queue.put((_DATASET_RESULT_SUCCESS, output_path))


def get_code_calibration_dataset(nsamples, datasets_version=None):
    """Build an exact-size code calibration mix compatible with datasets."""
    if datasets_version is None:
        import datasets

        datasets_version = datasets.__version__
    parsed_version = Version(str(datasets_version))
    sources = [("opencode-instruct:concat=true", 50), ("github-code-clean", 50)]
    if parsed_version > _GITHUB_CODE_CLEAN_MAX_DATASETS_VERSION:
        sources = [source for source in sources if source[0] != "github-code-clean"]
        logger.warning_once(
            "datasets %s does not support the script-based github-code-clean dataset; "
            "using OpenCodeInstruct only instead.",
            parsed_version,
        )

    weights = [weight for _, weight in sources]
    raw_counts = [nsamples * weight / sum(weights) for weight in weights]
    counts = [int(count) for count in raw_counts]
    remainder = nsamples - sum(counts)
    order = sorted(range(len(weights)), key=lambda index: (-(raw_counts[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return ",".join(f"{name}:num={count}" for (name, _), count in zip(sources, counts) if count)


def register_dataset(name):
    """Class decorator to register a DATASET subclass to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        name: A string. Define the dataset type.

    Returns:
        cls: The class of register.
    """

    def register(dataset):
        if isinstance(name, list):
            names = name
        else:
            names = [name]
        for global_name in names:
            CALIB_DATASETS[global_name] = dataset
        return dataset

    return register


def render_chat_sample(tokenizer, text, system_prompt=None, tools=None):
    """Render a single conversation to a string with the chat template.

    Args:
        tokenizer: The tokenizer providing ``apply_chat_template``.
        text: Either a plain user string or a list of chat messages (dicts with
            ``role``/``content``).
        system_prompt: Optional global system prompt. It is only prepended when
            the conversation does not already start with its own system turn.
        tools: Optional tool definitions for this conversation, passed to
            ``apply_chat_template(..., tools=...)``.

    Returns:
        The rendered chat string.
    """
    message = []
    is_full_conversation = isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict)

    # Respect a per-sample system prompt: if the conversation already starts
    # with its own system message, do not prepend the global system_prompt
    # (otherwise the rendered text would contain two system turns).
    has_own_system = is_full_conversation and text[0].get("role") == "system"
    if system_prompt is not None and system_prompt != "" and not has_own_system:
        message.append({"role": "system", "content": system_prompt})

    if is_full_conversation:
        message += text
    else:
        message.append({"role": "user", "content": text})

    sample_tools = tools or None

    # For full multi-turn conversations ending in an assistant reply we want
    # the whole trace as calibration text (including tool_calls / tool
    # responses), so we do not append a generation prompt.
    add_generation_prompt = not (is_full_conversation and message[-1].get("role") == "assistant")

    template_kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if sample_tools is not None:
        template_kwargs["tools"] = sample_tools

    try:
        return tokenizer.apply_chat_template(message, **template_kwargs)
    except Exception:
        logger.warning("Failed to apply chat template. removing the system role in chat history.")
        message_modified = [msg for msg in message if msg["role"] != "system"]
        try:
            return tokenizer.apply_chat_template(message_modified, **template_kwargs)
        except Exception:
            # Last resort: drop tools in case the template does not support them.
            template_kwargs.pop("tools", None)
            return tokenizer.apply_chat_template(message_modified, **template_kwargs)


def apply_chat_template_to_samples(samples, tokenizer, seqlen, system_prompt=None, tools=None):
    """Render samples with the tokenizer chat template.

    Args:
        samples: A list where each item is either a plain user string or a list
            of chat messages (dicts with ``role``/``content``).
        tokenizer: The tokenizer providing ``apply_chat_template``.
        seqlen: Maximum sequence length for tokenization.
        system_prompt: Optional system prompt prepended to each conversation.
        tools: Optional per-sample tool definitions aligned with ``samples``
            (a list the same length as ``samples``; each entry is the ``tools``
            list for that conversation, or ``None``). Passed to
            ``apply_chat_template(..., tools=...)`` so agent/tool-calling data
            keeps its tool schema. When the conversation already contains the
            final assistant turn (full multi-turn agent traces), the generation
            prompt is not appended.
    """
    rendered_messages = []
    # if system_prompt is None: ## remove system prompt as models like deepseek don't recommend using it
    #     system_prompt = "You are a helpful assistant."
    for idx, text in enumerate(samples):
        sample_tools = tools[idx] if (tools is not None and idx < len(tools)) else None
        rendered_messages.append(render_chat_sample(tokenizer, text, system_prompt, sample_tools))
    example = tokenizer(rendered_messages, truncation=True, max_length=seqlen)
    return example


def _make_map_fingerprint(dataset, tokenizer, seqlen, apply_chat_template, system_prompt, text_key="text"):
    """Compute a stable fingerprint for Dataset.map() calls.

    datasets uses dill to serialize the transform function for cache fingerprinting.
    HuggingFace tokenizer objects are not reliably serializable by dill, causing
    a random hash to be used each run — which breaks caching entirely.

    This function computes a deterministic fingerprint from stable string
    identifiers (tokenizer name, seqlen, etc.) so that caching works correctly
    and subsequent runs can load from disk instead of re-tokenizing in RAM.
    """
    import hashlib

    parts = [
        getattr(dataset, "_fingerprint", "no_fingerprint"),
        getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
        str(seqlen),
        str(apply_chat_template),
        str(system_prompt),
        text_key,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def get_tokenizer_function(tokenizer, seqlen, apply_chat_template=False, system_prompt=None):
    """Returns a default tokenizer function.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length of
    seqlen to the "text" field of examples.
    """

    def default_tokenizer_function(examples):
        if not apply_chat_template:
            example = tokenizer(examples["text"], truncation=True, max_length=seqlen)
        else:
            example = apply_chat_template_to_samples(examples["text"], tokenizer, seqlen, system_prompt)
        return example

    return default_tokenizer_function


@register_dataset(["nvidia/Nemotron-Pretraining-Dataset-sample", "nemotron-pretraining"])
def get_nemotron_pretraining_dataset(
    tokenizer,
    seqlen,
    dataset_name="nvidia/Nemotron-Pretraining-Dataset-sample",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
):
    """Calibration data from Nvidia's Nemotron pretraining corpus (public sample).

    Uses the Nemotron-CC-High-Quality config — filtered high-quality web text from
    the same distribution as Nemotron-H / Nemotron-Cascade-2 pretraining.
    """
    split = "train" if split is None else split
    if isinstance(split, list):
        if len(split) != 1:
            raise ValueError("Nemotron-Pretraining-Dataset-sample supports only one split at a time.")
        split = split[0]
    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )
    try:
        calib_dataset = load_dataset(
            dataset_name,
            name="Nemotron-CC-High-Quality",
            split=split,
            streaming=True,
        )
        calib_dataset = calib_dataset.shuffle(seed=seed).take(10000)
        calib_dataset = Dataset.from_list(list(calib_dataset))
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        sys.exit(1)
    calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )
    return calib_dataset


@register_dataset(["NeelNanda/pile-10k", "pile-10k"])
def get_pile_dataset(
    tokenizer,
    seqlen,
    dataset_name="NeelNanda/pile-10k",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    split = "train"

    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )
    calib_dataset = load_dataset("NeelNanda/pile-10k", split=split)
    calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )

    return calib_dataset


@register_dataset([_FINEWEB_EDU_HF_DATASET, _FINEWEB_EDU_MODELSCOPE_DATASET, "fineweb-edu"])
def get_fineweb_edu_dataset(
    tokenizer,
    seqlen,
    dataset_name="fineweb-edu",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
    max_samples=_FINEWEB_EDU_MIN_CANDIDATES,
):
    """Return a streaming FineWeb-Edu calibration dataset from either supported hub.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    dataset_name: FineWeb-Edu alias or explicit Hugging Face/ModelScope repository.
    split: The data split to use. FineWeb-Edu currently provides ``train``.
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A tokenized streaming dataset using the provided tokenizer and sequence length.
    """
    split = "train" if split is None else split
    if isinstance(split, list):
        if len(split) != 1:
            raise ValueError("FineWeb-Edu supports only one split at a time.")
        split = split[0]
    if split != "train":
        raise ValueError("FineWeb-Edu supports only the train split.")

    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )
    use_modelscope = dataset_name == _FINEWEB_EDU_MODELSCOPE_DATASET or (
        dataset_name == "fineweb-edu" and envs.AR_USE_MODELSCOPE
    )
    if use_modelscope:
        from transformers.utils.versions import require_version

        require_version(
            "modelscope",
            "Loading FineWeb-Edu from ModelScope requires `modelscope`; install it with `pip install modelscope`.",
        )
        from modelscope import MsDataset  # pylint: disable=E0401

        calib_dataset = MsDataset.load(
            _FINEWEB_EDU_MODELSCOPE_DATASET,
            subset_name=_FINEWEB_EDU_CONFIG,
            split=split,
            use_streaming=True,
        )
        # ModelScope's streaming reader can keep background HTTP requests alive
        # after iteration. Materialize the bounded sample before tokenization so
        # downloads finish cleanly and shuffle remains deterministic.
        calib_dataset = Dataset.from_list(list(calib_dataset.take(max_samples)))
        calib_dataset = calib_dataset.shuffle(seed=seed)
    else:
        calib_dataset = load_dataset(
            _FINEWEB_EDU_HF_DATASET,
            name=_FINEWEB_EDU_CONFIG,
            split=split,
            streaming=True,
        )
        calib_dataset = calib_dataset.shuffle(seed=seed).take(max_samples)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset(["BAAI/CCI3-HQ", "CCI3-HQ"])
def get_cci3_hq_dataset(
    tokenizer,
    seqlen,
    dataset_name="BAAI/CCI3-HQ",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
    nsamples=512,
):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )

    calib_dataset = load_dataset("BAAI/CCI3-HQ", split="train", streaming=True)
    calib_dataset = calib_dataset.shuffle(seed=seed).take(nsamples * envs.AR_CALIB_DATA_MULTIPLIER)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset(["codeparrot/github-code-clean", "github-code-clean"])
def get_github_code_clean_dataset(
    tokenizer,
    seqlen,
    dataset_name="codeparrot/github-code-clean",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
    nsamples=512,
):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """

    def get_default_tokenizer_function():
        """Returns a default tokenizer function.

        Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.
        apply_chat_template: Whether to apply chat template in tokenization.

        Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
         of seqlen to the "code" field of examples.
        """

        def default_tokenizer_function(examples):
            if not apply_chat_template:
                example = tokenizer(examples["code"], truncation=True, max_length=seqlen)
            else:
                example = apply_chat_template_to_samples(
                    examples["code"], tokenizer, seqlen, system_prompt=system_prompt
                )
            return example

        return default_tokenizer_function

    tokenizer_function = get_default_tokenizer_function()
    try:
        dataset_mit = load_dataset(
            "codeparrot/github-code-clean", "all-mit", split="train", trust_remote_code=True, streaming=True
        ).shuffle(seed=seed)
        dataset_apache = load_dataset(
            "codeparrot/github-code-clean", "all-apache-2.0", split="train", trust_remote_code=True, streaming=True
        ).shuffle(seed=seed)
    except RuntimeError as error:
        if "Dataset scripts are no longer supported" in str(error):
            raise RuntimeError(
                f"Failed to load dataset `{dataset_name}`\n"
                "💡 This dataset uses an old script-based format. To load it, please install `datasets<=3.6.0`:\n\n"
            )
        else:
            raise
    calib_dataset = concatenate_datasets([dataset_mit, dataset_apache])
    calib_dataset = calib_dataset.shuffle(seed=seed).take(
        nsamples * envs.AR_CALIB_DATA_MULTIPLIER
    )  ##TODO concat data'shuffle may have bugs
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset(["nvidia/OpenCodeInstruct", "opencode-instruct"])
def get_opencode_instruct_dataset(
    tokenizer,
    seqlen,
    dataset_name="nvidia/OpenCodeInstruct",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
    nsamples=512,
):
    """Return tokenized coding instructions and responses from OpenCodeInstruct."""
    split = "train" if split is None else split
    if isinstance(split, list):
        if len(split) != 1:
            raise ValueError("OpenCodeInstruct supports only the train split.")
        split = split[0]
    if split != "train":
        raise ValueError("OpenCodeInstruct supports only the train split.")

    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )

    dataset = load_dataset("nvidia/OpenCodeInstruct", split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed).take(nsamples * envs.AR_CALIB_DATA_MULTIPLIER)
    samples = []
    for data in dataset:
        if apply_chat_template:
            messages = []
            if system_prompt is not None and system_prompt != "":
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(
                [
                    {"role": "user", "content": data["input"]},
                    {"role": "assistant", "content": data["output"]},
                ]
            )
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except:
                logger.warning("Failed to apply chat template. removing the system role in chat history.")
                messages = [message for message in messages if message["role"] != "system"]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            samples.append({"text": text})
        else:
            samples.append({"text": f"{data['input']}\n\n{data['output']}"})
    random.Random(seed).shuffle(samples)

    calib_dataset = Dataset.from_list(samples)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )

    calib_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    input_ids, concat_input_ids = [eg["input_ids"] for eg in calib_dataset], []
    attention_mask_list, attention_mask = [], torch.ones([seqlen]).to(torch.int64)
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

            concat_input_ids.append(torch.cat(input_id_to_append).to(torch.int64).tolist())
            attention_mask_list.append(attention_mask.tolist())
            buffer_input_id = input_id[idx_keep:]
        else:
            buffer_input_id = torch.cat([buffer_input_id, input_id])

        if buffer_input_id.shape[-1] + os_cnt == seqlen:
            input_id_to_append = [buffer_input_id]
            if have_bos:
                input_id_to_append = [torch.tensor([bos_token_id])] + input_id_to_append
            if have_eos:
                input_id_to_append.append(torch.tensor([eos_token_id]))
            concat_input_ids.append(torch.cat(input_id_to_append).to(torch.int64).tolist())
            attention_mask_list.append(attention_mask.tolist())
            buffer_input_id = torch.Tensor().to(torch.int64)

    data = [{"input_ids": a, "attention_mask": b} for a, b in zip(concat_input_ids, attention_mask_list)]
    calib_dataset = Dataset.from_list(data)

    return calib_dataset


@register_dataset(["qwedsacf/competition_math", "competition_math"])
def get_competition_math_dataset(
    tokenizer,
    seqlen,
    dataset_name="qwedsacf/competition_math",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
    nsamples=512,
):
    """Return a calibration dataset built from the competition_math corpus.

    Each sample concatenates the ``problem`` and ``solution`` fields. Because the
    individual problem/solution pairs are much shorter than typical calibration
    sequence lengths (e.g. 2048), the tokenized samples are packed together into
    full ``seqlen``-length sequences instead of being dropped by the downstream
    length filter.

    Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum/target sequence length.
        dataset_name: Dataset identifier (defaults to ``qwedsacf/competition_math``).
        split: The data split to use. Defaults to ``train``.
        seed: The random seed for shuffling.
        apply_chat_template: Whether to apply the tokenizer's chat template.
        system_prompt: Optional system prompt for the chat template.
        nsamples: Number of calibration samples requested. The candidate pool is
            ``nsamples * AR_CALIB_DATA_MULTIPLIER`` before packing.

    Returns:
        A HuggingFace ``Dataset`` of packed, tokenized calibration sequences.
    """
    split = "train" if split is None else split
    if isinstance(split, list):
        if len(split) != 1:
            raise ValueError("competition_math supports only one split at a time.")
        split = split[0]

    # The chat template (when requested) is already rendered per-sample in the
    # loop below, so tokenize the resulting "text" plainly here to avoid applying
    # the template a second time (which would duplicate role markers/prompts).
    tokenizer_function = get_tokenizer_function(tokenizer, seqlen)

    # competition_math is small; load it fully instead of streaming.
    # Always use the canonical Hub repo id (the short "competition_math" alias
    # is only a registry key, not a valid Hub dataset name).
    dataset = load_dataset("qwedsacf/competition_math", split=split, trust_remote_code=False)
    dataset = dataset.shuffle(seed=seed)
    max_candidates = nsamples * envs.AR_CALIB_DATA_MULTIPLIER
    if len(dataset) > max_candidates:
        dataset = dataset.select(range(max_candidates))

    samples = []
    for data in dataset:
        problem = (data.get("problem") or "").strip()
        solution = (data.get("solution") or "").strip()
        if not problem and not solution:
            continue
        if apply_chat_template:
            messages = []
            if system_prompt is not None and system_prompt != "":
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(
                [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ]
            )
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                logger.warning("Failed to apply chat template. removing the system role in chat history.")
                messages = [message for message in messages if message["role"] != "system"]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            samples.append({"text": text})
        else:
            samples.append({"text": f"{problem}\n\n{solution}"})
    random.Random(seed).shuffle(samples)

    calib_dataset = Dataset.from_list(samples)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )

    # Sample-level concat by default: pack whole samples until each reaches
    # seqlen, then reset, so short problem/solution pairs are not dropped by the
    # downstream length filter.
    calib_dataset = _concat_samples_to_seqlen(calib_dataset, seqlen)

    return calib_dataset


@register_dataset(["HuggingFaceH4/ultrachat_200k", "ultrachat_200k"])
def get_ultrachat_dataset(
    tokenizer,
    seqlen,
    dataset_name="HuggingFaceH4/ultrachat_200k",
    split=None,
    seed=42,
    apply_chat_template=True,
    system_prompt=None,
    nsamples=512,
):
    if split is None:
        split = "train_sft"
    all_splits = ["train_sft", "test_sft", "train_gen", "test_gen"]
    if split not in all_splits:
        raise ValueError(f"split must be one of {all_splits} for ultrachat_200k ")

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True, trust_remote_code=True)
    dataset = dataset.shuffle(seed=seed).take(nsamples * envs.AR_CALIB_DATA_MULTIPLIER)

    def is_instruct_tokenizer(tokenizer):
        try:
            out = tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}])
            return bool(out and len(out) > 0)
        except Exception:
            return False

    is_instruct = is_instruct_tokenizer(tokenizer)

    if not is_instruct and apply_chat_template:
        logger.info("Tokenizer is not an instruct/chat model, but apply_chat_template=True. Setting to False.")
        apply_chat_template = False

    def tokenize_example_batch(examples):
        if not apply_chat_template:
            texts = []
            for message_list in examples["messages"]:
                combined = "".join([msg["content"] for msg in message_list])
                texts.append(combined)
            return tokenizer(texts, truncation=True, max_length=seqlen)
        else:
            return apply_chat_template_to_samples(examples["messages"], tokenizer, seqlen, system_prompt=system_prompt)

    dataset = dataset.map(tokenize_example_batch, batched=True)
    return dataset


@register_dataset(["openbmb/UltraData-SFT-Agent-2609", "UltraData-SFT-Agent-2609"])
def get_ultradata_agent_dataset(
    tokenizer,
    seqlen,
    dataset_name="openbmb/UltraData-SFT-Agent-2609",
    split=None,
    seed=42,
    apply_chat_template=True,
    system_prompt=None,
    nsamples=512,
):
    """Agent/tool-calling calibration dataset.

    Each sample carries a ``messages`` conversation (system/user/assistant with
    ``tool_calls``/tool turns) and a ``tools`` schema list. The chat template is
    rendered with ``tools=`` so the quantized model sees full tool-calling
    traces. ``apply_chat_template`` is forced on because the raw ``messages``
    structure is only meaningful once templated.

    The dataset ships multiple configs (``Code-Agent``, ``Search-Agent``,
    ``General-Agent``, ``Tool-Use``); all of them are loaded and interleaved so
    calibration covers the full range of agent behaviors.

    Length-aware sampling: conversations in this dataset can be very long. A
    sample whose *messages alone* already exceed ``seqlen`` would have its actual
    conversation content truncated, so such samples are deprioritized. We first
    collect samples whose rendered messages fit within ``seqlen`` (even if
    ``messages + tools`` exceeds it, since only the tool schema overflows); only
    if there are not enough of those do we fall back to the oversized ones.
    """
    if split is None:
        split = "train"

    # Always load from the canonical Hub repo id, regardless of whether the user
    # passed the full name or the short alias ("UltraData-SFT-Agent-2609").
    hub_repo_id = "openbmb/UltraData-SFT-Agent-2609"

    # This dataset is split into multiple configs; mix them all so the quantized
    # model sees the full range of agent behaviors (code / search / general /
    # tool-use). We avoid ``interleave_datasets`` on purpose: the configs have
    # slightly different ``tools`` feature schemas (e.g. an extra ``title`` key),
    # which makes HF feature alignment fail. Since we only read ``messages`` and
    # ``tools`` and render them ourselves, we scan each config independently and
    # then balance the final selection across configs.
    configs = ["Code-Agent", "Search-Agent", "General-Agent", "Tool-Use"]
    sub_datasets = []
    for cfg in configs:
        try:
            d = load_dataset(hub_repo_id, cfg, split=split, streaming=True)
            d = d.shuffle(seed=seed, buffer_size=1000)
            sub_datasets.append(d)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load config '%s' of %s: %s", cfg, hub_repo_id, e)

    if not sub_datasets:
        raise ValueError(f"Could not load any config of {hub_repo_id} (tried {configs}).")

    if not apply_chat_template:
        logger.info(
            "apply_chat_template=False is ignored for %s; agent/tool data requires chat templating.", hub_repo_id
        )

    # target = nsamples * envs.AR_CALIB_DATA_MULTIPLIER
    target = nsamples * 2
    n_cfg = len(sub_datasets)
    # Per-config quota so every config contributes a balanced share. Collect a
    # bit extra per config so that, if one config is short, others can cover the
    # shortfall during the balanced round-robin selection below.
    per_cfg_quota = -(-target // n_cfg)  # ceil(target / n_cfg)
    # Safety bound per config: don't scan a whole (huge) config looking for
    # short-enough samples when they are scarce.
    max_scan_per_cfg = target

    # The per-sample system prompt in this dataset is very long and pushes most
    # samples past ``seqlen``. Drop it with high probability so the bulk of the
    # calibration data reflects the actual conversation/tool content, while a
    # small fraction still keeps the system prompt so the model sees that format.
    drop_system_prob = 0.99
    rng = random.Random(seed)

    def _maybe_drop_system(messages):
        """Drop a leading system message with probability ``drop_system_prob``."""
        if (
            messages
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "system"
            and rng.random() < drop_system_prob
        ):
            return messages[1:]
        return messages

    def _token_len(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def _approx_word_count(messages):
        """Cheap whitespace-split length estimate of the raw conversation.

        Token count is almost always >= whitespace word count (subword / code /
        JSON only increase it), and the chat template only adds more tokens. So
        if this estimate already exceeds ``seqlen`` the templated sample is
        definitely longer than ``seqlen`` and we can skip the expensive
        ``apply_chat_template`` + tokenization entirely.
        """
        count = 0
        for msg in messages:
            content = msg.get("content")
            if content:
                count += len(content.split())
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                count += len(str(tool_calls).split())
        return count

    def _build_length_conversation(messages):
        """Build the message list used for the messages-only length check.

        Mirrors the system-prompt handling of ``render_chat_sample`` (tools are
        excluded here on purpose: we filter on the messages-only length).
        """
        conv = []
        has_own_system = len(messages) > 0 and isinstance(messages[0], dict) and messages[0].get("role") == "system"
        if system_prompt is not None and system_prompt != "" and not has_own_system:
            conv.append({"role": "system", "content": system_prompt})
        conv += messages
        return conv

    def _batch_msg_lengths(conversations):
        """Batch-render conversations and return their token lengths.

        We render to strings first (``tokenize=False`` on a batch reliably
        returns a list[str]) and then tokenize the strings in one batched call.
        This avoids the version-dependent return type of
        ``apply_chat_template(batch, tokenize=True)`` (which can be a dict-like
        ``BatchEncoding`` and make per-sample length extraction wrong).
        ``add_generation_prompt=False`` because these are full traces and the few
        generation-prompt tokens are negligible for a length filter.
        """
        if not conversations:
            return []
        try:
            rendered = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        except Exception:
            # Fallback: per-sample, dropping the system role if the template fails.
            rendered = []
            for conv in conversations:
                try:
                    rendered.append(tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False))
                except Exception:
                    conv2 = [m for m in conv if m.get("role") != "system"]
                    rendered.append(tokenizer.apply_chat_template(conv2, tokenize=False, add_generation_prompt=False))
        enc = tokenizer(rendered, add_special_tokens=False)
        return [len(ids) for ids in enc["input_ids"]]

    # Scan each config separately. Store raw (messages, tools) and only render
    # the finally selected samples with tools, so we don't pay the cost of fully
    # rendering the (large, mostly unused) fallback pool. The accurate length
    # check is done in a single batched template call per config.
    preferred_by_cfg = [[] for _ in sub_datasets]  # messages fit within seqlen
    fallback_by_cfg = [[] for _ in sub_datasets]  # messages exceed seqlen (gets truncated)

    for i, d in enumerate(sub_datasets):
        scanned = 0
        candidates = []  # (messages, tools) that passed the cheap word-count filter
        for sample in d:
            if len(candidates) >= per_cfg_quota or scanned >= max_scan_per_cfg:
                break
            scanned += 1
            messages = sample["messages"]
            tools = sample.get("tools") if isinstance(sample, dict) else sample["tools"]

            # Decide once (at read time) whether to drop the long system prompt,
            # so word-count, length check and final render all stay consistent.
            messages = _maybe_drop_system(messages)

            # Fast path: if even the raw whitespace word count exceeds seqlen, the
            # sample is definitely too long -> skip rendering/tokenizing.
            if _approx_word_count(messages) > seqlen:
                if len(fallback_by_cfg[i]) < per_cfg_quota:
                    fallback_by_cfg[i].append((messages, tools))
                continue
            candidates.append((messages, tools))

        # Batched accurate length check for this config's candidates.
        lengths = _batch_msg_lengths([_build_length_conversation(m) for m, _ in candidates])
        for (messages, tools), length in zip(candidates, lengths):
            if length <= seqlen:
                preferred_by_cfg[i].append((messages, tools))
            elif len(fallback_by_cfg[i]) < per_cfg_quota:
                fallback_by_cfg[i].append((messages, tools))

    def _round_robin_take(pools, limit, already):
        """Pull items round-robin across per-config pools until ``limit``."""
        out = []
        idx = [0] * len(pools)
        while already + len(out) < limit and any(idx[i] < len(pools[i]) for i in range(len(pools))):
            for i in range(len(pools)):
                if already + len(out) >= limit:
                    break
                if idx[i] < len(pools[i]):
                    out.append(pools[i][idx[i]])
                    idx[i] += 1
        return out

    # Balanced selection: round-robin across configs' preferred pools first,
    # then top up (still round-robin) from the fallback pools if short.
    selected = _round_robin_take(preferred_by_cfg, target, 0)
    n_preferred = len(selected)
    if len(selected) < target:
        selected += _round_robin_take(fallback_by_cfg, target, len(selected))
        if len(selected) > n_preferred:
            logger.info(
                "%s: only %d samples fit messages<=seqlen; topping up with %d oversized samples.",
                hub_repo_id,
                n_preferred,
                len(selected) - n_preferred,
            )

    # Global shuffle so configs/tiers are mixed across the calibration set.
    random.Random(seed).shuffle(selected)

    # Render the final selection with tools included.
    texts = [render_chat_sample(tokenizer, messages, system_prompt, tools=tools) for messages, tools in selected]

    import datasets

    calib_dataset = datasets.Dataset.from_list([{"text": t} for t in texts])
    calib_dataset = calib_dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, max_length=seqlen),
        batched=True,
        remove_columns=["text"],
    )

    # Pack whole traces up to seqlen before returning. "Preferred" traces are
    # chosen on their messages-only length (<= seqlen) and can remain shorter
    # than seqlen even after rendering tools, so returning them unpacked would
    # let _get_dataset_impl.filter_func drop them -- leaving calibration
    # undersized despite available oversized fallback data. Sample-level packing
    # concatenates short traces into full-length sequences, while oversized
    # traces (already truncated to seqlen) each emit a single full-length
    # sequence, so no usable data is lost.
    calib_dataset = _concat_samples_to_seqlen(calib_dataset, seqlen)
    return calib_dataset


@register_dataset("openbmb/Ultra-FineWeb")
def get_ultrafinweb_dataset(
    tokenizer,
    seqlen,
    dataset_name="openbmb/Ultra-FineWeb",
    split=None,
    seed=42,
    apply_chat_template=True,
    system_prompt=None,
    nsamples=512,
):
    if split is not None:
        if split not in ["en", "zh"]:
            raise ValueError("split must be one of ['en', 'zh'] for Ultra-FineWeb dataset")
        calib_dataset = load_dataset("openbmb/Ultra-FineWeb", split=split, streaming=True)
    else:
        calib_dataset = load_dataset("openbmb/Ultra-FineWeb", split="en", streaming=True)
        # dataset_ch = load_dataset("openbmb/Ultra-FineWeb", split='zh',
        #                           streaming=True, trust_remote_code=True).shuffle(seed=seed).take(2000)

        # calib_dataset = concatenate_datasets([dataset_en, dataset_ch]) ##concat dasetset could not shuffle

    calib_dataset = calib_dataset.shuffle(seed=seed).take(nsamples * envs.AR_CALIB_DATA_MULTIPLIER)

    def get_default_tokenizer_function():
        def default_tokenizer_function(examples):
            if not apply_chat_template:
                example = tokenizer(examples["content"], truncation=True, max_length=seqlen)
            else:
                example = apply_chat_template_to_samples(
                    examples["content"], tokenizer, seqlen, system_prompt=system_prompt
                )
            return example

        return default_tokenizer_function

    tokenizer_function = get_default_tokenizer_function()

    dataset = calib_dataset.map(tokenizer_function, batched=True)
    return dataset


@register_dataset(["madao33/new-title-chinese", "new-title-chinese"])
def get_new_chinese_title_dataset(
    tokenizer,
    seqlen,
    dataset_name="madao33/new-title-chinese",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
):
    """
    Returns a tokenized dataset for the specified parameters.

    Args:
        tokenizer: The tokenizer to use.
        seqlen: Maximum sequence length.
        dataset_name: Name of the dataset to load.
        split: Which split of the dataset to use.
        seed: Random seed for shuffling.
        apply_template: Whether to apply a template to the data.

    Returns:
        A tokenized and shuffled dataset.
    """

    def get_tokenizer_function():
        """Returns a default tokenizer function.

        Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.
        apply_chat_template: Whether to apply chat template in tokenization.

        Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
        of seqlen to the "text" field of examples.
        """

        def default_tokenizer_function(examples):
            if not apply_chat_template:
                example = tokenizer(examples["content"], truncation=True, max_length=seqlen)
            else:
                example = apply_chat_template_to_samples(
                    examples["content"], tokenizer, seqlen, system_prompt=system_prompt
                )
            return example

        return default_tokenizer_function

    split = "train"
    from datasets import load_dataset

    tokenizer_function = get_tokenizer_function()

    calib_dataset = load_dataset("madao33/new-title-chinese", split=split)
    calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "content"
        ),
    )

    return calib_dataset


@register_dataset("mbpp")
@register_dataset("google-research-datasets/mbpp")
def get_mbpp_dataset(
    tokenizer,
    seqlen,
    dataset_name="google-research-datasets/mbpp",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    dataset_name = "google-research-datasets/mbpp"

    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )

    samples = []
    splits = split
    if splits is None:
        splits = ["train", "validation", "test"]
    if isinstance(splits, str):
        splits = splits.split("+")

    for split_name in splits:
        dataset = load_dataset(dataset_name, split=split_name)
        for data in dataset:
            samples.append({"text": data["text"] + data["code"]})
    random.Random(seed).shuffle(samples)
    import datasets

    calib_dataset = datasets.Dataset.from_list(samples)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )

    return calib_dataset


@register_dataset("audiocaps")
def get_audiocaps_dataset(
    tokenizer,
    seqlen,
    dataset_name="audiocaps",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
):
    """Returns a calibration dataset from AudioCaps (audio caption text).

    AudioCaps provides human-written captions describing audio events.  The captions
    are rich in descriptive language about sounds, music, and speech — making them
    well-suited as calibration text for audio-related model quantization.

    The dataset CSV is downloaded from the official GitHub repository on first use.

    Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.
        dataset_name: Dataset identifier (ignored, always loads AudioCaps).
        split: Unused (AudioCaps train split is always used).
        seed: The random seed for shuffling.
        apply_chat_template: Whether to apply chat template in tokenization.
        system_prompt: Optional system prompt for chat template.

    Returns:
        A HuggingFace Dataset with tokenized audio captions.
    """
    import csv

    from auto_round.utils import download_audiocaps_csv

    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )

    cache_file = download_audiocaps_csv()

    samples = []
    with open(cache_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            caption = row.get("caption", "").strip()
            if caption:
                samples.append({"text": caption})

    if not samples:
        raise RuntimeError("AudioCaps dataset is empty or could not be parsed.")

    random.Random(seed).shuffle(samples)
    import datasets as ds

    calib_dataset = ds.Dataset.from_list(samples)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )
    return calib_dataset


@register_dataset("local")
def get_local_dataset(
    tokenizer, seqlen, dataset_name="./tmp.json", split=None, seed=42, apply_chat_template=False, system_prompt=None
):
    """Returns a dataloader for a custom dataset and split.
    We allow the input of a json or text file containing a processed text sample each line.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name or path of the dataset, which is a json or jsonl file.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for a custom dataset and split, using the provided tokenizer and sequence length.
    """
    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )

    def load_local_data(data_path):
        if data_path.endswith(".json"):
            with open(data_path, "r") as f:
                data = json.load(f)
            return data
        elif data_path.endswith(".jsonl"):
            data = []
            with open(data_path) as f:
                for line in f:
                    sample = json.loads(line)
                    data.append(sample)
            return data
        else:
            logger.error("invalid local file type, for now only support json/jsonl format data file.")

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
        if not isinstance(text, str):
            raise TypeError("data must be a string")
        text = text.rstrip()
        text = text.rstrip("\n")
        samples.append({"text": text})
    random.Random(seed).shuffle(samples)
    import datasets

    calib_dataset = datasets.Dataset.from_list(samples)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )
    return calib_dataset


def _concat_samples_to_seqlen(calib_dataset, seqlen):
    """Pack whole samples together until each reaches ``seqlen`` tokens.

    Unlike the continuous packing in :func:`concat_dataset_element` (where a
    sample's leftover tokens carry over into the next sequence, splitting
    samples across boundaries), this packs *whole* samples into a buffer and,
    once the buffer reaches ``seqlen``, emits a single ``seqlen``-length
    sequence and resets the buffer. The next sample therefore starts a fresh
    concatenation; any trailing buffer shorter than ``seqlen`` is dropped.

    Args:
        calib_dataset: A tokenized ``Dataset`` with an ``input_ids`` column.
        seqlen: Target sequence length for each packed sample.

    Returns:
        A new ``Dataset`` with packed ``input_ids`` / ``attention_mask`` columns.
    """
    packed = []
    buffer_ids = []
    for example in calib_dataset:
        ids = example["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        buffer_ids.extend(ids)
        if len(buffer_ids) >= seqlen:
            trimmed = buffer_ids[:seqlen]
            packed.append({"input_ids": trimmed, "attention_mask": [1] * seqlen})
            # Reset: the next sample starts a fresh concatenation.
            buffer_ids = []

    if not packed:
        logger.warning(
            "Sample-level concat produced no full-length sequences (all samples "
            "shorter than seqlen=%d combined did not reach it). Returning the "
            "original dataset unpacked.",
            seqlen,
        )
        return calib_dataset

    return Dataset.from_list(packed)


def _normalize_split(split):
    """Reduce a requested split spec to a single ``load_dataset`` split string.

    The dispatcher passes splits as lists (e.g. ``["train", "test"]``); combine
    every requested split into HuggingFace's ``"train+test"`` form so they are
    loaded together instead of only the first one being used. Returns ``None``
    when nothing was requested, so the caller can pick an available default from
    the resulting split dictionary.
    """
    if split is None:
        return None
    if isinstance(split, (list, tuple)):
        parts = [str(part) for part in split if part]
        return "+".join(parts) if parts else None
    return split


def _reduce_to_single_dataset(loaded):
    """Reduce a ``load_dataset`` result to a single (Iterable)Dataset.

    When a concrete split string is passed, ``load_dataset`` already returns a
    single dataset and this is a no-op. When no split is requested it returns a
    split dictionary; in that case pick a sensible default (prefer ``train``,
    otherwise the first available split) so downstream ``.map()`` / column
    access receive a single dataset rather than a dict or list.
    """
    if isinstance(loaded, (DatasetDict, IterableDatasetDict)):
        keys = list(loaded.keys())
        if not keys:
            raise ValueError("Loaded dataset contains no splits to select from.")
        key = "train" if "train" in keys else keys[0]
        return loaded[key]
    # Defensive: a list is returned if a list split ever reaches load_dataset;
    # combine the splits so callers always receive a single dataset.
    if isinstance(loaded, list):
        if not loaded:
            raise ValueError("Loaded dataset contains no splits to combine.")
        if len(loaded) == 1:
            return loaded[0]
        if any(isinstance(part, IterableDataset) for part in loaded):
            from datasets import interleave_datasets

            return interleave_datasets(loaded)
        return concatenate_datasets(loaded)
    return loaded


def _get_generic_dataset(
    tokenizer,
    seqlen,
    dataset_name,
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
    fields=None,
    template=None,
    separator="\n\n",
    streaming=True,
    nsamples=None,
    timeout=300,
    concat=False,
    trust_remote_code=False,
):
    """Generic dataset loader that works with any HuggingFace dataset.

    Automatically detects the text field if not specified, or uses the
    user-provided fields/template to extract text.

    By default uses streaming mode to avoid downloading the entire dataset.
    Samples are collected until ``nsamples`` valid samples (meeting ``seqlen``)
    are gathered, or the ``timeout`` is reached.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        seqlen: The maximum sequence length.
        dataset_name: HuggingFace dataset name or local path.
        split: The data split to use.
        seed: Random seed for shuffling.
        apply_chat_template: Whether to apply chat template.
        system_prompt: Optional system prompt.
        fields: Text field(s) to extract. A single string for one column,
            or a list of strings for multiple columns (concatenated with
            *separator*). If ``None``, the text field is auto-detected.
        template: Template string with {field} placeholders.
        separator: Separator for multi-field concatenation.
        streaming: Whether to use streaming mode (default True for remote
            datasets to avoid downloading the full dataset).
        nsamples: Maximum number of samples to collect (for streaming mode).
            If None, collects up to 10000 samples.
        timeout: Timeout in seconds for streaming collection. If the dataset
            cannot provide enough samples within this time, a warning is
            logged and the collected samples are used.
        concat: When True, pack whole samples together until each reaches
            ``seqlen`` tokens (resetting after every full sequence), so that
            datasets whose individual samples are shorter than ``seqlen`` are
            not dropped by the downstream length filter.
        trust_remote_code: Passed through to ``datasets.load_dataset``. Controlled
            by the AutoRound interface so the user decides whether to execute
            remote dataset-loading scripts. Defaults to ``False``.

    Returns:
        A tokenized HuggingFace Dataset.
    """
    import time

    # Determine if we should use streaming
    # Local files are always loaded fully (they're already on disk)
    is_local = is_local_path(dataset_name)
    use_streaming = streaming and not is_local

    # Normalize the requested split(s) once so local, streaming and
    # non-streaming branches behave consistently. The dispatcher passes splits
    # as lists; ``_normalize_split`` combines them into HF's "train+test" form
    # (or ``None`` when nothing was requested).
    split_arg = _normalize_split(split)

    # Load the dataset
    if is_local:
        # Local file: use json or text loader (always full load)
        if dataset_name.endswith((".json", ".jsonl")):
            loaded = load_dataset("json", data_files=dataset_name, split=split_arg)
        elif dataset_name.endswith(".txt"):
            loaded = load_dataset("text", data_files=dataset_name, split=split_arg)
        else:
            loaded = load_dataset(dataset_name, split=split_arg)
    else:
        # HuggingFace dataset. ``trust_remote_code`` is passed through from the
        # AutoRound interface so the user controls whether remote code runs.
        loaded = load_dataset(
            dataset_name,
            split=split_arg,
            trust_remote_code=trust_remote_code,
            streaming=use_streaming,
        )

    # When no split was requested, ``load_dataset`` returns a split dictionary;
    # reduce it (and any list result) to a single dataset for extraction below.
    calib_dataset = _reduce_to_single_dataset(loaded)

    # Determine the text field
    if fields is None and template is None:
        # Auto-detect. Streaming IterableDatasets may expose ``column_names`` as
        # None before the schema is resolved, so check for non-None metadata and
        # otherwise detect from a single sample.
        if getattr(calib_dataset, "column_names", None) is not None:
            fields = auto_detect_text_field(calib_dataset)
        else:
            # IterableDataset / unresolved schema – grab first sample to detect
            first_sample = next(iter(calib_dataset))
            fields = auto_detect_text_field_from_sample(first_sample)
        logger.info(f"Auto-detected text field '{fields}' for dataset '{dataset_name}'")

    # Build the text extraction function
    def _extract_text_from_sample(sample):
        """Extract text from a single sample dict."""
        try:
            return extract_text_from_sample(
                sample,
                fields=fields,
                template=template,
                separator=separator,
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping sample due to field extraction error: {e}")
            return None

    # Tokenizer function
    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )

    if use_streaming:
        # --- Streaming mode: collect samples until nsamples or timeout ---
        # Over-collect by AR_CALIB_DATA_MULTIPLIER (same as registered datasets)
        # so that enough samples survive the downstream seqlen filter in
        # ``_get_dataset_impl`` (short samples < seqlen are dropped there).
        target = nsamples * envs.AR_CALIB_DATA_MULTIPLIER if nsamples is not None else 10000
        start_time = time.time()
        collected_texts = []
        total_scanned = 0
        skipped = 0

        logger.info(
            f"Streaming dataset '{dataset_name}': collecting up to {target} samples " f"(timeout={timeout}s)..."
        )

        # Shuffle the iterable dataset for random sampling
        iter_ds = calib_dataset.shuffle(seed=seed)

        for sample in iter_ds:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(
                    f"Dataset '{dataset_name}' streaming timeout reached ({timeout}s). "
                    f"Collected {len(collected_texts)}/{target} samples after scanning "
                    f"{total_scanned} samples ({skipped} skipped). "
                    f"Using available samples for calibration."
                )
                break

            total_scanned += 1
            text = _extract_text_from_sample(sample)
            if text is None or not text.strip():
                skipped += 1
                continue

            collected_texts.append(text)

            if len(collected_texts) >= target:
                break

            # Progress logging every 1000 samples
            if total_scanned % 1000 == 0:
                logger.info(
                    f"  Streaming '{dataset_name}': {len(collected_texts)}/{target} "
                    f"valid samples collected ({total_scanned} scanned, {skipped} skipped, "
                    f"{elapsed:.1f}s elapsed)"
                )

        if len(collected_texts) == 0:
            raise ValueError(
                f"Dataset '{dataset_name}' yielded no valid samples within {timeout}s. "
                f"Scanned {total_scanned} samples. Check the dataset format or "
                f"specify 'fields' explicitly."
            )

        if len(collected_texts) < target:
            logger.warning(
                f"Dataset '{dataset_name}': only {len(collected_texts)}/{target} samples "
                f"collected (scanned {total_scanned} total, {skipped} skipped). "
                f"Calibration will use {len(collected_texts)} samples."
            )

        # Build a Dataset from collected texts
        calib_dataset = Dataset.from_dict({"text": collected_texts})

    else:
        # --- Non-streaming mode: full dataset ---
        def _extract_text_batched(examples):
            """Extract text from batched examples."""
            texts = []
            for i in range(len(next(iter(examples.values())))):
                sample = {k: v[i] for k, v in examples.items()}
                text = _extract_text_from_sample(sample)
                if text is not None:
                    texts.append(text)
                else:
                    texts.append("")  # placeholder, will be filtered later
            return {"text": texts}

        if fields is not None or template is not None:
            calib_dataset = calib_dataset.map(
                _extract_text_batched,
                batched=True,
                remove_columns=calib_dataset.column_names if hasattr(calib_dataset, "column_names") else None,
            )
        else:
            # Auto-detected field: rename to "text" if needed
            if fields != "text":
                calib_dataset = calib_dataset.rename_column(fields, "text")

        calib_dataset = calib_dataset.shuffle(seed=seed)

    # Filter out empty/short texts and tokenize
    calib_dataset = calib_dataset.filter(lambda x: len(x["text"].strip()) > 0)

    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )

    # Pack short samples into full seqlen-length sequences when requested.
    if concat:
        calib_dataset = _concat_samples_to_seqlen(calib_dataset, seqlen)

    return calib_dataset


def get_dataset_len(dataset):
    """Calculates the length of a dataset.

    Args:
        dataset: The dataset object, which can be any iterable or collection.

    Returns:
        int: The length of the dataset.

    Raises:
        If the dataset does not support `len()`, iterates through it to count the number of elements.
    """
    try:
        dataset_len = len(dataset)
        return dataset_len
    except Exception:
        cnt = 0
        for _ in dataset:
            cnt += 1
        return cnt


def select(dataset, indices):
    """Selects specific elements from a dataset based on given indices.

    Args:
        dataset: The dataset object to iterate over.
        indices: An iterable of integers specifying the indices to select.

    Yields:
        Elements of the dataset corresponding to the specified indices.

    Notes:
        Stops iterating once the highest index in `indices` has been processed
        to optimize performance.
    """
    indices = set(indices)
    for idx, sample in enumerate(dataset):
        if idx in indices:
            yield sample
        if idx > max(indices):
            break


def select_dataset(dataset, indices):
    """Selects elements from a dataset using its native `select` method, if available.

    Args:
        dataset: The dataset object, which may have a `select` method.
        indices: An iterable of integers specifying the indices to select.

    Returns:
        A subset of the dataset, either using the dataset's `select` method or the
        `select` function defined above as a fallback.
    """
    try:
        return dataset.select(indices)
    except Exception:
        list_data = list(select(dataset, indices))
        import pandas as pd

        df = pd.DataFrame(list_data)
        dataset = Dataset.from_pandas(df)
        return dataset


def _get_dataset_impl(
    tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", seed=42, nsamples=512, trust_remote_code=False
):
    """Internal implementation: generate a dataset for calibration.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for tokenization.
        seqlen (int): The exact sequence length. samples < seqlen will be dropped,
                      samples longer than seqlen will be truncated
        dataset_name (str, optional): The name of the dataset or datasets separated by commas.
                                     Defaults to "NeelNanda/pile-10k".
        split (str, optional): The data split to use. Defaults to None.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        nsamples (int, optional): The total number of samples to include. Defaults to 512.
        apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
        Dataset: The processed dataset ready for calibration.
    """
    # Normalize CalibDataset objects to spec strings
    if isinstance(dataset_name, CalibDataset):
        dataset_name = dataset_name.to_spec_string()
    elif isinstance(dataset_name, (list, tuple)):
        dataset_name = normalize_dataset_spec(dataset_name)

    # Split on unescaped commas so that commas inside option values (e.g. a
    # system_prompt or template) are not mistaken for dataset separators.
    dataset_names = split_unescaped(dataset_name, ",")

    def cast_dataset_columns(dataset):
        features = dict(dataset.features)
        features["input_ids"] = Sequence(Value("int64"))
        features["attention_mask"] = Sequence(Value("int8"))
        return dataset.cast(Features(features))

    def filter_func(example):
        if isinstance(example["input_ids"], list):
            example["input_ids"] = torch.tensor(example["input_ids"])
        if example["input_ids"].shape[-1] < seqlen:
            return False
        input_ids = example["input_ids"][:seqlen]
        input_ids_list = input_ids.tolist()
        return not (len(input_ids_list) > 1 and seqlen > 2 and input_ids_list.count(input_ids_list[-1]) > seqlen // 2)

    def concat_dataset_element(dataset):
        input_ids, concat_input_ids = [eg["input_ids"] for eg in dataset], []
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
        data = [{"input_ids": a, "attention_mask": b} for a, b in zip(concat_input_ids, attention_mask_list)]
        import datasets

        dataset_new = datasets.Dataset.from_list(data)
        return dataset_new

    datasets, data_lens = [], {}
    system_prompt = "You are a helpful assistant."
    for name in dataset_names:
        split = None
        do_concat = False
        apply_chat_template = False
        fields = None
        template = None
        separator = "\n\n"
        timeout = 300

        if ":" in name:
            # Split on unescaped ":" so values containing ":" (e.g. templates)
            # are not truncated; the first segment is the dataset name.
            segments = split_unescaped(name, ":")
            name = segments[0]
            for ele in segments[1:]:
                if "=" in ele:
                    key, raw_value = ele.split("=", 1)
                elif ele.strip() in ("concat", "apply_chat_template"):
                    # Preserve the bare boolean syntax (e.g. "pile-10k:apply_chat_template").
                    key, raw_value = ele, "true"
                else:
                    # Ignore incomplete non-boolean options.
                    continue
                key = key.strip()
                # Restore escaped delimiters / whitespace; keep separator spaces.
                value = unescape_spec_value(raw_value if key == "separator" else raw_value.strip())
                if key == "split":
                    split = value.split("+")
                elif key == "num":
                    data_lens[name] = int(value)
                elif key == "concat":
                    do_concat = value.lower() != "false"
                elif key == "apply_chat_template":
                    apply_chat_template = value.lower() != "false"
                elif key == "system_prompt":
                    system_prompt = value
                    apply_chat_template = True
                elif key == "fields":
                    # Single field: "fields=text" -> "text"
                    # Multiple fields: "fields=q+a" -> ["q", "a"]
                    if "+" in value:
                        fields = value.split("+")
                    else:
                        fields = value
                elif key == "template":
                    template = value
                elif key == "separator":
                    separator = value
                elif key == "timeout":
                    timeout = int(value)
        # If user explicitly specified fields/template, use the generic
        # loader regardless of whether the dataset is registered.
        use_generic = fields is not None or template is not None

        if use_generic or is_local_path(name):
            if is_local_path(name) and not use_generic:
                # Local file without explicit field spec: use the local handler
                dataset = CALIB_DATASETS["local"](
                    tokenizer,
                    seqlen,
                    seed=seed,
                    split=split,
                    dataset_name=name,
                    apply_chat_template=apply_chat_template,
                    system_prompt=system_prompt,
                )
            else:
                # Generic loader (for local files with field spec, or any HF dataset)
                dataset = _get_generic_dataset(
                    tokenizer,
                    seqlen,
                    dataset_name=name,
                    split=split,
                    seed=seed,
                    apply_chat_template=apply_chat_template,
                    system_prompt=system_prompt,
                    fields=fields,
                    template=template,
                    separator=separator,
                    nsamples=nsamples,
                    timeout=timeout,
                    concat=do_concat,
                    trust_remote_code=trust_remote_code,
                )
                # Packing already handled inside the generic loader.
                do_concat = False
        else:
            calib_name = name
            if name not in CALIB_DATASETS:
                calib_name = name.split("/")[-1]
                for key in CALIB_DATASETS:
                    if calib_name in key:
                        calib_name = key
                        break
            get_dataset = CALIB_DATASETS.get(calib_name)
            if get_dataset is None:
                # Fallback: use generic loader for any HuggingFace dataset
                logger.info(f"Dataset '{name}' not in registry, using generic loader with auto field detection.")
                dataset = _get_generic_dataset(
                    tokenizer,
                    seqlen,
                    dataset_name=name,
                    split=split,
                    seed=seed,
                    apply_chat_template=apply_chat_template,
                    system_prompt=system_prompt,
                    fields=fields,
                    template=template,
                    separator=separator,
                    nsamples=nsamples,
                    timeout=timeout,
                    concat=do_concat,
                    trust_remote_code=trust_remote_code,
                )
                # Packing already handled inside the generic loader.
                do_concat = False
            else:
                if get_dataset is None:
                    filtered_keys = [k for k in CALIB_DATASETS.keys() if "/" not in k]
                    raise ValueError(
                        f"Dataset '{name}' is not found. Please choose from the supported datasets: {filtered_keys}."
                    )
                dataset_kwargs = dict(
                    tokenizer=tokenizer,
                    seqlen=seqlen,
                    seed=seed,
                    split=split,
                    dataset_name=name,
                    apply_chat_template=apply_chat_template,
                    system_prompt=system_prompt,
                )
                if get_dataset is get_fineweb_edu_dataset:
                    dataset_kwargs["max_samples"] = max(
                        _FINEWEB_EDU_MIN_CANDIDATES,
                        data_lens.get(name, nsamples),
                    )
                elif get_dataset in (
                    get_cci3_hq_dataset,
                    get_github_code_clean_dataset,
                    get_opencode_instruct_dataset,
                    get_competition_math_dataset,
                    get_ultrachat_dataset,
                    get_ultrafinweb_dataset,
                    get_ultradata_agent_dataset,
                ):
                    dataset_kwargs["nsamples"] = nsamples
                dataset = get_dataset(**dataset_kwargs)

        if do_concat:
            dataset = concat_dataset_element(dataset)

        if isinstance(dataset, IterableDataset):
            # Filter and limit the stream before materializing it in memory.
            dataset = dataset.filter(filter_func)
            if name in data_lens:
                dataset = select_dataset(dataset, range(data_lens[name]))
            # select_dataset may have already materialized the requested rows.
            if isinstance(dataset, IterableDataset):
                dataset = Dataset.from_list(list(itertools.islice(dataset, nsamples)))
            dataset = cast_dataset_columns(dataset)
        else:
            # Cast before filter/select creates indices, avoiding a full values
            # buffer conversion for each indexed Arrow row.
            dataset = cast_dataset_columns(dataset)
            dataset = dataset.filter(filter_func)
            if name in data_lens:
                dataset = select_dataset(dataset, range(data_lens[name]))
        # Format last: tensorizing whole batches during filter/select would run torch ops that
        # are not fork-safe (libgomp) in the preprocessing subprocess.
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        datasets.append(dataset)

    if len(datasets) == 1:
        dataset_final = datasets[0]
    else:
        indices = range(len(datasets))
        lens = []
        for i in range(len(datasets)):
            cnt = get_dataset_len(datasets[i])
            lens.append(cnt)
        res = sorted(zip(indices, lens), key=lambda x: x[1])

        # res = sorted(zip(indices, datasets), key=lambda x: len(x[1]))
        indices = [item[0] for item in res]
        datasets = [datasets[item[0]] for item in res]
        dataset_names = [dataset_names[index] for index in indices]
        # Keep the lengths aligned with the reordered datasets.
        lens = [item[1] for item in res]
        cnt = 0 if not data_lens else sum(data_lens.values())
        dataset_cnt_info = {}
        if cnt > nsamples:
            cnt = 0

        # Datasets are visited from smallest to largest, so dividing the remaining
        # budget by the number of datasets still to fill lets later (larger) datasets
        # take up what earlier (smaller) ones could not provide.
        remaining = sum(1 for dataset_name in dataset_names if dataset_name.split(":")[0] not in data_lens)
        for i in range(len(datasets)):
            name = dataset_names[i].split(":")[0]
            if name not in data_lens:
                target_cnt = (nsamples - cnt) // remaining
                remaining -= 1
                target_cnt = min(target_cnt, lens[i])
                cnt += target_cnt
            else:
                target_cnt = data_lens[name]
            datasets[i] = select_dataset(datasets[i], range(target_cnt))
            dataset_cnt_info[name] = target_cnt
        if len(datasets) > 1:
            from datasets import concatenate_datasets

            # Source metadata is not part of the calibration input and can have
            # incompatible schemas. Restrict columns only for the merge operation.
            datasets = [dataset.select_columns(["input_ids", "attention_mask"]) for dataset in datasets]
            dataset_final = concatenate_datasets(datasets)
            dataset_final = dataset_final.shuffle(seed=seed)
            logger.info(dataset_cnt_info)
        else:
            dataset_final = datasets[0]

    if len(dataset_final) > nsamples:
        dataset_final = select_dataset(dataset_final, range(nsamples))
    return dataset_final


def get_dataset(tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", seed=42, nsamples=512, trust_remote_code=False):
    """Generate a dataset for calibration.

    Uses a subprocess for preprocessing to ensure all temporary memory is fully
    reclaimed by the OS when the subprocess exits.  The HuggingFace ``datasets``
    library automatically caches intermediate results (e.g. ``.map()``,
    ``.filter()``), so the main process can reload them cheaply after the
    subprocess finishes.

    Set environment variable ``AR_DISABLE_DATASET_SUBPROCESS=1`` to disable
    subprocess mode and run preprocessing in the main process.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        seqlen (int): The exact sequence length.
        dataset_name (str, optional): Dataset name(s) separated by commas, or a
            ``CalibDataset`` object. Defaults to "NeelNanda/pile-10k".
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        nsamples (int, optional): Total number of samples to include. Defaults to 512.

    Returns:
        Dataset: The processed dataset ready for calibration.
    """
    # Normalize CalibDataset objects to spec strings
    if isinstance(dataset_name, CalibDataset):
        dataset_name = dataset_name.to_spec_string()
    elif isinstance(dataset_name, (list, tuple)):
        dataset_name = normalize_dataset_spec(dataset_name)

    # Allow disabling subprocess mode via environment variable
    if envs.AR_DISABLE_DATASET_SUBPROCESS:
        try:
            return _get_dataset_impl(tokenizer, seqlen, dataset_name, seed, nsamples, trust_remote_code)
        except Exception as error:
            return _fallback_to_fineweb_edu(error, tokenizer, seqlen, dataset_name, seed, nsamples, trust_remote_code)

    # Run preprocessing in a subprocess so all temporary memory is freed on exit.
    logger.info("Preprocessing calibration dataset in a subprocess to avoid memory leaks...")

    subprocess_network_error = None
    with tempfile.TemporaryDirectory(prefix="auto_round_calib_") as temp_dir:
        try:
            if os.name == "nt":
                raise OSError("fork is not available on Windows")

            # macOS crashes with SIGSEGV (exit code -11) when using "fork" after threads
            # have been started (PyTorch, tokenizers, and HuggingFace datasets all use
            # threads).  Use "spawn" on macOS, which is safe but requires pickling args.
            mp_context = "spawn" if sys.platform == "darwin" else "fork"
            ctx = multiprocessing.get_context(mp_context)
            result_queue = ctx.Queue()
            p = ctx.Process(
                target=_preprocess_dataset_in_subprocess,
                args=(
                    result_queue,
                    tokenizer,
                    seqlen,
                    dataset_name,
                    seed,
                    nsamples,
                    os.path.join(temp_dir, "dataset"),
                    trust_remote_code,
                ),
            )
            p.start()
            p.join()

            try:
                result_status, result_value = result_queue.get(timeout=1)
            except queue.Empty:
                result_status, result_value = None, None
            result_queue.close()
            result_queue.join_thread()
            if p.exitcode != 0:
                if result_status == _DATASET_RESULT_ERROR and result_value is not None:
                    subprocess_network_error = ConnectionError(result_value)
                else:
                    raise RuntimeError(f"Dataset preprocessing subprocess exited with code {p.exitcode}")
            elif result_status == _DATASET_RESULT_SUCCESS:
                return load_from_disk(result_value, keep_in_memory=True)
            else:
                raise RuntimeError("Dataset preprocessing subprocess returned no dataset")

        except Exception as e:
            logger.warning(f"Subprocess dataset preprocessing failed ({e}), falling back to in-process mode.")

    if subprocess_network_error is not None:
        return _fallback_to_fineweb_edu(
            subprocess_network_error, tokenizer, seqlen, dataset_name, seed, nsamples, trust_remote_code
        )

    try:
        return _get_dataset_impl(tokenizer, seqlen, dataset_name, seed, nsamples, trust_remote_code)
    except Exception as error:
        return _fallback_to_fineweb_edu(error, tokenizer, seqlen, dataset_name, seed, nsamples, trust_remote_code)


def get_dataloader(
    tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", seed=42, bs=8, nsamples=512, trust_remote_code=False
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
        apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
        DataLoader: The DataLoader for the calibrated dataset.
    """

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

    dataset_final = get_dataset(tokenizer, seqlen, dataset_name, seed, nsamples, trust_remote_code)
    calib_dataloader = DataLoader(dataset_final, batch_size=bs, shuffle=False, collate_fn=collate_batch)
    return calib_dataloader
