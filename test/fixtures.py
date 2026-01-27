import os
import shutil

import datasets
import pytest
import torch
import transformers

from .helpers import (
    DataLoader,
    deepseek_v2_name_or_path,
    gemma_name_or_path,
    get_tiny_model,
    gptj_name_or_path,
    lamini_name_or_path,
    opt_name_or_path,
    phi2_name_or_path,
    qwen_2_5_vl_name_or_path,
    qwen_moe_name_or_path,
    qwen_name_or_path,
    qwen_vl_name_or_path,
    save_tiny_model,
)

datasets.original_load_dataset = datasets.load_dataset


def patch_load_dataset(*args, **kwargs):
    if len(args) > 0 and "openbookqa" in args[0]:
        args = ("allenai/openbookqa",) + args[1:]
    if "path" in kwargs:
        if "openbookqa" in kwargs["path"] and "allenai/openbookqa" not in kwargs["path"]:
            kwargs["path"] = kwargs["path"].replace("openbookqa", "allenai/openbookqa")
    if "name" in kwargs:
        if "openbookqa" in kwargs["name"] and "allenai/openbookqa" not in kwargs["name"]:
            kwargs["name"] = kwargs["name"].replace("openbookqa", "allenai/openbookqa")
    return datasets.original_load_dataset(*args, **kwargs)


datasets.load_dataset = patch_load_dataset


# Create tiny model path fixtures for testing
@pytest.fixture(scope="session")
def tiny_opt_model_path():
    model_name_or_path = opt_name_or_path
    tiny_model_path = "./tmp/tiny_opt_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_lamini_model_path():
    model_name_or_path = lamini_name_or_path
    tiny_model_path = "./tmp/tiny_lamini_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_gptj_model_path():
    model_name_or_path = gptj_name_or_path
    tiny_model_path = "./tmp/tiny_gptj_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_phi2_model_path():
    model_name_or_path = phi2_name_or_path
    tiny_model_path = "./tmp/tiny_phi2_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_deepseek_v2_model_path():
    model_name_or_path = deepseek_v2_name_or_path
    tiny_model_path = "./tmp/tiny_deepseek_v2_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_gemma_model_path():
    model_name_or_path = gemma_name_or_path
    tiny_model_path = "./tmp/tiny_gemma_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_qwen_model_path():
    model_name_or_path = qwen_name_or_path
    tiny_model_path = "./tmp/tiny_qwen_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_untied_qwen_model_path():
    model_name_or_path = qwen_name_or_path
    tiny_model_path = "./tmp/tiny_untied_qwen_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, force_untie=True)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_qwen_moe_model_path():
    model_name_or_path = qwen_moe_name_or_path
    tiny_model_path = "./tmp/tiny_qwen_moe_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_qwen_vl_model_path():
    model_name_or_path = qwen_vl_name_or_path
    tiny_model_path = "./tmp/tiny_qwen_vl_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2, is_mllm=True)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_qwen_2_5_vl_model_path():
    model_name_or_path = qwen_2_5_vl_name_or_path
    tiny_model_path = "./tmp/tiny_qwen_2_5_vl_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2, is_mllm=True)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(autouse=True, scope="session")
def clean_tmp_model_folder():
    yield
    shutil.rmtree("./tmp", ignore_errors=True)  # unittest default workspace
    shutil.rmtree("./tmp_autoround", ignore_errors=True)  # autoround default workspace


# Create objective fixtures for testing
@pytest.fixture(scope="function")
def tiny_opt_model():
    model_name_or_path = opt_name_or_path
    return get_tiny_model(model_name_or_path, num_layers=2)


@pytest.fixture(scope="function")
def opt_model():
    model_name_or_path = opt_name_or_path
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype="auto", trust_remote_code=True)
    return model


@pytest.fixture(scope="session")
def opt_tokenizer():
    model_name_or_path = opt_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    return tokenizer


@pytest.fixture(scope="function")
def model():
    model_name_or_path = opt_name_or_path
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype="auto", trust_remote_code=True)
    return model


@pytest.fixture(scope="session")
def tokenizer():
    model_name_or_path = opt_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    return tokenizer


@pytest.fixture(scope="session")
def dataloader():
    return DataLoader()
