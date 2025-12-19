import shutil

import pytest
import torch
import transformers

from .helpers import (
    get_tiny_model,
    gptj_name_or_path,
    lamini_name_or_path,
    opt_name_or_path,
    qwen_name_or_path,
)


class DataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


# Create tiny model path fixtures for testing
@pytest.fixture(scope="session")
def tiny_opt_model_path():
    model_name_or_path = opt_name_or_path
    tiny_model_path = "./tmp_tiny_opt_model_path"
    model = get_tiny_model(model_name_or_path, num_layers=3)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.save_pretrained(tiny_model_path)
    tokenizer.save_pretrained(tiny_model_path)
    print(f"[Fixture]: built tiny model path:{tiny_model_path} for testing in session")
    yield tiny_model_path
    shutil.rmtree(tiny_model_path)


@pytest.fixture(scope="session")
def tiny_qwen_model_path():
    model_name_or_path = qwen_name_or_path
    tiny_model_path = "./tmp_tiny_qwen_model_path"
    model = get_tiny_model(model_name_or_path, num_layers=3)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.save_pretrained(tiny_model_path)
    tokenizer.save_pretrained(tiny_model_path)
    print(f"[Fixture]: built tiny model path:{tiny_model_path} for testing in session")
    yield tiny_model_path
    shutil.rmtree(tiny_model_path)


@pytest.fixture(scope="session")
def tiny_lamini_model_path():
    model_name_or_path = lamini_name_or_path
    tiny_model_path = "./tmp_tiny_lamini_model_path"
    model = get_tiny_model(model_name_or_path, num_layers=3)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.save_pretrained(tiny_model_path)
    tokenizer.save_pretrained(tiny_model_path)
    print(f"[Fixture]: built tiny model path:{tiny_model_path} for testing in session")
    yield tiny_model_path
    shutil.rmtree(tiny_model_path)


@pytest.fixture(scope="session")
def tiny_gptj_model_path():
    model_name_or_path = gptj_name_or_path
    tiny_model_path = "./tmp_tiny_gptj_model_path"
    model = get_tiny_model(model_name_or_path, num_layers=3)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.save_pretrained(tiny_model_path)
    tokenizer.save_pretrained(tiny_model_path)
    print(f"[Fixture]: built tiny model path:{tiny_model_path} for testing in session")
    yield tiny_model_path
    shutil.rmtree(tiny_model_path)


# Create objective fixtures for testing
@pytest.fixture(scope="function")
def tiny_opt_model():
    model_name_or_path = opt_name_or_path
    return get_tiny_model(model_name_or_path, num_layers=3)


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


@pytest.fixture(scope="session")
def dataloader():
    return DataLoader()
