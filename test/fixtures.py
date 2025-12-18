import shutil

import pytest
import torch
import transformers

from .helpers import opt_name_or_path


class DataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


@pytest.fixture(scope="session")
def tiny_opt_model_path():
    tiny_opt_model_path = "./tmp_tiny_opt_model_path"
    model = transformers.AutoModelForCausalLM.from_pretrained(opt_name_or_path, dtype="auto", trust_remote_code=True)
    model.config.num_hidden_layers = 3
    setattr(model.model.decoder, "layers", model.model.decoder.layers[:3])
    tokenizer = transformers.AutoTokenizer.from_pretrained(opt_name_or_path, trust_remote_code=True)
    model.save_pretrained(tiny_opt_model_path)
    tokenizer.save_pretrained(tiny_opt_model_path)
    print("[Fixture]: built tiny model path for testing in session")
    yield tiny_opt_model_path
    shutil.rmtree(tiny_opt_model_path)


@pytest.fixture(scope="function")
def tiny_opt_model():
    model = transformers.AutoModelForCausalLM.from_pretrained(opt_name_or_path, dtype="auto", trust_remote_code=True)
    model.config.num_hidden_layers = 3
    setattr(model.model.decoder, "layers", model.model.decoder.layers[:3])
    return model


@pytest.fixture(scope="function")
def tiny_opt_model():
    model = transformers.AutoModelForCausalLM.from_pretrained(opt_name_or_path, dtype="auto", trust_remote_code=True)
    model.config.num_hidden_layers = 3
    setattr(model.model.decoder, "layers", model.model.decoder.layers[:3])
    return model


@pytest.fixture(scope="function")
def opt_model():
    model = transformers.AutoModelForCausalLM.from_pretrained(opt_name_or_path, dtype="auto", trust_remote_code=True)
    return model


@pytest.fixture(scope="session")
def opt_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(opt_name_or_path, trust_remote_code=True)
    return tokenizer


@pytest.fixture(scope="session")
def dataloader():
    return DataLoader()
