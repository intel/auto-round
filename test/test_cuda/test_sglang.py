import shutil
import sys
from pathlib import Path

import pytest
import torch
import sglang as sgl
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for _ in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model_name():
    return "/models/opt-125m"


@pytest.fixture(scope="session")
def save_dir(tmp_path_factory):
    # pytest-managed temp directory
    return tmp_path_factory.mktemp("autoround_saved")


@pytest.fixture(scope="session")
def model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    return model, tokenizer


@pytest.fixture(scope="session")
def llm_dataloader():
    return LLMDataLoader()


@pytest.fixture(autouse=True)
def cleanup():
    """
    Auto cleanup after each test
    """
    yield
    shutil.rmtree("runs", ignore_errors=True)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def _run_sglang_inference(model_path: Path):
    llm = sgl.Engine(model_path=str(model_path), mem_fraction_static=0.7)
    prompts = ["Hello, my name is"]
    sampling_params = {"temperature": 0.6, "top_p": 0.95}
    outputs = llm.generate(prompts, sampling_params)
    return outputs[0]["text"]


def test_ar_format_sglang(model_name, save_dir, llm_dataloader):
    autoround = AutoRound(
        model_name,
        scheme="W4A16",
        iters=2,
        seqlen=2,
        dataset=llm_dataloader,
    )

    autoround.quantize_and_save(
        output_dir=save_dir,
        inplace=True,
        format="auto_round",
    )

    generated_text = _run_sglang_inference(save_dir)
    print(generated_text)

    assert "!!!" not in generated_text

    shutil.rmtree(save_dir, ignore_errors=True)


def test_mixed_ar_format_sglang(model_name, save_dir, llm_dataloader):
    layer_config = {
        "self_attn": {"bits": 16, "act_bits": 16},
        "lm_head": {"bits": 16, "act_bits": 16},
        "fc1": {"bits": 16, "act_bits": 16},
    }

    autoround = AutoRound(
        model_name,
        scheme="W4A16",
        iters=2,
        seqlen=2,
        dataset=llm_dataloader,
        layer_config=layer_config,
    )

    autoround.quantize_and_save(
        output_dir=save_dir,
        inplace=True,
        format="auto_round",
    )

    generated_text = _run_sglang_inference(save_dir)
    print(generated_text)

    assert "!!!" not in generated_text

    shutil.rmtree(save_dir, ignore_errors=True)

