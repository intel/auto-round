import os
import sys
from typing import Mapping

import pytest

from .fixtures import (
    dataloader,
    opt_model,
    opt_tokenizer,
    tiny_gptj_model_path,
    tiny_lamini_model_path,
    tiny_opt_model,
    tiny_opt_model_path,
    tiny_qwen_model_path,
)
from .helpers import model_infer

# Easy debugging without installing auto-round.
sys.path.insert(0, "..")


### HPU related configuration, usage: `pytest --mode=compile/lazy``
def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="lazy",
        help="{compile|lazy}, default lazy. Choose mode to run tests",
    )


backup_env = pytest.StashKey[Mapping]()


def pytest_configure(config):
    pytest.mode = config.getoption("--mode")
    assert pytest.mode.lower() in ["lazy", "compile"]

    config.stash[backup_env] = os.environ

    if pytest.mode == "lazy":
        os.environ["PT_HPU_LAZY_MODE"] = "1"
    elif pytest.mode == "compile":
        os.environ["PT_HPU_LAZY_MODE"] = "0"
        os.environ["PT_ENABLE_INT64_SUPPORT"] = "1"


def pytest_unconfigure(config):
    os.environ.clear()
    os.environ.update(config.stash[backup_env])
