import os
from typing import Mapping

import pytest


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
