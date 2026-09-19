import os
import sys
from typing import Mapping

import pytest

from .fixtures import *

# Easy debugging without installing auto-round.
sys.path.insert(0, "..")

# Workaround: some gguf builds report version 'N/A' which is not PEP 440
# compliant and causes packaging.version.InvalidVersion inside transformers.
try:
    import gguf as _gguf_mod
    from packaging.version import Version

    try:
        Version(_gguf_mod.__version__)
    except Exception:
        _gguf_mod.__version__ = "0.10.0"
except ImportError:
    pass


try:
    import torch

    # When loaded via the "meta" device, `gptqmodel==6.0.3` raises an error because the
    # internal loading process within the `transformers` library defaults to "meta" mode.
    # Importing under a CPU device context avoids that failure during module loading.
    with torch.device("cpu"):
        import gptqmodel  # pylint: disable=E0401
except ImportError:
    pass


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

    config.addinivalue_line(
        "markers",
        "enable_torch_compile: allow this test to use real torch.compile instead of the default no-op patch",
    )

    config.stash[backup_env] = os.environ

    if pytest.mode == "lazy":
        os.environ["PT_HPU_LAZY_MODE"] = "1"
    elif pytest.mode == "compile":
        os.environ["PT_HPU_LAZY_MODE"] = "0"
        os.environ["PT_ENABLE_INT64_SUPPORT"] = "1"


def pytest_unconfigure(config):
    os.environ.clear()
    os.environ.update(config.stash[backup_env])


@pytest.fixture(scope="session", autouse=True)
def _warmup_triton_cuda_driver():
    """Force Triton's CUDA driver singleton to snapshot the *real* device capability.

    Triton's ``GPUDriver.__init__`` (triton/backends/driver.py) does::

        self.get_device_capability = torch.cuda.get_device_capability

    This copies the function object once, when the process-wide
    ``triton.runtime.driver.active`` singleton is first constructed. If that
    first construction happens to occur while some test has patched
    ``torch.cuda.get_device_capability`` (e.g. ``mock_fp8_capable_device``
    faking an FP8-capable device), the mocked return value gets baked in
    *permanently* for the rest of the process -- unaffected by the patch
    being un-applied later. Every subsequent Triton kernel is then compiled
    for that fake architecture, causing
    ``RuntimeError: Triton Error [CUDA]: no kernel image is available for
    execution on the device`` on the *real* hardware regardless of its actual
    compute capability (reproduced on both sm80 and sm120).
    See https://github.com/intel/auto-round/issues/2048.

    Triggering the singleton construction here -- as an autouse, session
    scoped fixture that pytest always instantiates before any function scoped
    fixture -- guarantees it captures the real capability before any test
    gets a chance to mock it.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        import triton

        triton.runtime.driver.active.get_current_target()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def disable_torch_compile_by_default(request, monkeypatch):
    """Use a no-op torch.compile by default to reduce test overhead and flakiness.

    Mark tests with ``@pytest.mark.enable_torch_compile`` to opt in to real torch.compile.
    """
    if request.node.get_closest_marker("enable_torch_compile"):
        return

    try:
        import torch
    except Exception:
        return

    monkeypatch.setattr(torch, "compile", lambda function, *args, **kwargs: function, raising=False)
