import pytest
import torch


@pytest.fixture(autouse=True)
def disable_torch_compile(monkeypatch):
    """Skip torch.compile overhead in CPU tests."""
    monkeypatch.setattr(torch, "compile", lambda function, *args, **kwargs: function)
