import pytest


def is_pytest_mode_compile():
    return pytest.mode == "compile"


def is_pytest_mode_lazy():
    return pytest.mode == "lazy"
