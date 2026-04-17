import os

from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.compressors_new.entry import _maybe_disable_hpu_eager_pipeline


def test_hpu_fp8_static_eager_guard_sets_env(monkeypatch):
    monkeypatch.delenv("PT_HPU_EAGER_PIPELINE_ENABLE", raising=False)
    monkeypatch.delenv("PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE", raising=False)

    _maybe_disable_hpu_eager_pipeline(RTNConfig(), "FP8_STATIC", "hpu")

    assert os.getenv("PT_HPU_EAGER_PIPELINE_ENABLE") == "0"
    assert os.getenv("PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE") == "0"


def test_hpu_non_fp8_static_does_not_set_env(monkeypatch):
    monkeypatch.delenv("PT_HPU_EAGER_PIPELINE_ENABLE", raising=False)
    monkeypatch.delenv("PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE", raising=False)

    _maybe_disable_hpu_eager_pipeline(RTNConfig(), "W4A16", "hpu")

    assert os.getenv("PT_HPU_EAGER_PIPELINE_ENABLE") is None
    assert os.getenv("PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE") is None


def test_hpu_fp8_static_eager_guard_respects_user_env(monkeypatch):
    monkeypatch.setenv("PT_HPU_EAGER_PIPELINE_ENABLE", "1")
    monkeypatch.setenv("PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE", "1")

    _maybe_disable_hpu_eager_pipeline(RTNConfig(), "FP8_STATIC", "hpu")

    assert os.getenv("PT_HPU_EAGER_PIPELINE_ENABLE") == "1"
    assert os.getenv("PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE") == "1"
