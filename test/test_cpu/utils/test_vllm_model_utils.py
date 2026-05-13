from types import SimpleNamespace

import pytest

import auto_round.utils.model as model_utils


class _DummyNonVllm:
    pass


def test_is_vllm_model_none_and_non_vllm():
    assert model_utils.is_vllm_model(None) is False
    assert model_utils.is_vllm_model(_DummyNonVllm()) is False


def test_is_vllm_model_with_expected_nested_attr():
    fake_model = object()
    fake_vllm_obj = type("LLM", (), {"__module__": "vllm"})()
    fake_vllm_obj.llm_engine = SimpleNamespace(
        engine_core=SimpleNamespace(
            engine_core=SimpleNamespace(
                model_executor=SimpleNamespace(
                    driver_worker=SimpleNamespace(
                        worker=SimpleNamespace(model_runner=SimpleNamespace(model=fake_model))
                    )
                )
            )
        )
    )

    assert model_utils.is_vllm_model(fake_vllm_obj) is True


def test_is_vllm_model_with_missing_nested_attr_returns_false():
    fake_vllm_obj = type("LLM", (), {"__module__": "vllm"})()
    fake_vllm_obj.llm_engine = SimpleNamespace()

    assert model_utils.is_vllm_model(fake_vllm_obj) is False


def test_vllm_load_model_raises_when_vllm_not_installed(monkeypatch):
    monkeypatch.setattr(model_utils, "check_vllm_installed", lambda: False)

    with pytest.raises(ImportError):
        model_utils.vllm_load_model("dummy-model")
