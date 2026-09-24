"""Regression tests for subprocess dataset preprocessing (#1890).

On macOS, ``multiprocessing.get_context("fork")`` triggers SIGSEGV (exit -11)
because PyTorch and tokenizers start threads before the subprocess is spawned.
The fix selects ``"spawn"`` on macOS and ``"fork"`` on Linux.
"""

import os

import pytest
from datasets import Dataset


class _FakeProcess:
    """Minimal subprocess stub: starts, joins, and exits cleanly."""

    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def join(self):
        pass

    exitcode = 0


class _FakeQueue:
    def get(self, timeout=None):
        raise __import__("queue").Empty

    def close(self):
        pass

    def join_thread(self):
        pass


def _fake_get_context(captured):
    """Return a factory that records the requested multiprocessing context name."""

    def get_context(method):
        captured["method"] = method

        class _FakeCtx:
            Process = _FakeProcess
            Queue = _FakeQueue

        return _FakeCtx()

    return get_context


def test_mac_uses_spawn_context(monkeypatch):
    """On macOS, get_dataset must request the ``spawn`` multiprocessing context."""
    import auto_round.calib_dataset as cd

    captured = {}
    monkeypatch.setattr(cd.multiprocessing, "get_context", _fake_get_context(captured))
    monkeypatch.setattr(cd.sys, "platform", "darwin")
    monkeypatch.setattr(cd.os, "name", "posix")
    monkeypatch.setattr(cd, "_get_dataset_impl", lambda *a, **kw: None)
    monkeypatch.setattr(cd.envs, "AR_DISABLE_DATASET_SUBPROCESS", False)

    cd.get_dataset(tokenizer=None, seqlen=512)

    assert captured.get("method") == "spawn", f"expected 'spawn' on macOS, got {captured.get('method')!r}"


def test_linux_uses_fork_context(monkeypatch):
    """On Linux, get_dataset must request the ``fork`` multiprocessing context."""
    import auto_round.calib_dataset as cd

    captured = {}
    monkeypatch.setattr(cd.multiprocessing, "get_context", _fake_get_context(captured))
    monkeypatch.setattr(cd.sys, "platform", "linux")
    monkeypatch.setattr(cd.os, "name", "posix")
    monkeypatch.setattr(cd, "_get_dataset_impl", lambda *a, **kw: None)
    monkeypatch.setattr(cd.envs, "AR_DISABLE_DATASET_SUBPROCESS", False)

    cd.get_dataset(tokenizer=None, seqlen=512)

    assert captured.get("method") == "fork", f"expected 'fork' on Linux, got {captured.get('method')!r}"


def test_windows_falls_back_to_inprocess(monkeypatch):
    """On Windows (os.name == 'nt'), subprocess is skipped and in-process runs."""
    import auto_round.calib_dataset as cd

    inprocess_called = []
    monkeypatch.setattr(cd.os, "name", "nt")
    monkeypatch.setattr(cd, "_get_dataset_impl", lambda *a, **kw: inprocess_called.append(True))
    monkeypatch.setattr(cd.envs, "AR_DISABLE_DATASET_SUBPROCESS", False)

    cd.get_dataset(tokenizer=None, seqlen=512)

    assert inprocess_called, "in-process fallback should have been called on Windows"


def test_subprocess_network_error_falls_back_without_retrying_source(monkeypatch):
    """A child-process network error must switch directly to FineWeb-Edu."""
    import auto_round.calib_dataset as cd

    class _NetworkErrorQueue:
        def get(self, timeout=None):
            return cd._DATASET_RESULT_ERROR, "simulated proxy failure"

        def close(self):
            pass

        def join_thread(self):
            pass

    class _NetworkErrorProcess(_FakeProcess):
        exitcode = 1

    class _NetworkErrorContext:
        Process = _NetworkErrorProcess
        Queue = _NetworkErrorQueue

    fallback_dataset = Dataset.from_dict({"input_ids": [[1]], "attention_mask": [[1]]})
    fallback = []

    def fallback_to_fineweb(error, tokenizer, seqlen, dataset_name, seed, nsamples):
        fallback.append((error, dataset_name))
        return fallback_dataset

    monkeypatch.setattr(cd.multiprocessing, "get_context", lambda method: _NetworkErrorContext())
    monkeypatch.setattr(cd.os, "name", "posix")
    monkeypatch.setattr(cd.sys, "platform", "linux")
    monkeypatch.setattr(cd.envs, "AR_DISABLE_DATASET_SUBPROCESS", False)
    monkeypatch.setattr(cd, "_fallback_to_fineweb_edu", fallback_to_fineweb)
    monkeypatch.setattr(cd, "_get_dataset_impl", lambda *args: pytest.fail("source retried"))

    result = cd.get_dataset(tokenizer=None, seqlen=128, dataset_name="source")

    assert result is fallback_dataset
    assert isinstance(fallback[0][0], ConnectionError)
    assert fallback[0][1] == "source"


@pytest.mark.skipif(os.name == "nt", reason="fork is unavailable on Windows")
def test_subprocess_returns_saved_dataset_without_reprocessing(monkeypatch, tmp_path):
    import auto_round.calib_dataset as cd

    parent_pid = os.getpid()
    output_path = tmp_path / "output_path"
    original_save = Dataset.save_to_disk

    def save_dataset(dataset, path):
        output_path.write_text(path)
        return original_save(dataset, path)

    def build_dataset(*args):
        assert os.getpid() != parent_pid, "dataset rebuilt in parent process"
        return Dataset.from_dict({"input_ids": [[1, 2]], "attention_mask": [[1, 1]]})

    monkeypatch.setattr(cd, "_get_dataset_impl", build_dataset)
    monkeypatch.setattr(cd.Dataset, "save_to_disk", save_dataset)
    monkeypatch.setattr(cd.envs, "AR_DISABLE_DATASET_SUBPROCESS", False)

    result = cd.get_dataset(tokenizer=None, seqlen=2)

    assert result[0]["input_ids"] == [1, 2]
    assert not os.path.exists(output_path.read_text())
    assert result[0]["attention_mask"] == [1, 1]
