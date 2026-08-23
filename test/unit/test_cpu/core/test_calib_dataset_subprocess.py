"""Regression tests for subprocess dataset preprocessing (#1890).

On macOS, ``multiprocessing.get_context("fork")`` triggers SIGSEGV (exit -11)
because PyTorch and tokenizers start threads before the subprocess is spawned.
The fix selects ``"spawn"`` on macOS and ``"fork"`` on Linux.
"""

import os


class _FakeProcess:
    """Minimal subprocess stub: starts, joins, and exits cleanly."""

    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def join(self):
        pass

    exitcode = 0


def _fake_get_context(captured):
    """Return a factory that records the requested multiprocessing context name."""

    def get_context(method):
        captured["method"] = method

        class _FakeCtx:
            Process = _FakeProcess

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
