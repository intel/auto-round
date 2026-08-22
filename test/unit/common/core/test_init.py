import inspect
import logging

from auto_round import AutoRound
from auto_round.auto_scheme import AutoScheme
from auto_round.cli.parser import build_quantize_parser
from auto_round.compressors.base import BaseOrchestrator
from auto_round.utils.device_manager import default_enable_torch_compile


def test_torch_compile_platform_default_is_deferred():
    assert inspect.signature(AutoRound.__new__).parameters["enable_torch_compile"].default is None
    assert inspect.signature(BaseOrchestrator.__init__).parameters["enable_torch_compile"].default is None


def test_dataset_default_is_deferred():
    assert inspect.signature(AutoRound.__new__).parameters["dataset"].default is None
    assert inspect.signature(BaseOrchestrator.__init__).parameters["dataset"].default is None


def test_cli_torch_compile_flags():
    parser = build_quantize_parser()
    assert parser.parse_args(["--model", "test-model"]).enable_torch_compile is None
    assert parser.parse_args(["--model", "test-model", "--enable_torch_compile"]).enable_torch_compile is True
    assert parser.parse_args(["--model", "test-model", "--disable_torch_compile"]).enable_torch_compile is False


def test_cli_dataset_tracks_explicit_value():
    parser = build_quantize_parser()
    assert parser.parse_args(["--model", "test-model"]).dataset is None
    assert (
        parser.parse_args(["--model", "test-model", "--dataset", "NeelNanda/pile-10k"]).dataset == "NeelNanda/pile-10k"
    )


def test_auto_scheme_inherits_torch_compile_setting():
    auto_scheme = AutoScheme(avg_bits=4.0, options=["W4A16"])
    assert auto_scheme.enable_torch_compile is None


def test_xpu_torch_compile_is_disabled_by_default():
    assert not default_enable_torch_compile("xpu", platform_name="linux")
    assert default_enable_torch_compile("cuda", platform_name="linux")


def test_torch_compile_runtime_defaults(tiny_opt_model_path):
    ar = AutoRound(model=tiny_opt_model_path, scheme="W4A16", iters=0, nsamples=1)
    assert ar.enable_torch_compile == default_enable_torch_compile(ar.device)

    ar = AutoRound(
        model=tiny_opt_model_path,
        scheme="W4A16",
        iters=0,
        nsamples=1,
        enable_torch_compile=False,
    )
    assert not ar.enable_torch_compile

    ar = AutoRound(model=tiny_opt_model_path, scheme="NVFP4", iters=0, nsamples=1)
    assert ar.enable_torch_compile == default_enable_torch_compile(ar.device)


def test_torch_compile_windows_defaults(monkeypatch, caplog, tiny_opt_model_path):
    monkeypatch.setattr("auto_round.compressors.base.sys.platform", "win32")

    with caplog.at_level(logging.WARNING):
        ar = AutoRound(model=tiny_opt_model_path, scheme="W4A16", iters=0, nsamples=1)
    assert not ar.enable_torch_compile
    if str(ar.device).split(":", 1)[0] == "xpu":
        assert "disabled by default on XPU" in caplog.text
    else:
        assert "disabled by default on Windows" in caplog.text
        assert "cl.exe" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ar = AutoRound(
            model=tiny_opt_model_path,
            scheme="W4A16",
            iters=0,
            nsamples=1,
            enable_torch_compile=True,
        )
    assert ar.enable_torch_compile
    assert "Forcing `torch.compile` on Windows" in caplog.text

    ar = AutoRound(
        model=tiny_opt_model_path,
        scheme="W4A16",
        iters=0,
        nsamples=1,
        enable_torch_compile=False,
    )
    assert not ar.enable_torch_compile
