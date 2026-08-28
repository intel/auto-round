import inspect
import logging

from auto_round import AutoRound
from auto_round.auto_scheme import AutoScheme
from auto_round.cli.parser import build_quantize_parser
from auto_round.compressors.base import MIN_ITERS_FOR_TORCH_COMPILE, BaseOrchestrator
from auto_round.logger import logger
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


def test_xpu_torch_compile_is_enabled_by_default():
    assert default_enable_torch_compile("xpu", platform_name="linux")
    assert default_enable_torch_compile("cuda", platform_name="linux")


def test_torch_compile_runtime_defaults(tiny_opt_model_path):
    ar = AutoRound(model=tiny_opt_model_path, scheme="W4A16", iters=200, nsamples=1)
    assert ar.enable_torch_compile == default_enable_torch_compile(ar.device)

    ar = AutoRound(
        model=tiny_opt_model_path,
        scheme="W4A16",
        iters=200,
        nsamples=1,
        enable_torch_compile=False,
    )
    assert not ar.enable_torch_compile

    ar = AutoRound(model=tiny_opt_model_path, scheme="NVFP4", iters=200, nsamples=1)
    assert ar.enable_torch_compile == default_enable_torch_compile(ar.device)


def _assert_compile(ar, expected: bool):
    """Assert the effective torch.compile flag, tolerating the model-free path.

    ``ModelFreeCompressor`` has no ``compress_context``; the regular compressor
    mirrors ``enable_torch_compile`` onto it, so check both when present.
    """
    assert ar.enable_torch_compile is expected
    compress_context = getattr(ar, "compress_context", None)
    if compress_context is not None:
        assert compress_context.enable_torch_compile is expected


def test_torch_compile_disabled_for_rtn_and_short_signround(tiny_opt_model_path, monkeypatch):
    """RTN / opt-RTN and `iters` < 10 never amortize the torch.compile cost."""
    # Pin the platform default to "enabled" so the algorithm heuristic is observable
    # regardless of the host device / OS.
    monkeypatch.setattr("auto_round.compressors.base.default_enable_torch_compile", lambda *a, **k: True)

    # SignRound with too few iterations.
    for iters in (1, MIN_ITERS_FOR_TORCH_COMPILE - 1):
        ar = AutoRound(model=tiny_opt_model_path, scheme="W4A16", iters=iters, nsamples=1)
        _assert_compile(ar, False)

    # Enough iterations to pay back the compilation cost.
    ar = AutoRound(model=tiny_opt_model_path, scheme="W4A16", iters=MIN_ITERS_FOR_TORCH_COMPILE, nsamples=1)
    _assert_compile(ar, True)


def test_explicit_torch_compile_overrides_algorithm_heuristic(tiny_opt_model_path, monkeypatch):
    """An explicit `enable_torch_compile` is always honored, even for RTN / tiny iters."""
    monkeypatch.setattr("auto_round.compressors.base.default_enable_torch_compile", lambda *a, **k: False)

    for kwargs in ({"iters": 0}, {"iters": 1}, {"iters": 0, "disable_opt_rtn": True}):
        ar = AutoRound(
            model=tiny_opt_model_path,
            scheme="W4A16",
            nsamples=1,
            enable_torch_compile=True,
            **kwargs,
        )
        _assert_compile(ar, True)


def test_torch_compile_state_is_always_logged(tiny_opt_model_path, monkeypatch, caplog):
    """`post_init` reports the final torch.compile decision on every run."""
    monkeypatch.setattr("auto_round.compressors.base.default_enable_torch_compile", lambda *a, **k: True)
    # AutoRound's logger uses a private handler with propagation disabled.
    monkeypatch.setattr(logger, "propagate", True)

    # RTN turns compilation off, and says why.
    with caplog.at_level(logging.INFO):
        ar = AutoRound(model=tiny_opt_model_path, scheme="W4A16", iters=0, nsamples=1)
        ar.post_init()
    assert "`torch.compile` is disabled" in caplog.text
    assert "single pass" in caplog.text

    # An explicit opt-in is honored and reported as enabled.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        ar = AutoRound(
            model=tiny_opt_model_path,
            scheme="W4A16",
            iters=0,
            nsamples=1,
            enable_torch_compile=True,
        )
        ar.post_init()
    assert "`torch.compile` is enabled" in caplog.text


def test_torch_compile_kept_for_auto_scheme_with_rtn(tiny_opt_model_path, monkeypatch):
    """AutoScheme's delta-loss pass still relies on torch.compile to save VRAM."""
    monkeypatch.setattr("auto_round.compressors.base.default_enable_torch_compile", lambda *a, **k: True)

    auto_scheme = AutoScheme(avg_bits=4.0, options=["W4A16"])
    ar = AutoRound(model=tiny_opt_model_path, scheme=auto_scheme, iters=0, nsamples=1)
    assert ar.enable_torch_compile


def test_torch_compile_windows_defaults(monkeypatch, caplog, tiny_opt_model_path):
    monkeypatch.setattr("auto_round.compressors.base.sys.platform", "win32")

    with caplog.at_level(logging.WARNING):
        ar = AutoRound(model=tiny_opt_model_path, scheme="W4A16", iters=0, nsamples=1)
    assert not ar.enable_torch_compile
    if str(ar.device).split(":", 1)[0] == "xpu":
        assert "disabled by default on XPU" not in caplog.text
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


def test_argparse_check(tiny_opt_model_path):
    return  # TODO wenhuach
    ar = AutoRound(model=tiny_opt_model_path, scheme="NVFP4", enable_torch_compile=True)
    assert ar.enable_torch_compile, "NVFP4 should preserve the torch.compile setting."
    ar = AutoRound(model=tiny_opt_model_path, scheme="FP8_STATIC", enable_torch_compile=True)
    assert not ar.enable_torch_compile, "FP8_STATIC cannot work with torch.compile."

    # Regression for issue #2034: gradient_accumulate_steps must flow from the CLI
    # args all the way to the quantizer. Previously the CLI path dropped the flag,
    # so CalibrationState defaulted to 1 regardless of the user's value.
    steps = 8

    ar = AutoRound(
        tiny_opt_model_path,
        scheme="W4A16",
        gradient_accumulate_steps=steps,
        iters=1,
        nsamples=1,
        seqlen=8,
        low_cpu_mem_usage=False,
    )
    ar.post_init()  # triggers _build_quantizer() → bind()
    assert ar.gradient_accumulate_steps == steps
    assert ar.quantizer.gradient_accumulate_steps == steps  # TODO wenhuach recover
    # Compressor and quantizer must share exactly the same CalibrationState instance.
    assert ar.quantizer._calibration_state is ar._calibration_state
