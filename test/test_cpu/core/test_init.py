import inspect

from auto_round import AutoRound
from auto_round.auto_scheme import AutoScheme
from auto_round.cli.parser import build_quantize_parser
from auto_round.compressors.base import BaseOrchestrator
from auto_round.compressors.entry import AutoRound as NewAutoRound
from auto_round.compressors.entry import AutoRoundCompatible


def test_torch_compile_enabled_by_default():
    assert inspect.signature(AutoRound.__new__).parameters["enable_torch_compile"].default is True
    assert inspect.signature(NewAutoRound.__new__).parameters["enable_torch_compile"].default is True
    assert inspect.signature(AutoRoundCompatible.__new__).parameters["enable_torch_compile"].default is True
    assert inspect.signature(BaseOrchestrator.__init__).parameters["enable_torch_compile"].default is True


def test_cli_torch_compile_flags():
    parser = build_quantize_parser()
    assert parser.parse_args(["--model", "test-model"]).enable_torch_compile is True
    assert parser.parse_args(["--model", "test-model", "--enable_torch_compile"]).enable_torch_compile is True
    assert parser.parse_args(["--model", "test-model", "--disable_torch_compile"]).enable_torch_compile is False


def test_auto_scheme_inherits_torch_compile_setting():
    auto_scheme = AutoScheme(avg_bits=4.0, options=["W4A16"])
    assert auto_scheme.enable_torch_compile is None


def test_torch_compile_runtime_defaults(tiny_opt_model_path):
    ar = AutoRound(model=tiny_opt_model_path, scheme="W4A16", iters=0, nsamples=1)
    assert ar.enable_torch_compile

    ar = AutoRound(
        model=tiny_opt_model_path,
        scheme="W4A16",
        iters=0,
        nsamples=1,
        enable_torch_compile=False,
    )
    assert not ar.enable_torch_compile

    ar = AutoRound(model=tiny_opt_model_path, scheme="NVFP4", iters=0, nsamples=1)
    assert ar.enable_torch_compile


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

    ar = NewAutoRound(
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
