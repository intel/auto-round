import argparse

from auto_round import AutoRound
from auto_round.compressors.entry import AutoRound as NewAutoRound

# def test_argparse_check(tiny_opt_model_path):
#     ar = AutoRound(model=tiny_opt_model_path, scheme="NVFP4", enable_torch_compile=True)
#     assert not ar.enable_torch_compile, "NVFP4 cannot work with torch.compile."
#     ar = AutoRound(model=tiny_opt_model_path, scheme="FP8_STATIC", enable_torch_compile=True)
#     assert not ar.enable_torch_compile, "FP8_STATIC cannot work with torch.compile."
#
#     # Regression for issue #2034: gradient_accumulate_steps must flow from the CLI
#     # args all the way to the quantizer. Previously the CLI path dropped the flag,
#     # so CalibrationState defaulted to 1 regardless of the user's value.
#     steps = 8
#
#     ar = NewAutoRound(
#         tiny_opt_model_path,
#         scheme="W4A16",
#         gradient_accumulate_steps=steps,
#         iters=1,
#         nsamples=1,
#         seqlen=8,
#         low_cpu_mem_usage=False,
#     )
#     ar.post_init()  # triggers _build_quantizer() → bind()
#     assert ar.gradient_accumulate_steps == steps
#     assert ar.quantizer.gradient_accumulate_steps == steps #TODO wenhuach recover
#     # Compressor and quantizer must share exactly the same CalibrationState instance.
#     assert ar.quantizer._calibration_state is ar._calibration_state
