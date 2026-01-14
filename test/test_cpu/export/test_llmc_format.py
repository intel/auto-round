import shutil

import pytest

from auto_round import AutoRound


@pytest.mark.parametrize("scheme", ["NVFP4", "MXFP4", "FPW8A16", "FP8_STATIC", "MXFP8"])
# TODO: FP8_DYNAMIC
def test_export_format(tiny_opt_model_path, scheme):
    autoround = AutoRound(
        tiny_opt_model_path,
        iters=0,
        scheme=scheme,
    )
    autoround.quantize_and_save("temp_model_path", format="llm_compressor")
    shutil.rmtree("temp_model_path", ignore_errors=True)


def test_alias_export_format(tiny_opt_model_path):
    autoround = AutoRound(
        tiny_opt_model_path,
        scheme="FP8_STATIC",
        iters=0,
    )
    autoround.quantize_and_save("temp_model_path_alias", format="compressed_tensors")
    # shutil.rmtree("temp_model_path", ignore_errors=True)
