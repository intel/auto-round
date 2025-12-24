from auto_round import AutoRound


def test_torch_compile():
    ar = AutoRound(model="facebook/opt-125m", scheme="NVFP4", enable_torch_compile=True)
    assert not ar.enable_torch_compile, "NVFP4 cannot work with torch.compile."
    ar = AutoRound(model="facebook/opt-125m", scheme="FP8_STATIC", enable_torch_compile=True)
    assert not ar.enable_torch_compile, "FP8_STATIC cannot work with torch.compile."
