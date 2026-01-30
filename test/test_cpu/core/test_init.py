from auto_round import AutoRound


def test_torch_compile(tiny_opt_model_path):
    ar = AutoRound(model=tiny_opt_model_path, scheme="NVFP4", enable_torch_compile=True)
    ar._post_init()
    assert not ar.enable_torch_compile, "NVFP4 cannot work with torch.compile."
    ar = AutoRound(model=tiny_opt_model_path, scheme="FP8_STATIC", enable_torch_compile=True)
    ar._post_init()
    assert not ar.enable_torch_compile, "FP8_STATIC cannot work with torch.compile."
