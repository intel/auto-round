from auto_round import AutoRound

from ..helpers import qwen_name_or_path


class TestAlgExt:
    def test_alg_ext(self, tiny_opt_model_path, tiny_qwen_model_path):
        model_name = tiny_opt_model_path
        ar = AutoRound(model_name, scheme="W2A16", iters=1, nsamples=1, enable_alg_ext=True)
        ar.quantize()

        model_name = tiny_qwen_model_path
        ar = AutoRound(model_name, scheme="gguf:q4_k_s", iters=1, nsamples=1, enable_alg_ext=True)
        ar.quantize()

        from auto_round.auto_scheme import AutoScheme

        scheme = AutoScheme(options=["mxfp4", "mxfp8"], avg_bits=5.5, ignore_scale_zp_bits=True)
        model_name = tiny_qwen_model_path
        ar = AutoRound(model_name, scheme=scheme, iters=1, nsamples=1, enable_alg_ext=True, enable_torch_compile=True)
        ar.quantize()

    def test_alg_ext_import(self):
        from auto_round.alg_ext import wrapper_autoround

    def test_all_support_dtype(self, tiny_opt_model_path):
        model_name = tiny_opt_model_path
        for scheme in ["MXFP4", "NVFP4", "W2A16G64"]:
            ar = AutoRound(
                model_name, scheme=scheme, iters=1, nsamples=1, enable_alg_ext=True, enable_torch_compile=True
            )
            ar.quantize()
