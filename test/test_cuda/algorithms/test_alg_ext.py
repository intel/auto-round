import shutil
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...helpers import evaluate_accuracy, get_model_path

AUTO_ROUND_PATH = __file__.split("/")
AUTO_ROUND_PATH = "/".join(AUTO_ROUND_PATH[: AUTO_ROUND_PATH.index("test")])


class TestAlgExt:

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_folder = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_folder, ignore_errors=True)

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        yield
        shutil.rmtree("runs", ignore_errors=True)

    def test_gguf_q2_k_s_uses_dq_wrapper_block(self, tiny_qwen_model_path):
        """Regression test: enable_alg_ext + gguf:q2_k_s must use DQWrapperLinear.

        gguf:q2_k_s overrides data_type to "int_asym_dq" at format-resolution
        time.  The quantizer must be created *after* that override so that
        wrapper_autoround() sees the final data_type and sets dq_wrapper_block
        (which wraps layers with DQWrapperLinear) instead of falling back to
        the plain wrapper_block (which produces WrapperLinear).
        """
        from auto_round.alg_ext import dq_wrapper_block

        ar = AutoRound(
            tiny_qwen_model_path,
            bits=4,
            format="gguf:q2_k_s",
            iters=1,
            nsamples=1,
            seqlen=32,
            enable_alg_ext=True,
        )
        # post_init() runs the full pipeline (resolve_scheme → resolve_formats →
        # create_quantizer → ...).  quantizer only exists afterwards.
        ar.post_init()

        assert ar.quantizer.wrapper_block.__name__ == dq_wrapper_block.__name__, (
            f"Expected wrapper_block to be '{dq_wrapper_block.__name__}', "
            f"got '{ar.quantizer.wrapper_block.__name__}'. "
            "This likely means the quantizer was created before GGUF format "
            "overrides were applied (data_type was not yet 'int_asym_dq')."
        )

    def test_int2_g64_asym_enable_alg_ext_keeps_config(self, tiny_qwen_model_path):
        """Regression test: asym int2/g64 keeps the requested tuning config."""

        ar = AutoRound(
            tiny_qwen_model_path,
            bits=2,
            group_size=64,
            sym=False,
            iters=1,
            nsamples=1,
            seqlen=32,
            enable_alg_ext=True,
            enable_minmax_tuning=False,
            enable_norm_bias_tuning=True,
            enable_quanted_input=False,
        )
        ar.post_init()

        assert ar.quantizer.bits == 2
        assert ar.quantizer.group_size == 64
        assert ar.quantizer.sym is False
        assert ar.quantizer.enable_alg_ext is True
        assert ar.quantizer.enable_minmax_tuning is False
        assert ar.quantizer.enable_norm_bias_tuning is True
        assert ar.quantizer.enable_quanted_input is False

        ar.quantize()

        assert ar.quantizer.enable_minmax_tuning is False
        assert ar.quantizer.enable_norm_bias_tuning is True
        assert ar.quantizer.enable_quanted_input is False

    @pytest.mark.parametrize("scheme", ["MXFP4", "NVFP4", "W2A16G64", "gguf:q2_k_s,gguf:q4_k_s"])
    def test_all_support_dtype(self, scheme, tiny_qwen_model_path):
        from auto_round.auto_scheme import AutoScheme

        avg_bits = 2 if scheme == "W2A16G64" else 4
        scheme = AutoScheme(options=scheme, avg_bits=avg_bits, ignore_scale_zp_bits=True)
        ar = AutoRound(
            tiny_qwen_model_path,
            scheme=scheme,
            iters=1,
            nsamples=1,
            seqlen=32,
            enable_alg_ext=True,
            enable_torch_compile=True,
        )
        ar.quantize()

    @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    @pytest.mark.skipif(reason="Time-consuming for accuracy evaluation")
    def test_2bits(self):
        model_name = get_model_path("facebook/opt-125m")
        ar = AutoRound(model=model_name, bits=2, group_size=64, enable_alg_ext=True)
        _, quantized_model_path = ar.quantize_and_save(self.save_folder)
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        evaluate_accuracy(model, tokenizer, threshold=0.22, batch_size=64)

    @pytest.mark.skip_ci(reason="Not necessary to test all case in CI")
    def test_cli(self, tiny_opt_model_path):
        import os

        python_path = sys.executable

        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' CUDA_VISIBLE_DEVICES=0 {python_path} -m auto_round --model {tiny_opt_model_path} --iters 1 --device auto --enable_alg_ext --disable_minmax_tuning --disable_quanted_input --avg_bits 2 --options=W2A16,W4A16 --ignore_scale_zp_bits --nsamples 1 --seqlen 32"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' CUDA_VISIBLE_DEVICES=0 {python_path} -m auto_round --model {tiny_opt_model_path} --iters 1 --device auto --enable_alg_ext --avg_bits 5.5 --options=mxfp4,mxfp8 --ignore_scale_zp_bits --enable_torch_compile --nsamples 1 --seqlen 32"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
