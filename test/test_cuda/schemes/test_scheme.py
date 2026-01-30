import os
import shutil

import pytest
from packaging import version

from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme

from ...helpers import get_model_path, save_tiny_model, transformers_version


class TestAutoRound:
    save_dir = "./saved"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    # Tuning tests
    def test_gguf(self, tiny_qwen_model_path):
        ar = AutoRound(tiny_qwen_model_path, scheme="W2A16", nsamples=1, iters=1)
        ar.quantize_and_save(self.save_dir, format="gguf:q4_k_m")
        assert ar.bits == 4
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_w4a16(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="W4A16", nsamples=1, iters=1)
        assert ar.bits == 4
        ar.quantize()

    def test_w2a16(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="W2A16", nsamples=1, iters=1)
        assert ar.bits == 2
        ar.quantize()

    def test_mxfp4(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="MXFP4_RCEIL", nsamples=1, iters=1)
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "mx_fp"
        assert ar.act_data_type == "mx_fp_rceil"
        ar.quantize()

    def test_fp8_static(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="FP8_STATIC", nsamples=1, iters=1)
        assert ar.bits == 8
        assert ar.act_bits == 8
        assert ar.data_type == "fp"
        assert ar.act_data_type == "fp"
        assert ar.group_size == -1
        assert ar.act_dynamic is False
        ar.quantize()

    ## RTN tests
    def test_w2a16_rtn(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="W2A16", nsamples=1, iters=0)
        assert ar.bits == 2
        ar.quantize()

    def test_mxfp4_rtn(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="MXFP4", nsamples=1, iters=0)
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "mx_fp"
        assert ar.act_data_type == "mx_fp"
        ar.quantize()

    def test_fp8_static_rtn(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="FP8_STATIC", nsamples=1, iters=0)
        assert ar.bits == 8
        assert ar.act_bits == 8
        assert ar.data_type == "fp"
        assert ar.act_data_type == "fp"
        assert ar.group_size == -1
        assert ar.act_dynamic is False
        ar.quantize()

    def test_scheme_in_layer_config(self):
        model_path = get_model_path("facebook/opt-125m")
        layer_config = {
            "model.decoder.layers.2.self_attn": {"bits": 2},
            "model.decoder.layers.3.self_attn.v_proj": "W8A16",
            "model.decoder.layers.4.self_attn.k_proj": QuantizationScheme.from_dict({"group_size": 64}),
        }
        ar = AutoRound(model_path, scheme="W3A16", nsamples=1, iters=1, layer_config=layer_config)

        ar.quantize()
        for n, m in ar.model.named_modules():
            if n == "model.decoder.layers.2.self_attn.q_proj":
                assert m.bits == 2
            if n == "model.decoder.layers.2.self_attn.k_proj":
                assert m.bits == 2
            if n == "model.decoder.layers.3.self_attn.v_proj":
                assert m.bits == 8
            if n == "model.decoder.layers.4.self_attn.k_proj":
                assert m.group_size == 64

    @pytest.mark.skipif(
        transformers_version >= version.parse("5.0.0"), reason="transformers v5 MOE model has breaking changes"
    )
    def test_q2k_mixed(self):
        model_path = "/data0/MiroThinker-v1.5-30B"
        saved_tiny_model_path = save_tiny_model(
            model_path,
            "./tmp/tiny_qwen_model_path",
            num_layers=3,
            is_mllm=False,
        )
        autoround = AutoRound(
            saved_tiny_model_path,
            iters=0,
            nsamples=1,
            seqlen=16,
            disable_opt_rtn=True,
        )
        quantized_model_path = "./saved"
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q2_k_mixed")

        gguf_file = os.listdir(quantized_model_path)[0]
        file_size = os.path.getsize(os.path.join(quantized_model_path, gguf_file)) / 1024**2
        assert abs(file_size - 1236) < 5.0
        from gguf.gguf_reader import GGUFReader

        gguf_model = GGUFReader(os.path.join(quantized_model_path, gguf_file))
        assert gguf_model.get_tensor(2).name == "blk.0.attn_v.weight"
        assert gguf_model.get_tensor(2).tensor_type.name == "Q4_K"
        assert gguf_model.get_tensor(9).name == "blk.0.ffn_up_exps.weight"
        assert gguf_model.get_tensor(9).tensor_type.name == "Q2_K"

        shutil.rmtree(saved_tiny_model_path, ignore_errors=True)
        shutil.rmtree(quantized_model_path, ignore_errors=True)
