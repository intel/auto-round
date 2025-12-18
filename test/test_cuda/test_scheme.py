import shutil

import pytest

from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme


class TestAutoRound:
    @classmethod
    def setup_class(self):
        self.model_name = "/models/opt-125m"
        self.save_folder = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    # Tuning tests
    def test_gguf(self):
        ar = AutoRound("/models/Qwen3-0.6B", scheme="W2A16", nsamples=1, iters=1)
        ar.quantize_and_save(self.save_folder, format="gguf:q4_k_m")
        assert ar.bits == 4
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_w4a16(self):
        ar = AutoRound(self.model_name, scheme="W4A16", nsamples=1, iters=1)
        assert ar.bits == 4
        ar.quantize()

    def test_w2a16(self):
        ar = AutoRound(self.model_name, scheme="W2A16", nsamples=1, iters=1)
        assert ar.bits == 2
        ar.quantize()

    def test_mxfp4(self):
        ar = AutoRound(self.model_name, scheme="MXFP4", nsamples=1, iters=1)
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "mx_fp"
        assert ar.act_data_type == "mx_fp_rceil"
        ar.quantize()

    def test_fp8_static(self):
        ar = AutoRound(self.model_name, scheme="FP8_STATIC", nsamples=1, iters=1)
        assert ar.bits == 8
        assert ar.act_bits == 8
        assert ar.data_type == "fp"
        assert ar.act_data_type == "fp"
        assert ar.group_size == -1
        assert ar.act_dynamic == False
        ar.quantize()

    ## RTN tests
    def test_w2a16_rtn(self):
        ar = AutoRound(self.model_name, scheme="W2A16", nsamples=1, iters=0)
        assert ar.bits == 2
        ar.quantize()

    def test_mxfp4_rtn(self):
        ar = AutoRound(self.model_name, scheme="MXFP4", nsamples=1, iters=0)
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "mx_fp"
        assert ar.act_data_type == "mx_fp_rceil"
        ar.quantize()

    def test_fp8_static_rtn(self):
        ar = AutoRound(self.model_name, scheme="FP8_STATIC", nsamples=1, iters=0)
        assert ar.bits == 8
        assert ar.act_bits == 8
        assert ar.data_type == "fp"
        assert ar.act_data_type == "fp"
        assert ar.group_size == -1
        assert ar.act_dynamic == False
        ar.quantize()

    def test_scheme_in_layer_config(self):
        layer_config = {
            "model.decoder.layers.2.self_attn": {"bits": 2},
            "model.decoder.layers.3.self_attn.v_proj": "W8A16",
            "model.decoder.layers.4.self_attn.k_proj": QuantizationScheme.from_dict({"group_size": 64}),
        }
        ar = AutoRound(self.model_name, scheme="W3A16", nsamples=1, iters=1, layer_config=layer_config)

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
