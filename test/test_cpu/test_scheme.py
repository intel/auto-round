import shutil

import pytest
import torch

from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme


class TestAutoRound:
    @classmethod
    def setup_class(self):
        self.model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        self.save_folder = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_gguf(self, dataloader):
        ar = AutoRound(
            "/tf_dataset/auto_round/models/Qwen/Qwen3-0.6B",
            scheme="W2A16",
            nsamples=1,
            iters=1,
            seqlen=2,
            dataset=dataloader,
        )
        ar.quantize_and_save(self.save_folder, format="gguf:q4_k_m")
        assert ar.bits == 4
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_w4a16(self, dataloader):
        ar = AutoRound(self.model_name, scheme="W4A16", nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        assert ar.bits == 4
        ar.quantize()

    def test_w2a16_rtn(self, dataloader):
        ar = AutoRound(self.model_name, scheme="W2A16", nsamples=1, iters=0, seqlen=2, dataset=dataloader)
        assert ar.bits == 2
        ar.quantize()

    def test_mxfp4(self, dataloader):
        ar = AutoRound(self.model_name, scheme="MXFP4", nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "mx_fp"
        assert ar.act_data_type == "mx_fp_rceil"
        ar.quantize()

    def test_vllm(self):
        from auto_round import AutoRoundMLLM

        ar = AutoRoundMLLM(
            "/tf_dataset/auto_round/models/Qwen/Qwen2-VL-2B-Instruct", scheme="W2A16", nsamples=1, iters=1, seqlen=2
        )
        assert ar.bits == 2
        assert ar.act_bits == 16

    def test_nvfp4(self, dataloader):
        ar = AutoRound(self.model_name, scheme="NVFP4", nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "nv_fp"
        assert ar.act_data_type == "nv_fp4_with_static_gs"
        ar.quantize()

    def test_all_scheme(self, dataloader):
        import copy

        preset_schemes = ["W8A16", "MXFP8", "FPW8A16", "FP8_STATIC", "GGUF:Q2_K_S", "GGUF:Q4_K_M"]
        for scheme in preset_schemes:
            model_name = self.model_name
            if "gguf" in scheme.lower():
                model_name = "/tf_dataset/auto_round/models/Qwen/Qwen2.5-1.5B-Instruct"
            print(f"scheme={scheme}")
            ar = AutoRound(model_name, scheme=scheme, nsamples=1, iters=1, seqlen=2, dataset=dataloader)
            ar.quantize_and_save(self.save_folder)
            shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_scheme_in_layer_config(self, dataloader):
        layer_config = {
            "model.decoder.layers.2.self_attn": {"bits": 2},
            "model.decoder.layers.3.self_attn.v_proj": "W8A16",
            "model.decoder.layers.4.self_attn.k_proj": QuantizationScheme.from_dict({"group_size": 64}),
        }
        ar = AutoRound(
            "/tf_dataset/auto_round/models/facebook/opt-125m",
            scheme="W3A16",
            nsamples=1,
            iters=1,
            layer_config=layer_config,
            seqlen=2,
            dataset=dataloader,
        )

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

    def test_parse_available_devices(self):
        from auto_round.utils.device import parse_available_devices

        device_list = parse_available_devices("auto")
        self.assertTrue(len(device_list) == 1 and "cpu" in device_list)
        device_list = parse_available_devices("a:cuda:0,b:cuda:1,c:cpu")
        self.assertTrue(len(device_list) == 3)
        assert device_list == ["cuda:0", "cuda:1", "cpu"]
        device_list = parse_available_devices("0,1")
        self.assertTrue(len(device_list) == 1 and "cpu" in device_list)
