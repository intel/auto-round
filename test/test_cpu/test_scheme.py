import shutil
import sys
import unittest

import torch

sys.path.insert(0, "../..")
from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        self.save_folder = "./saved"
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_gguf(self):
        ar = AutoRound(
            "/tf_dataset/auto_round/models/Qwen/Qwen3-0.6B",
            scheme="W2A16",
            nsamples=1,
            iters=1,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        ar.quantize_and_save(self.save_folder, format="gguf:q4_k_m")
        self.assertEqual(ar.bits, 4)
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_w4a16(self):
        ar = AutoRound(self.model_name, scheme="W4A16", nsamples=1, iters=1, seqlen=2, dataset=self.llm_dataloader)
        self.assertEqual(ar.bits, 4)
        ar.quantize()

    def test_w2a16_rtn(self):
        ar = AutoRound(self.model_name, scheme="W2A16", nsamples=1, iters=0, seqlen=2, dataset=self.llm_dataloader)
        self.assertEqual(ar.bits, 2)
        ar.quantize()

    def test_mxfp4(self):
        ar = AutoRound(self.model_name, scheme="MXFP4", nsamples=1, iters=1, seqlen=2, dataset=self.llm_dataloader)
        self.assertEqual(ar.bits, 4)
        self.assertEqual(ar.act_bits, 4)
        self.assertEqual(ar.data_type, "mx_fp")
        self.assertEqual(ar.act_data_type, "mx_fp_rceil")
        ar.quantize()

    def test_vllm(self):
        from auto_round import AutoRoundMLLM

        ar = AutoRoundMLLM(
            "/tf_dataset/auto_round/models/Qwen/Qwen2-VL-2B-Instruct", scheme="W2A16", nsamples=1, iters=1, seqlen=2
        )
        self.assertEqual(ar.bits, 2)
        self.assertEqual(ar.act_bits, 16)

    def test_nvfp4(self):
        ar = AutoRound(self.model_name, scheme="NVFP4", nsamples=1, iters=1, seqlen=2, dataset=self.llm_dataloader)
        self.assertEqual(ar.bits, 4)
        self.assertEqual(ar.act_bits, 4)
        self.assertEqual(ar.data_type, "nv_fp")
        self.assertEqual(ar.act_data_type, "nv_fp4_with_static_gs")
        ar.quantize()

    def test_all_scheme(self):
        import copy

        preset_schemes = ["W8A16", "MXFP8", "FPW8A16", "FP8_STATIC", "GGUF:Q2_K_S", "GGUF:Q4_K_M"]
        for scheme in preset_schemes:
            model_name = self.model_name
            if "gguf" in scheme.lower():
                model_name = "/tf_dataset/auto_round/models/Qwen/Qwen2.5-1.5B-Instruct"
            print(f"scheme={scheme}")
            ar = AutoRound(model_name, scheme=scheme, nsamples=1, iters=1, seqlen=2, dataset=self.llm_dataloader)
            ar.quantize_and_save(self.save_folder)
            shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_scheme_in_layer_config(self):
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
            dataset=self.llm_dataloader,
        )

        ar.quantize()
        for n, m in ar.model.named_modules():
            if n == "model.decoder.layers.2.self_attn.q_proj":
                self.assertEqual(m.bits, 2)
            if n == "model.decoder.layers.2.self_attn.k_proj":
                self.assertEqual(m.bits, 2)
            if n == "model.decoder.layers.3.self_attn.v_proj":
                self.assertEqual(m.bits, 8)
            if n == "model.decoder.layers.4.self_attn.k_proj":
                self.assertEqual(m.group_size, 64)


if __name__ == "__main__":
    unittest.main()
