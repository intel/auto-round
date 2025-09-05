import shutil
import sys
import unittest

from auto_round.schemes import QuantizationScheme

sys.path.insert(0, "../..")

from auto_round import AutoRound


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "/models/opt-125m"
        self.save_folder = "./saved"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    ## tuning tests
    def test_gguf(self):
        ar = AutoRound("/models/Qwen3-0.6B", scheme="W2A16", nsamples=1, iters=1)
        ar.quantize_and_save(self.save_folder, format="gguf:q4_k_m")
        self.assertEqual(ar.bits, 4)
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_w4a16(self):
        ar = AutoRound(self.model_name, scheme="W4A16", nsamples=1, iters=1)
        self.assertEqual(ar.bits, 4)
        ar.quantize()

    def test_w2a16(self):
        ar = AutoRound(self.model_name, scheme="W2A16", nsamples=1, iters=1)
        self.assertEqual(ar.bits, 2)
        ar.quantize()

    def test_mxfp4(self):
        ar = AutoRound(self.model_name, scheme="MXFP4", nsamples=1, iters=1)
        self.assertEqual(ar.bits, 4)
        self.assertEqual(ar.act_bits, 4)
        self.assertEqual(ar.data_type, "mx_fp")
        self.assertEqual(ar.act_data_type, "mx_fp_rceil")
        ar.quantize()
    
    def test_fp8_static(self):
        ar = AutoRound(self.model_name, scheme="FPW8_STATIC", nsamples=1, iters=1)
        self.assertEqual(ar.bits, 8)
        self.assertEqual(ar.act_bits, 8)
        self.assertEqual(ar.data_type, "fp")
        self.assertEqual(ar.act_data_type, "fp")
        self.assertEqual(ar.group_size, -1)
        self.assertEqual(ar.act_dynamic, False)
        ar.quantize()

    ## RTN tests
    def test_w2a16_rtn(self):
        ar = AutoRound(self.model_name, scheme="W2A16", nsamples=1, iters=0)
        self.assertEqual(ar.bits, 2)
        ar.quantize()

    def test_mxfp4_rtn(self):
        ar = AutoRound(self.model_name, scheme="MXFP4", nsamples=1, iters=0)
        self.assertEqual(ar.bits, 4)
        self.assertEqual(ar.act_bits, 4)
        self.assertEqual(ar.data_type, "mx_fp")
        self.assertEqual(ar.act_data_type, "mx_fp_rceil")
        ar.quantize()
    
    def test_fp8_static_rtn(self):
        ar = AutoRound(self.model_name, scheme="FPW8_STATIC", nsamples=1, iters=0)
        self.assertEqual(ar.bits, 8)
        self.assertEqual(ar.act_bits, 8)
        self.assertEqual(ar.data_type, "fp")
        self.assertEqual(ar.act_data_type, "fp")
        self.assertEqual(ar.group_size, -1)
        self.assertEqual(ar.act_dynamic, False)
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
                self.assertEqual(m.bits, 2)
            if n == "model.decoder.layers.2.self_attn.k_proj":
                self.assertEqual(m.bits, 2)
            if n == "model.decoder.layers.3.self_attn.v_proj":
                self.assertEqual(m.bits, 8)
            if n == "model.decoder.layers.4.self_attn.k_proj":
                self.assertEqual(m.group_size, 64)
    

if __name__ == "__main__":
    unittest.main()

