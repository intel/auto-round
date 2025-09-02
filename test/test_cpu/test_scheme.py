import shutil
import sys
import unittest

sys.path.insert(0, "../..")

from auto_round import AutoRound


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "facebook/opt-125m"
        self.save_folder = "./saved"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_gguf(self):
        ar = AutoRound("Qwen/Qwen3-0.6B", scheme="W2A16", nsamples=1, iters=1)
        ar.quantize_and_save(self.save_folder, format="gguf:q4_k_m")
        self.assertEqual(ar.bits, 4)
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_w4a16(self):
        ar = AutoRound(self.model_name, scheme="W4A16", nsamples=1, iters=1)
        self.assertEqual(ar.bits, 4)
        ar.quantize()

    def test_w2a16_rtn(self):
        ar = AutoRound(self.model_name, scheme="W2A16", nsamples=1, iters=0)
        self.assertEqual(ar.bits, 2)
        ar.quantize()

    def test_mxfp4_rtn(self):
        ar = AutoRound(self.model_name, scheme="MXFP4", nsamples=1, iters=1)
        self.assertEqual(ar.bits, 4)
        self.assertEqual(ar.act_bits, 4)
        self.assertEqual(ar.data_type, "mx_fp")
        self.assertEqual(ar.act_data_type, "mx_fp_rceil")
        ar.quantize()


if __name__ == "__main__":
    unittest.main()
