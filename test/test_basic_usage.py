import os
import shutil
import sys
import unittest


class TestAutoRoundCmd(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_auto_round_cmd(self):
        python_path = sys.executable

        res = os.system(f"{python_path} ../auto_round/__main__.py --model 'facebook/opt-125m' --iter 2 --nsamples 1 --format auto_gptq,auto_round")
        print(res)
