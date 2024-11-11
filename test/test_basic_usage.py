import os
import shutil
import sys
import unittest

sys.path.insert(0, '..')


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

        # test llm script
        res = os.system(
            f"cd .. && {python_path} -m auto_round -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have acheck"

        res = os.system(
            f"cd .. && {python_path} -m auto_round --model 'facebook/opt-125m' --iter 2 --nsamples 1 --format auto_gptq,auto_round --disable_eval --output_dir ./saved")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have acheck"
        
        
        # test mllm script
        res = os.system(
            f"cd .. && {python_path} -m auto_round --mllm -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have acheck"

        res = os.system(
            f"cd .. && {python_path} -m auto_round --mllm --iter 2 --nsamples 10 --format auto_round --output_dir ./saved")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have acheck"


if __name__ == "__main__":
    unittest.main()