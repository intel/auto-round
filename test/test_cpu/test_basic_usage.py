import os
import shutil
import sys
import unittest

sys.path.insert(0, "../..")


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

        ##test llm script
        # res = os.system(
        #     f"cd ../.. && {python_path} -m auto_round -h")
        # if res > 0 or res == -1:
        #     assert False, "cmd line test fail, please have a check"
        #
        res = os.system(
            f"cd ../.. && {python_path} -m auto_round --model 'facebook/opt-125m' --seqlen 32 --iter 2 --nsamples 1 --format auto_gptq,auto_round --output_dir ./saved --tasks piqa"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"cd ../.. && {python_path} -m auto_round --model 'facebook/opt-125m' --seqlen 8 --iter 1 --nsamples 1 --eval_task_by_task --tasks openbookqa --bs 32"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"cd ../.. && {python_path} -c 'from auto_round.__main__ import run_light; run_light()' --seqlen 8 --iter 2 --nsamples 8 --output_dir ./saved --tasks lambada_openai"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        # test mllm script
        # test auto_round_mllm help
        res = os.system(f"cd ../.. && {python_path} -m auto_round --mllm -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        # test auto_round_mllm --eval help
        res = os.system(f"cd ../.. && {python_path} -m auto_round --mllm --eval -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        # test auto_round_mllm --lmms help
        res = os.system(f"cd ../.. && {python_path} -m auto_round --mllm --lmms -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"cd ../.. && {python_path} -m auto_round --mllm --iter 2 --nsamples 10 --seqlen 32 --format auto_round --output_dir ./saved"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"cd ../.. && {python_path} -m auto_round --mllm --iter 2 --nsamples 10  --seqlen 256 --format auto_round"
            " --quant_nontext_module --output_dir ./saved "
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"


if __name__ == "__main__":
    unittest.main()
