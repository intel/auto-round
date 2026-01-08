import os
import shutil
import sys

from ...helpers import get_model_path


class TestAutoRoundCmd:
    @classmethod
    def setup_class(self):
        pass

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
        shutil.rmtree("../../saved", ignore_errors=True)
        shutil.rmtree("../../tmp_autoround", ignore_errors=True)

    def test_auto_round_cmd(self, tiny_opt_model_path, tiny_qwen_vl_model_path):
        python_path = sys.executable

        # Test llm script
        res = os.system(f"cd .. && {python_path} -m auto_round -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"cd .. && {python_path} -m auto_round --model {tiny_opt_model_path} --seqlen 32 --iter 2 --nsamples 1 --format auto_gptq,auto_round --output_dir ./saved  --tasks piqa"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"cd .. && {python_path} -m auto_round --model {tiny_opt_model_path} --seqlen 8 --iter 1 --nsamples 1 --eval_task_by_task --tasks openbookqa --bs 32"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"cd .. && {python_path} -c 'from auto_round.__main__ import run_light; run_light()' --seqlen 8 --iter 2 --nsamples 8 --output_dir ./saved --tasks lambada_openai"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        # test mllm script

        # test auto_round_mllm --eval help
        res = os.system(f"cd .. && {python_path} -m auto_round --eval -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        # test auto_round_mllm --lmms help
        res = os.system(f"cd .. && {python_path} -m auto_round --eval --lmms -h")
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"cd .. && {python_path} -m auto_round --mllm --model {tiny_qwen_vl_model_path} --iter 2 --nsamples 2 --seqlen 32 --format auto_round --output_dir ./saved"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"cd .. && {python_path} -m auto_round --mllm --iter 2 --nsamples 2 --model {tiny_qwen_vl_model_path} --seqlen 32 --format auto_round"
            " --quant_nontext_module --output_dir ./saved "
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
