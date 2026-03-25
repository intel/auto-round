import copy
import shutil

import pytest
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...helpers import get_model_path, save_tiny_model


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

    def test_transform_mxfp4_quant_infer(self):
        model_name = get_model_path("qwen/Qwen3-0.6B")
        scheme = "MXFP4"

        from auto_round.utils import llm_load_model

        model, tokenizer = llm_load_model(
            model_name,
            platform="hf",
            device="cpu",  # always load cpu first
            model_dtype=None,
            trust_remote_code=True,
        )

        from auto_round.experimental.transform.apply import apply_hadamard_transform
        from auto_round.experimental.transform.hadamard_config import HadamardConfig

        hadamard_config = HadamardConfig()
        model = apply_hadamard_transform(
            model,
            hadamard_config,
        )

        ar = AutoRound(
            model=model,
            iters=0,
            seqlen=2,
            scheme=scheme,
            hadamard_config=hadamard_config,
        )
        compressed_model, _ = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        tokenizer.save_pretrained(self.save_dir)

        model = AutoModelForCausalLM.from_pretrained(self.save_dir, torch_dtype="auto", device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        from ...helpers import generate_prompt

        generate_prompt(model, tokenizer)

    def test_transform_mxfp4_tuning_quant_infer(self):
        model_name = get_model_path("qwen/Qwen3-0.6B")
        scheme = "MXFP4"

        from auto_round.utils import llm_load_model

        model, tokenizer = llm_load_model(
            model_name,
            platform="hf",
            device="cpu",  # always load cpu first
            model_dtype=None,
            trust_remote_code=True,
        )

        from auto_round.experimental.transform.apply import apply_hadamard_transform
        from auto_round.experimental.transform.hadamard_config import HadamardConfig

        hadamard_config = HadamardConfig()
        model = apply_hadamard_transform(
            model,
            hadamard_config,
            need_calibration=True
        )

        ar = AutoRound(
            model=model,
            iters=2,
            seqlen=2,
            tokenizer=tokenizer,
            scheme=scheme,
            hadamard_config=hadamard_config,
        )
        compressed_model, _ = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        tokenizer.save_pretrained(self.save_dir)

        model = AutoModelForCausalLM.from_pretrained(self.save_dir, torch_dtype="auto", device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        from ...helpers import generate_prompt

        generate_prompt(model, tokenizer)

    def test_random_transform_mxfp4_quant_infer(self):
        model_name = get_model_path("qwen/Qwen3-0.6B")
        scheme = "MXFP4"

        from auto_round.utils import llm_load_model

        model, tokenizer = llm_load_model(
            model_name,
            platform="hf",
            device="cpu",  # always load cpu first
            model_dtype=None,
            trust_remote_code=True,
        )

        from auto_round.experimental.transform.apply import apply_hadamard_transform
        from auto_round.experimental.transform.hadamard_config import HadamardConfig

        hadamard_config = HadamardConfig(hadamard_type="random_hadamard")
        model = apply_hadamard_transform(
            model,
            hadamard_config,
        )

        ar = AutoRound(
            model=model,
            iters=0,
            seqlen=2,
            scheme=scheme,
            hadamard_config=hadamard_config,
        )
        compressed_model, _ = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        tokenizer.save_pretrained(self.save_dir)

        model = AutoModelForCausalLM.from_pretrained(self.save_dir, torch_dtype="auto", device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        from ...helpers import generate_prompt

        generate_prompt(model, tokenizer)
