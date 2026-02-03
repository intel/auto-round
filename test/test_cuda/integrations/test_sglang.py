import json
import shutil
import sys
from pathlib import Path

import pytest
import sglang as sgl
import torch

from auto_round import AutoRound

from ...helpers import get_model_path, opt_name_or_path


class TestAutoRound:
    save_dir = "./saved"
    model_name = opt_name_or_path

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

    def _run_sglang_inference(self, model_path: Path):
        llm = sgl.Engine(model_path=str(model_path), mem_fraction_static=0.7)
        prompts = ["Hello, my name is"]
        sampling_params = {"temperature": 0.6, "top_p": 0.95}
        outputs = llm.generate(prompts, sampling_params)
        return outputs[0]["text"]

    def test_ar_format_sglang(self, dataloader):
        autoround = AutoRound(
            self.model_name,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )

        autoround.quantize_and_save(
            output_dir=self.save_dir,
            inplace=True,
            format="auto_round",
        )

        generated_text = self._run_sglang_inference(self.save_dir)
        print(generated_text)

        assert "!!!" not in generated_text

        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_mixed_ar_format_sglang(self, dataloader):
        layer_config = {
            "self_attn": {"bits": 8},
            "lm_head": {"bits": 16},
            "fc1": {"bits": 16, "act_bits": 16},
        }

        autoround = AutoRound(
            self.model_name,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )

        autoround.quantize_and_save(
            output_dir=self.save_dir,
            inplace=True,
            format="auto_round",
        )
        config_file = Path(self.save_dir) / "config.json"
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        quant_config = config.get("quantization_config", {})
        extra_config = quant_config.get("extra_config", {})
        # check extra_config only saved attributes differing from Scheme values
        assert "act_bits" not in extra_config[".*fc1.*"].keys()
        assert "group_size" not in extra_config[".*fc1.*"].keys()
        assert "bits" in extra_config[".*fc1.*"].keys() and extra_config[".*fc1.*"]["bits"] == 16
        assert "bits" in extra_config[".*self_attn.*"].keys() and extra_config[".*self_attn.*"]["bits"] == 8
        generated_text = self._run_sglang_inference(self.save_dir)
        print(generated_text)

        assert "!!!" not in generated_text

        shutil.rmtree(self.save_dir, ignore_errors=True)
