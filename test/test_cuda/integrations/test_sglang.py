import gc
import json
import shutil
import sys
from pathlib import Path

import pytest
import sglang as sgl
import torch

from auto_round import AutoRound

from ...helpers import get_model_path, qwen_name_or_path


class TestAutoRound:
    model_name = qwen_name_or_path

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("runs", ignore_errors=True)

    def _run_sglang_inference(self, model_path: Path):
        # SM 12.x (Blackwell) GPUs require CUDA >= 12.9 for sglang's gptq_marlin_repack JIT kernel.
        # Skip inference when the environment is known to be incompatible.
        if torch.cuda.is_available():
            try:
                major, minor = torch.cuda.get_device_capability()
                if major >= 12:
                    cuda_ver = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
                    if cuda_ver < (12, 9):
                        pytest.skip(
                            f"SM {major}.{minor} GPU requires CUDA >= 12.9 for sglang GPTQ JIT kernels "
                            f"(installed: CUDA {torch.version.cuda})"
                        )
            except Exception:
                pass
        llm = sgl.Engine(
            model_path=str(model_path), mem_fraction_static=0.5, disable_piecewise_cuda_graph=True, cuda_graph_bs=[1]
        )
        try:
            prompts = ["Hello, my name is"]
            sampling_params = {"temperature": 0.6, "top_p": 0.95}
            outputs = llm.generate(prompts, sampling_params)
            return outputs[0]["text"]
        finally:
            llm.shutdown()
            del llm
            gc.collect()
            torch.cuda.empty_cache()

    def test_ar_format_sglang(self, dataloader):
        autoround = AutoRound(
            self.model_name,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )

        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=self.save_dir,
            inplace=True,
            format="auto_round",
        )

        generated_text = self._run_sglang_inference(quantized_model_path)
        print(generated_text)

        assert "!!!" not in generated_text

    def test_mixed_ar_format_sglang(self, dataloader):
        layer_config = {
            "self_attn": {"bits": 8},
            "lm_head": {"bits": 16},
            "mlp": {"bits": 16, "act_bits": 16},
        }

        autoround = AutoRound(
            self.model_name,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )

        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=self.save_dir,
            inplace=True,
            format="auto_round",
        )
        config_file = Path(quantized_model_path) / "config.json"
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        quant_config = config.get("quantization_config", {})
        extra_config = quant_config.get("extra_config", {})
        # check extra_config only saved attributes differing from Scheme values
        assert "act_bits" not in extra_config[".*mlp.*"].keys()
        assert "group_size" not in extra_config[".*mlp.*"].keys()
        assert "bits" in extra_config[".*mlp.*"].keys() and extra_config[".*mlp.*"]["bits"] == 16
        assert "bits" in extra_config[".*self_attn.*"].keys() and extra_config[".*self_attn.*"]["bits"] == 8
        generated_text = self._run_sglang_inference(quantized_model_path)
        print(generated_text)

        assert "!!!" not in generated_text

        shutil.rmtree(self.save_dir, ignore_errors=True)

    # TODO: transformers already fixed this bug, need to upgrade sglang
    @pytest.mark.xfail(reason="sglang is not upgraded with latest transformers")
    def test_qwen2_5_vl_loading(self, tiny_qwen_2_5_vl_model_path):
        from auto_round.utils import mllm_load_model

        layer_config = {
            "self_attn": {"bits": 8},
            "lm_head": {"bits": 16},
            "mlp": {"bits": 16, "act_bits": 16},
        }

        model, processor, tokenizer, image_processor = mllm_load_model(tiny_qwen_2_5_vl_model_path)

        autoround = AutoRound(
            model,
            tokenizer,
            scheme="W4A16",
            iters=1,
            nsamples=1,
            seqlen=32,
            processor=processor,
            image_processor=image_processor,
            layer_config=layer_config,
        )

        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=self.save_dir,
            inplace=True,
            format="auto_round",
        )

        generated_text = self._run_sglang_inference(quantized_model_path)
        print(generated_text)

        assert "!!!" not in generated_text

    @pytest.mark.skip_ci(reason="Cannot work well in CI env")
    def test_awq_format_sglang(self, dataloader):
        autoround = AutoRound(
            self.model_name,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )

        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=self.save_dir,
            inplace=True,
            format="auto_round:auto_awq",
        )

        generated_text = self._run_sglang_inference(quantized_model_path)
        print(generated_text)

        assert "!!!" not in generated_text
