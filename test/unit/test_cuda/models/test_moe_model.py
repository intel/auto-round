from test.helpers import check_version

import pytest
import torch
from transformers import AutoRoundConfig

from auto_round import AutoRound


def _calibration_data(seqlen):
    return [
        {
            "input_ids": torch.ones((1, seqlen), dtype=torch.long),
            "attention_mask": torch.ones((1, seqlen), dtype=torch.bool),
        }
    ]


@pytest.mark.skipif(not check_version("transformers>=5.2.0"), reason="requires transformers >= 5.2.0")
@pytest.mark.timeout(90)
def test_qwen3_5_moe(tiny_qwen35_moe_text_model_path, tmp_path):
    from transformers import Qwen3_5MoeForCausalLM

    ar = AutoRound(
        tiny_qwen35_moe_text_model_path,
        dataset=_calibration_data(8),
        nsamples=1,
        seqlen=8,
        iters=1,
        # The tiny expert down projection is 64-wide; use a supported group size instead of fake fallback.
        group_size=32,
    )
    quantized_model, quantized_model_path = ar.quantize_and_save(format="auto_round", output_dir=tmp_path / "quantized")
    assert quantized_model is not None, "Quantized model should not be None."

    # Small tensors can select a fake backend that has no CUDA kernel; optimized backend selection has dedicated CUDA tests.
    loaded_model = Qwen3_5MoeForCausalLM.from_pretrained(
        quantized_model_path, quantization_config=AutoRoundConfig(backend="torch")
    ).to("cuda")
    inp = torch.randint(0, loaded_model.config.vocab_size, (1, 8), device="cuda")
    with torch.inference_mode():
        loaded_out = loaded_model(inp)
    assert torch.isfinite(loaded_out.logits).all()


@pytest.mark.skip_ci(
    reason="Architecture: Conditional Qwen3.5 fused-MoE validates the large VLM fixture; retain in the full test tier"
)
@pytest.mark.skipif(not check_version("transformers>=5.2.0"), reason="requires transformers >= 5.2.0")
@pytest.mark.timeout(300)
def test_qwen3_5_moe_conditional(tiny_qwen35_moe_model_path, tmp_path):
    from transformers import Qwen3_5MoeForConditionalGeneration

    ar = AutoRound(
        tiny_qwen35_moe_model_path,
        dataset=_calibration_data(32),
        nsamples=2,
        seqlen=32,
        iters=1,
    )
    quantized_model, quantized_model_path = ar.quantize_and_save(format="auto_round", output_dir=tmp_path / "quantized")
    assert quantized_model is not None, "Quantized model should not be None."

    loaded_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(quantized_model_path).to("cuda")
    inp = torch.randint(0, loaded_model.config.text_config.vocab_size, (1, 64), device="cuda")
    with torch.inference_mode():
        loaded_out = loaded_model(inp)
    assert torch.isfinite(loaded_out.logits).all()
