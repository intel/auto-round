import shutil

import pytest
import torch
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

from auto_round import AutoRound

from ...helpers import check_version


@pytest.mark.skipif(not check_version("transformers>=5.2.0"), reason="requires transformers >= 5.2.0")
def test_qwen3_5_moe(tiny_qwen35_moe_model_path):
    from transformers import Qwen3_5MoeForConditionalGeneration

    output_dir = "test_quantized_qwen35_moe"
    ar = AutoRound(
        tiny_qwen35_moe_model_path,
        nsamples=2,
        seqlen=32,
        iters=1,
    )
    quantized_model, quantized_model_path = ar.quantize_and_save(format="auto_round", output_dir=output_dir)
    assert quantized_model is not None, "Quantized model should not be None."

    loaded_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(quantized_model_path)
    loaded_model.to("cuda")

    inp = torch.randint(0, 100, (1, 64)).to("cuda")
    with torch.inference_mode():
        loaded_out = loaded_model(inp)

    shutil.rmtree(output_dir, ignore_errors=True)
