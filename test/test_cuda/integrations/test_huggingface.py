import pytest

from ...helpers import evaluate_accuracy

model_name_or_path = "Intel/Qwen3.5-2B-int4-AutoRound"


@pytest.mark.skip_ci(reason="Requires downloading a large model from HuggingFace")
def test_hf():
    evaluate_accuracy(
        model_name_or_path,
        threshold=0.4,
        model_type="hf",
        limit=100,
    )


@pytest.mark.skip_ci(reason="Requires downloading a large model from HuggingFace")
def test_hf_multimodal():
    evaluate_accuracy(model_name_or_path, threshold=0.4, batch_size=8, model_type="hf-multimodal", limit=100)
