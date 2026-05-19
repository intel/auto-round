import pytest

from ...helpers import evaluate_accuracy

model_name_or_path = "Intel/Qwen3.5-2B-int4-AutoRound"


@pytest.mark.skip_ci(reason="Requires downloading a large model from HuggingFace")
def test_evaluate_accuracy():
    evaluate_accuracy(model_name_or_path, threshold=0.5)
