from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_round import AutoRound
from auto_round.utils.code_calibration import build_code_calibration_dataset, detect_code_model


@pytest.mark.parametrize(
    "model,config",
    [
        ("Qwen/Qwen3-Coder-30B", None),
        ("/models/CodeLlama-7b", None),
        ("bigcode/starcoder2-15b", None),
        ("generic/model", SimpleNamespace(architectures=["DeepseekCoderForCausalLM"])),
        ("generic/model", SimpleNamespace(finetuning_task="code-generation")),
        ("generic/model", SimpleNamespace(task_specific_params={"text-to-code": {}})),
    ],
)
def test_detect_code_model(model, config):
    detection = detect_code_model(model, config)
    assert detection.is_code
    assert detection.source
    assert detection.match


@pytest.mark.parametrize(
    "model,config",
    [
        ("Qwen/Qwen3-4B", None),
        ("org/encoder-decoder", None),
        ("org/audio-codec", None),
        ("/srv/code/checkpoints/Llama-3", None),
        ("org/notstarcoder-model", None),
        ("generic/model", SimpleNamespace(_name_or_path="/srv/code/checkpoints/Llama-3")),
        ("generic/model", SimpleNamespace(architectures=["SomeEncoderDecoderModel"])),
    ],
)
def test_does_not_misclassify_general_models(model, config):
    assert not detect_code_model(model, config).is_code


@pytest.mark.parametrize(
    "datasets_version,expected",
    [
        (
            "3.6.0",
            "opencode-instruct:num=64,github-code-clean:num=51,mbpp:split=train:num=13",
        ),
        ("5.0.0", "opencode-instruct:num=107,mbpp:split=train:num=21"),
    ],
)
def test_build_code_calibration_dataset(datasets_version, expected):
    selection = build_code_calibration_dataset(128, datasets_version)
    assert selection.dataset == expected


def test_tiny_code_calibration_mix_omits_zero_sample_sources():
    selection = build_code_calibration_dataset(1, "3.6.0")
    assert selection.dataset == "opencode-instruct:num=1"


def test_automatic_code_dataset_and_explicit_override(tiny_opt_model_path, tmp_path, monkeypatch):
    import datasets

    monkeypatch.setattr(datasets, "__version__", "5.0.0")
    code_model_path = tmp_path / "Qwen3-Coder-smoke"
    code_model_path.symlink_to(Path(tiny_opt_model_path).resolve(), target_is_directory=True)
    common = dict(iters=1, nsamples=6, seqlen=8, device_map="cpu", low_cpu_mem_usage=False)

    autoround = AutoRound(str(code_model_path), **common)
    assert autoround.dataset == "opencode-instruct:num=5,mbpp:split=train:num=1"

    autoround = AutoRound(str(code_model_path), dataset="pile-10k", alg_configs="auto_round", **common)
    assert autoround.dataset == "pile-10k"
