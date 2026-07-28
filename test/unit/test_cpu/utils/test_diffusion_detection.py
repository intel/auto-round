from pathlib import Path

from auto_round.utils.model import is_diffusion_model


def test_local_diffusion_pipeline_is_detected_from_model_index(monkeypatch, tmp_path):
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("auto_round.utils.model.check_diffusers_installed", lambda: None)

    def fail_auto_config(*_args, **_kwargs):
        raise AssertionError("local diffusion pipelines must be detected before AutoConfig")

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", fail_auto_config)

    assert is_diffusion_model(str(tmp_path)) is True


def test_missing_local_pipeline_component_does_not_probe_auto_config(monkeypatch, tmp_path):
    component_path = Path(tmp_path, "feature_extractor")

    def fail_auto_config(*_args, **_kwargs):
        raise AssertionError("missing local component paths must not be sent to AutoConfig")

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", fail_auto_config)

    assert is_diffusion_model(str(component_path)) is False
