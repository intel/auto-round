"""Regression checks for MiniMax-H3 W4A16 discovery support."""

import json


def test_minimax_h3_blocks_are_registered():
    from auto_round.algorithms.block_runner import BlockForwardRunner

    assert BlockForwardRunner.DIFFUSION_OUTPUT_CONFIGS["MiniMaxH3TransformerBlock"] == ["hidden_states"]
    assert BlockForwardRunner.DIFFUSION_OUTPUT_CONFIGS["MiniMaxH3TokenRefinerBlock"] == ["hidden_states"]


def test_modular_diffusion_index_is_detected(tmp_path, monkeypatch):
    from auto_round.utils import model as model_utils

    (tmp_path / "modular_model_index.json").write_text(json.dumps({"workflows": {}}))
    monkeypatch.setattr(model_utils, "check_diffusers_installed", lambda: None)
    def fail_config(*args, **kwargs):
        raise RuntimeError("no config in synthetic modular checkpoint")

    monkeypatch.setattr(model_utils.transformers.AutoConfig, "from_pretrained", fail_config)

    assert model_utils.is_diffusion_model(str(tmp_path)) is True
