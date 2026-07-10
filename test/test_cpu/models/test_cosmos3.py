# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit and integration tests for Cosmos3 diffusion model support."""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestCosmos3ForwardMode:
    """Test _cosmos3_forward_mode() thread-local context manager."""

    def test_forward_mode_sets_and_clears_state(self):
        from auto_round.special_model_handler import _cosmos3_forward_mode, _cosmos3_forward_state

        assert not getattr(_cosmos3_forward_state, "active", False)
        with _cosmos3_forward_mode():
            assert getattr(_cosmos3_forward_state, "active", False) is True
        assert not getattr(_cosmos3_forward_state, "active", False)

    def test_forward_mode_cleanup_on_exception(self):
        from auto_round.special_model_handler import _cosmos3_forward_mode, _cosmos3_forward_state

        with pytest.raises(RuntimeError):
            with _cosmos3_forward_mode():
                raise RuntimeError("simulated error")
        assert not getattr(_cosmos3_forward_state, "active", False)

    def test_forward_mode_nested_context(self):
        from auto_round.special_model_handler import _cosmos3_forward_mode, _cosmos3_forward_state

        with _cosmos3_forward_mode():
            assert getattr(_cosmos3_forward_state, "active", False) is True
            with _cosmos3_forward_mode():
                assert getattr(_cosmos3_forward_state, "active", False) is True
            assert not getattr(_cosmos3_forward_state, "active", False)


class TestWrapCosmos3LayerDualMode:
    """Test _wrap_cosmos3_layer_dual_mode() layer monkey-patching."""

    def test_patch_idempotent(self):
        from auto_round.special_model_handler import _wrap_cosmos3_layer_dual_mode

        mock_layer = MagicMock()
        orig_mock_forward = MagicMock(return_value=("und_out", "gen_out"))
        mock_layer.forward = orig_mock_forward
        mock_layer._autoround_dual_mode_patched = False

        _wrap_cosmos3_layer_dual_mode(mock_layer)
        first_forward = mock_layer.forward
        _wrap_cosmos3_layer_dual_mode(mock_layer)
        assert mock_layer.forward is first_forward

    def test_dual_mode_splits_combined_sequence(self):
        from auto_round.special_model_handler import _wrap_cosmos3_layer_dual_mode

        mock_layer = MagicMock()
        mock_und_out = torch.randn(2, 4, 8)
        mock_gen_out = torch.randn(3, 4, 8)
        orig_mock_forward = MagicMock(return_value=(mock_und_out, mock_gen_out))
        mock_layer.forward = orig_mock_forward
        mock_layer._autoround_dual_mode_patched = False

        _wrap_cosmos3_layer_dual_mode(mock_layer)

        und_seq = torch.randn(5, 4, 8)
        gen_seq = torch.randn(0, 4, 8)
        rotary_emb = (torch.randn(2, 4, 8), torch.randn(2, 4, 8))

        result = mock_layer.forward(und_seq, gen_seq, rotary_emb)

        orig_mock_forward.assert_called_once()
        call_args = orig_mock_forward.call_args[0]
        und_arg, gen_arg, rot_arg = call_args

        assert und_arg.shape[0] == 2
        assert gen_arg.shape[0] == 3
        assert result.shape[0] == 5

    def test_active_mode_bypasses_splitting(self):
        from auto_round.special_model_handler import _cosmos3_forward_mode, _wrap_cosmos3_layer_dual_mode

        mock_layer = MagicMock()
        mock_und_out = torch.randn(2, 4, 8)
        mock_gen_out = torch.randn(3, 4, 8)
        orig_mock_forward = MagicMock(return_value=(mock_und_out, mock_gen_out))
        mock_layer.forward = orig_mock_forward
        mock_layer._autoround_dual_mode_patched = False

        _wrap_cosmos3_layer_dual_mode(mock_layer)

        und_seq = torch.randn(5, 4, 8)
        gen_seq = torch.randn(3, 4, 8)
        rotary_emb = (torch.randn(2, 4, 8), torch.randn(2, 4, 8))

        with _cosmos3_forward_mode():
            mock_layer.forward(und_seq, gen_seq, rotary_emb)

        orig_mock_forward.assert_called_once()
        call_args = orig_mock_forward.call_args[0]
        assert torch.equal(call_args[0], und_seq)
        assert torch.equal(call_args[1], gen_seq)

    def test_mismatched_sequence_length_triggers_split(self):
        from auto_round.special_model_handler import _wrap_cosmos3_layer_dual_mode

        mock_layer = MagicMock()
        mock_layer.forward = MagicMock(return_value=(torch.zeros(1, 4, 8), torch.zeros(1, 4, 8)))
        mock_layer._autoround_dual_mode_patched = False

        orig_mock_forward = mock_layer.forward
        _wrap_cosmos3_layer_dual_mode(mock_layer)

        und_seq = torch.randn(4, 4, 8)
        gen_seq = torch.zeros(0, 4, 8)
        rotary_emb = (torch.randn(2, 4, 8), torch.randn(2, 4, 8))

        mock_layer.forward(und_seq, gen_seq, rotary_emb)

        call_args = orig_mock_forward.call_args[0]
        assert call_args[0].shape[0] == 2
        assert call_args[1].shape[0] == 2


class TestGetCosmos3MultimodalBlock:
    """Test _get_cosmos3_multimodal_block() block name discovery."""

    def test_returns_layers_list_with_quant_vision_false(self):
        from auto_round.special_model_handler import _get_cosmos3_multimodal_block

        mock_model = MagicMock()
        mock_model.layers = [MagicMock() for _ in range(8)]

        result = _get_cosmos3_multimodal_block(mock_model, quant_vision=False)

        assert len(result) == 1
        assert len(result[0]) == 8
        assert result[0][0] == "layers.0"
        assert result[0][7] == "layers.7"

    def test_returns_layers_list_with_quant_vision_true(self):
        from auto_round.special_model_handler import _get_cosmos3_multimodal_block

        mock_model = MagicMock()
        mock_model.layers = [MagicMock() for _ in range(4)]

        result = _get_cosmos3_multimodal_block(mock_model, quant_vision=True)

        # quant_vision is ignored for Cosmos3 — only text layers are quantized
        assert len(result) == 1
        assert result[0][0] == "layers.0"

    def test_returns_empty_when_no_layers(self):
        from auto_round.special_model_handler import _get_cosmos3_multimodal_block

        mock_model = MagicMock(spec=[])
        del mock_model.layers

        result = _get_cosmos3_multimodal_block(mock_model)
        assert result == []


class TestBypassCosmos3SafetyChecker:
    """Test _bypass_cosmos3_safety_checker() import safety and patching."""

    @staticmethod
    def _make_real_safety_checker_class():
        return type(
            "CosmosSafetyChecker",
            (),
            {
                "_autoround_patched": False,
                "__init__": lambda self: None,
                "to": lambda self, *a, **kw: self,
            },
        )

    def test_noop_when_diffusers_not_installed(self):
        from auto_round.special_model_handler import _bypass_cosmos3_safety_checker

        with patch("builtins.__import__", side_effect=ImportError("No module named 'diffusers'")):
            _bypass_cosmos3_safety_checker()

    def test_noop_when_safety_checker_class_not_found(self):
        from auto_round.special_model_handler import _bypass_cosmos3_safety_checker

        mock_module = MagicMock()
        mock_module.CosmosSafetyChecker = None
        with patch.dict("sys.modules", {"diffusers.pipelines.cosmos.pipeline_cosmos3_omni": mock_module}):
            _bypass_cosmos3_safety_checker()

    def test_patches_safety_checker_methods(self):
        from auto_round.special_model_handler import _bypass_cosmos3_safety_checker

        safety_checker_cls = self._make_real_safety_checker_class()
        mock_module = MagicMock()
        mock_module.CosmosSafetyChecker = safety_checker_cls

        with patch.dict("sys.modules", {"diffusers.pipelines.cosmos.pipeline_cosmos3_omni": mock_module}):
            _bypass_cosmos3_safety_checker()

        assert safety_checker_cls._autoround_patched is True
        assert callable(safety_checker_cls.__init__)
        assert callable(safety_checker_cls.check_text_safety)
        assert callable(safety_checker_cls.check_video_safety)
        assert safety_checker_cls.check_text_safety("any prompt") is True

    def test_idempotent_patch(self):
        from auto_round.special_model_handler import _bypass_cosmos3_safety_checker

        safety_checker_cls = self._make_real_safety_checker_class()
        mock_module = MagicMock()
        mock_module.CosmosSafetyChecker = safety_checker_cls

        with patch.dict("sys.modules", {"diffusers.pipelines.cosmos.pipeline_cosmos3_omni": mock_module}):
            _bypass_cosmos3_safety_checker()
            first_init = safety_checker_cls.__init__
            assert safety_checker_cls._autoround_patched is True
            _bypass_cosmos3_safety_checker()
            # On second call, the function returns early (idempotent)
            assert safety_checker_cls.__init__ is first_init


class TestAlignCosmos3QuantConfigToVllmOmni:
    """Test _align_cosmos3_quant_config_to_vllm_omni() config rewriting."""

    def _write_temp_transformer_dir(self, config_data, quant_config_data=None):
        tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmpdir, "transformer"))

        with open(os.path.join(tmpdir, "transformer", "config.json"), "w") as f:
            json.dump(config_data, f)

        if quant_config_data is not None:
            with open(os.path.join(tmpdir, "transformer", "quantization_config.json"), "w") as f:
                json.dump(quant_config_data, f)

        return tmpdir

    def test_rewrites_layers_block_in_config_json(self):
        from auto_round.special_model_handler import _align_cosmos3_quant_config_to_vllm_omni

        config = {
            "quantization_config": {
                "bits": 4,
                "block_name_to_quantize": "layers",
            }
        }
        tmpdir = self._write_temp_transformer_dir(config)

        try:
            _align_cosmos3_quant_config_to_vllm_omni(os.path.join(tmpdir, "transformer"))

            with open(os.path.join(tmpdir, "transformer", "config.json")) as f:
                result = json.load(f)

            assert result["quantization_config"]["block_name_to_quantize"] == [
                "language_model.layers",
                "gen_layers",
            ]
        finally:
            shutil.rmtree(tmpdir)

    def test_rewrites_layers_block_in_quantization_config_json(self):
        from auto_round.special_model_handler import _align_cosmos3_quant_config_to_vllm_omni

        # Top-level config.json must also have block_name_to_quantize for the
        # rewrite cascade to reach the standalone quantization_config.json.
        config = {
            "quantization_config": {
                "bits": 4,
                "block_name_to_quantize": "layers",
            }
        }
        quant_config = {
            "bits": 4,
            "block_name_to_quantize": "layers",
        }
        tmpdir = self._write_temp_transformer_dir(config, quant_config)

        try:
            _align_cosmos3_quant_config_to_vllm_omni(os.path.join(tmpdir, "transformer"))

            with open(os.path.join(tmpdir, "transformer", "quantization_config.json")) as f:
                result = json.load(f)

            assert result["block_name_to_quantize"] == ["language_model.layers", "gen_layers"]
        finally:
            shutil.rmtree(tmpdir)

    def test_leaves_non_layers_blocks_unchanged(self):
        from auto_round.special_model_handler import _align_cosmos3_quant_config_to_vllm_omni

        config = {
            "quantization_config": {
                "bits": 4,
                "block_name_to_quantize": ["transformer_blocks", "single_transformer_blocks"],
            }
        }
        tmpdir = self._write_temp_transformer_dir(config)

        try:
            _align_cosmos3_quant_config_to_vllm_omni(os.path.join(tmpdir, "transformer"))

            with open(os.path.join(tmpdir, "transformer", "config.json")) as f:
                result = json.load(f)

            assert result["quantization_config"]["block_name_to_quantize"] == [
                "transformer_blocks",
                "single_transformer_blocks",
            ]
        finally:
            shutil.rmtree(tmpdir)

    def test_handles_string_blocks_format(self):
        from auto_round.special_model_handler import _align_cosmos3_quant_config_to_vllm_omni

        config = {
            "quantization_config": {
                "bits": 4,
                "block_name_to_quantize": "layers , single_transformer_blocks",
            }
        }
        tmpdir = self._write_temp_transformer_dir(config)

        try:
            _align_cosmos3_quant_config_to_vllm_omni(os.path.join(tmpdir, "transformer"))

            with open(os.path.join(tmpdir, "transformer", "config.json")) as f:
                result = json.load(f)

            assert result["quantization_config"]["block_name_to_quantize"] == [
                "language_model.layers",
                "gen_layers",
                "single_transformer_blocks",
            ]
        finally:
            shutil.rmtree(tmpdir)

    def test_noop_when_config_json_missing(self):
        from auto_round.special_model_handler import _align_cosmos3_quant_config_to_vllm_omni

        tmpdir = tempfile.mkdtemp()
        try:
            _align_cosmos3_quant_config_to_vllm_omni(os.path.join(tmpdir, "transformer"))
        finally:
            shutil.rmtree(tmpdir)

    def test_noop_when_no_quantization_config(self):
        from auto_round.special_model_handler import _align_cosmos3_quant_config_to_vllm_omni

        config = {"model_type": "cosmos3_vllm_omni"}
        tmpdir = self._write_temp_transformer_dir(config)

        try:
            _align_cosmos3_quant_config_to_vllm_omni(os.path.join(tmpdir, "transformer"))

            with open(os.path.join(tmpdir, "transformer", "config.json")) as f:
                result = json.load(f)

            assert "quantization_config" not in result
        finally:
            shutil.rmtree(tmpdir)


class TestCosmos3Registration:
    """Test Cosmos3 registration in special model handler registries."""

    def test_cosmos3_vllm_omni_registered_in_special_multimodal_block(self):
        from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK

        assert "cosmos3_vllm_omni" in SPECIAL_MULTIMODAL_BLOCK
        assert callable(SPECIAL_MULTIMODAL_BLOCK["cosmos3_vllm_omni"])

    def test_cosmos3_quant_model_registered_in_special_shared_cache_keys(self):
        from auto_round.special_model_handler import SPECIAL_SHARED_CACHE_KEYS

        assert "Cosmos3VllmOmniQuantModel" in SPECIAL_SHARED_CACHE_KEYS
        assert SPECIAL_SHARED_CACHE_KEYS["Cosmos3VllmOmniQuantModel"] == ("rotary_emb",)


class TestCosmos3LoadAndQuantize:
    """End-to-end tests for ``load_cosmos3_diffusion`` with mocked diffusers pipe.

    The diffusers ``Cosmos3OmniPipeline.from_pretrained`` call is mocked so the
    tests exercise the real AutoRound wiring (config construction, layer
    patching, save_config flow) without requiring a multi-GB Cosmos3 checkpoint.
    """

    @staticmethod
    def _build_fake_diffusers_pipe(n_layers=2):
        """Build a minimal stand-in for Cosmos3OmniPipeline with nn.Module layers."""
        import torch.nn as nn

        class _FakeLayer(nn.Module):
            def forward(self, *args, **kwargs):
                return (args[0] if args else None, args[1] if len(args) > 1 else None)

        class _FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_FakeLayer() for _ in range(n_layers)])

        class _FakePipe:
            def __init__(self):
                self.transformer = _FakeTransformer()
                self.config = {"_class_name": "Cosmos3OmniDiffusersPipeline"}

            def to(self, *args, **kwargs):
                return self

        return _FakePipe()

    @pytest.fixture
    def tiny_cosmos3_model_path(self):
        """Write only the files consumed by ``load_cosmos3_diffusion`` for read-side checks."""
        tmpdir = tempfile.mkdtemp()

        transformer_config = {
            "_class_name": "Cosmos3VFMTransformer",
            "architectures": ["Cosmos3VFMTransformer"],
            "model_type": "cosmos3_vllm_omni",
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 32000,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_head_dim": 32,
            "latent_patch_size": 2,
            "max_action_dim": 64,
            "num_embodiment_domains": 1,
            "guidance_embed": True,
        }

        os.makedirs(os.path.join(tmpdir, "transformer"))
        with open(os.path.join(tmpdir, "transformer", "config.json"), "w") as f:
            json.dump(transformer_config, f)

        model_index = {
            "_class_name": "Cosmos3OmniDiffusersPipeline",
            "_name_or_path": tmpdir,
            "scheduler": ["diffusers", "DDPMScheduler"],
            "transformer": ["diffusers", "Cosmos3VFMTransformer"],
        }
        with open(os.path.join(tmpdir, "model_index.json"), "w") as f:
            json.dump(model_index, f)

        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_cosmos3_diffusion_returns_pipe_and_model(self, tiny_cosmos3_model_path):
        from auto_round.special_model_handler import load_cosmos3_diffusion

        fake_pipe = self._build_fake_diffusers_pipe(n_layers=2)
        fake_cls = MagicMock()
        fake_cls.from_pretrained.return_value = fake_pipe
        with patch.dict("sys.modules", {"diffusers": MagicMock(Cosmos3OmniPipeline=fake_cls)}):
            pipe, model = load_cosmos3_diffusion(tiny_cosmos3_model_path, "cpu")

        # Pipe shape
        assert hasattr(pipe, "transformer")
        assert hasattr(pipe, "diffusers_pipe")
        assert hasattr(pipe, "save_config")
        assert pipe._autoround_diffusion_pipe is True
        assert pipe.transformer is model
        assert pipe.dtype == torch.bfloat16

        # Model shape
        assert hasattr(model, "config")
        assert hasattr(model, "layers")
        assert model.config.model_type == "cosmos3_vllm_omni"
        assert model.config._class_name == "Cosmos3VFMTransformer"
        assert "Cosmos3VFMTransformer" in model.config.architectures
        assert len(model.layers) == 2

    def test_load_cosmos3_patches_each_layer(self, tiny_cosmos3_model_path):
        """Each real layer must be wrapped with the dual-mode forward patch."""
        from auto_round.special_model_handler import load_cosmos3_diffusion

        fake_pipe = self._build_fake_diffusers_pipe(n_layers=3)
        fake_cls = MagicMock()
        fake_cls.from_pretrained.return_value = fake_pipe
        with patch.dict("sys.modules", {"diffusers": MagicMock(Cosmos3OmniPipeline=fake_cls)}):
            _, model = load_cosmos3_diffusion(tiny_cosmos3_model_path, "cpu")

        for layer in model.layers:
            assert getattr(layer, "_autoround_dual_mode_patched", False) is True

    def test_load_cosmos3_pipe_save_config_rewrites_block_names(self, tiny_cosmos3_model_path, tmp_path):
        """Calling pipe.save_config() must invoke the vllm-omni config realignment."""
        from auto_round.special_model_handler import load_cosmos3_diffusion

        fake_pipe = self._build_fake_diffusers_pipe(n_layers=2)
        fake_cls = MagicMock()
        fake_cls.from_pretrained.return_value = fake_pipe

        target_dir = os.path.join(str(tmp_path), "saved")
        os.makedirs(os.path.join(target_dir, "transformer"), exist_ok=True)
        cfg = {
            "_class_name": "Cosmos3VFMTransformer",
            "model_type": "cosmos3_vllm_omni",
            "quantization_config": {"bits": 4, "block_name_to_quantize": "layers"},
        }
        with open(os.path.join(target_dir, "transformer", "config.json"), "w") as f:
            json.dump(cfg, f)
        with open(os.path.join(target_dir, "transformer", "quantization_config.json"), "w") as f:
            json.dump({"bits": 4, "block_name_to_quantize": "layers"}, f)

        with patch.dict("sys.modules", {"diffusers": MagicMock(Cosmos3OmniPipeline=fake_cls)}):
            pipe, _ = load_cosmos3_diffusion(tiny_cosmos3_model_path, "cpu")
            pipe.save_config(target_dir)

        with open(os.path.join(target_dir, "transformer", "config.json")) as f:
            saved_cfg = json.load(f)
        assert saved_cfg["quantization_config"]["block_name_to_quantize"] == [
            "language_model.layers",
            "gen_layers",
        ]
        with open(os.path.join(target_dir, "transformer", "quantization_config.json")) as f:
            saved_qcfg = json.load(f)
        assert saved_qcfg["block_name_to_quantize"] == ["language_model.layers", "gen_layers"]


class TestCosmos3DiffusionLoadModel:
    """Test the diffusion_load_model dispatch for Cosmos3."""

    def _write_model_index(self, class_name):
        tmpdir = tempfile.mkdtemp()
        model_index = {"_class_name": class_name}
        with open(os.path.join(tmpdir, "model_index.json"), "w") as f:
            json.dump(model_index, f)
        return tmpdir

    def test_cosmos3_detected_by_class_name_in_model_index(self):
        from auto_round.utils.model import diffusion_load_model

        tmpdir = self._write_model_index("Cosmos3OmniDiffusersPipeline")
        try:
            with patch(
                "auto_round.special_model_handler.load_cosmos3_diffusion"
            ) as mock_load:
                mock_load.return_value = (MagicMock(), MagicMock())
                result = diffusion_load_model(tmpdir)

                mock_load.assert_called_once()
                assert result is mock_load.return_value
        finally:
            shutil.rmtree(tmpdir)

    def test_cosmos3_detected_by_alternate_class_name(self):
        from auto_round.utils.model import diffusion_load_model

        tmpdir = self._write_model_index("Cosmos3OmniPipeline")
        try:
            with patch(
                "auto_round.special_model_handler.load_cosmos3_diffusion"
            ) as mock_load:
                mock_load.return_value = (MagicMock(), MagicMock())
                result = diffusion_load_model(tmpdir)

                mock_load.assert_called_once()
                assert result is mock_load.return_value
        finally:
            shutil.rmtree(tmpdir)

    def test_non_cosmos3_pipeline_skips_cosmos3_loader(self):
        """Routing logic: non-Cosmos class names must not invoke load_cosmos3_diffusion.

        We mock out the actual diffusers pipeline loading to keep this a pure
        unit test on the dispatch logic in diffusion_load_model().
        """
        from auto_round.utils.model import diffusion_load_model

        tmpdir = self._write_model_index("DDPMScheduler")
        try:
            with patch(
                "auto_round.special_model_handler.load_cosmos3_diffusion"
            ) as mock_cosmos:
                with patch(
                    "auto_round.utils.common.LazyImport", return_value=MagicMock()
                ):
                    try:
                        diffusion_load_model(tmpdir)
                    except Exception:
                        pass
                    mock_cosmos.assert_not_called()
        finally:
            shutil.rmtree(tmpdir)
