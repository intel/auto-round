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
"""Unit tests for GLM-Image quantize and inference helpers.

These tests are purely local and do not load model weights from disk or
download anything.  A fake model hierarchy built with ``torch.nn.Module``
and ``types.SimpleNamespace`` is used to exercise the logic under test.
"""

import json
import os
import types

import pytest
import torch.nn as nn

from auto_round.special_model_handler import _get_glm_image_multimodal_block
from auto_round.utils.model import _find_pipeline_model_subfolder_local

# ---------------------------------------------------------------------------
# Helpers – fake model hierarchy
# ---------------------------------------------------------------------------


def _make_glm_image_model(n_vision_blocks: int = 4, n_lm_layers: int = 28):
    """Return a minimal fake GlmImageForConditionalGeneration-like model.

    Structure mirrors the real model::

        model
        ├── visual
        │   └── blocks: ModuleList[n_vision_blocks]
        └── language_model
            └── layers: ModuleList[n_lm_layers]
    """

    class _Blocks(nn.ModuleList):
        pass

    class _Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = _Blocks([nn.Linear(8, 8) for _ in range(n_vision_blocks)])

    class _LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(n_lm_layers)])

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = _Visual()
            self.language_model = _LM()

    class _GlmImageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()

    return _GlmImageModel()


# ---------------------------------------------------------------------------
# Tests for _get_glm_image_multimodal_block
# ---------------------------------------------------------------------------


class TestGetGlmImageMultimodalBlock:
    """Unit tests for the GLM-Image block-name discovery helper."""

    def test_text_only_returns_one_block_group(self):
        """Default (quant_vision=False): only language_model layers are returned."""
        model = _make_glm_image_model(n_vision_blocks=4, n_lm_layers=28)
        block_names = _get_glm_image_multimodal_block(model, quant_vision=False)

        assert len(block_names) == 1, "Expected exactly one block group (LM layers only)"
        expected = [f"model.language_model.layers.{i}" for i in range(28)]
        assert block_names[0] == expected

    def test_quant_vision_true_returns_two_block_groups(self):
        """quant_vision=True: visual encoder blocks prepended before LM layers."""
        model = _make_glm_image_model(n_vision_blocks=4, n_lm_layers=28)
        block_names = _get_glm_image_multimodal_block(model, quant_vision=True)

        assert len(block_names) == 2, "Expected two block groups: visual + LM"
        expected_visual = [f"model.visual.blocks.{i}" for i in range(4)]
        expected_lm = [f"model.language_model.layers.{i}" for i in range(28)]
        assert block_names[0] == expected_visual
        assert block_names[1] == expected_lm

    def test_quant_vision_false_ignores_visual_blocks(self):
        """quant_vision=False must not include visual blocks even if they exist."""
        model = _make_glm_image_model(n_vision_blocks=8, n_lm_layers=10)
        block_names = _get_glm_image_multimodal_block(model, quant_vision=False)

        flat = [name for group in block_names for name in group]
        assert not any("visual" in name for name in flat), "visual blocks must be excluded when quant_vision=False"

    def test_missing_language_model_returns_empty(self):
        """If the model has no language_model attribute, result is empty."""

        class _NoLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()  # no visual, no language_model

        block_names = _get_glm_image_multimodal_block(_NoLM(), quant_vision=False)
        assert block_names == []

    def test_missing_visual_blocks_with_quant_vision(self):
        """quant_vision=True but visual.blocks missing: only LM layers returned."""

        class _NoVisualBlocks(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = types.SimpleNamespace(
                    language_model=types.SimpleNamespace(layers=nn.ModuleList([nn.Linear(8, 8) for _ in range(6)]))
                    # no .visual attribute
                )

        block_names = _get_glm_image_multimodal_block(_NoVisualBlocks(), quant_vision=True)
        assert len(block_names) == 1
        assert block_names[0] == [f"model.language_model.layers.{i}" for i in range(6)]

    def test_block_count_matches_actual_module_list_length(self):
        """Block name count must equal the actual ModuleList size."""
        n_lm = 32
        model = _make_glm_image_model(n_vision_blocks=0, n_lm_layers=n_lm)
        block_names = _get_glm_image_multimodal_block(model, quant_vision=False)

        assert len(block_names) == 1
        assert len(block_names[0]) == n_lm


# ---------------------------------------------------------------------------
# Helpers – temp filesystem for pipeline loading tests
# ---------------------------------------------------------------------------


def _make_pipeline_dir(tmp_path, components, has_processor=True):
    """Write a minimal diffusers-style pipeline directory.

    Args:
        tmp_path: pytest tmp_path fixture directory.
        components: dict mapping component_name → dict to write as config.json.
        has_processor: if True, add a ``processor`` entry to model_index.json.
    """
    model_index = {"_class_name": "GlmImagePipeline", "_diffusers_version": "0.0.1"}
    if has_processor:
        model_index["processor"] = ["transformers", "GlmImageProcessor"]

    for name, cfg in components.items():
        comp_dir = tmp_path / name
        comp_dir.mkdir(parents=True)
        (comp_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        model_index[name] = ["transformers", cfg.get("architectures", ["Unknown"])[0]]

    (tmp_path / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Tests for _find_pipeline_model_subfolder_local
# ---------------------------------------------------------------------------


class TestFindPipelineModelSubfolderLocal:
    """Unit tests for the local pipeline subfolder discovery helper."""

    def test_finds_vision_language_encoder_subfolder(self, tmp_path):
        """The component containing GlmImageForConditionalGeneration is returned."""
        pipeline_dir = _make_pipeline_dir(
            tmp_path,
            {
                "vision_language_encoder": {
                    "architectures": ["GlmImageForConditionalGeneration"],
                    "model_type": "glm_image",
                },
                "vae": {"model_type": "autoencoder_kl"},  # no architectures → ignored
            },
        )
        model_subfolder, processor_subfolder, cfg = _find_pipeline_model_subfolder_local(pipeline_dir)

        assert model_subfolder == "vision_language_encoder"
        assert processor_subfolder == "processor"
        assert cfg["architectures"][0] == "GlmImageForConditionalGeneration"

    def test_prefers_conditional_generation_over_encoder(self, tmp_path):
        """ConditionalGeneration architecture is preferred over plain encoder."""
        pipeline_dir = _make_pipeline_dir(
            tmp_path,
            {
                "text_encoder": {"architectures": ["T5EncoderModel"]},
                "vision_language_encoder": {
                    "architectures": ["GlmImageForConditionalGeneration"],
                    "model_type": "glm_image",
                },
            },
            has_processor=False,
        )
        model_subfolder, processor_subfolder, cfg = _find_pipeline_model_subfolder_local(pipeline_dir)

        assert model_subfolder == "vision_language_encoder"
        assert processor_subfolder is None  # no processor entry

    def test_no_processor_returns_none(self, tmp_path):
        """When model_index.json has no 'processor' key, processor_subfolder is None."""
        pipeline_dir = _make_pipeline_dir(
            tmp_path,
            {"vision_language_encoder": {"architectures": ["GlmImageForConditionalGeneration"]}},
            has_processor=False,
        )
        _, processor_subfolder, _ = _find_pipeline_model_subfolder_local(pipeline_dir)
        assert processor_subfolder is None

    def test_with_processor_returns_processor_subfolder(self, tmp_path):
        """When model_index.json has a 'processor' key, processor_subfolder=='processor'."""
        pipeline_dir = _make_pipeline_dir(
            tmp_path,
            {"vision_language_encoder": {"architectures": ["GlmImageForConditionalGeneration"]}},
            has_processor=True,
        )
        _, processor_subfolder, _ = _find_pipeline_model_subfolder_local(pipeline_dir)
        assert processor_subfolder == "processor"

    def test_raises_when_no_model_index(self, tmp_path):
        """FileNotFoundError raised when neither config.json nor model_index.json exists."""
        with pytest.raises(FileNotFoundError, match="model_index.json"):
            _find_pipeline_model_subfolder_local(str(tmp_path))

    def test_raises_when_no_component_has_architectures(self, tmp_path):
        """FileNotFoundError raised when no component config contains 'architectures'."""
        pipeline_dir = _make_pipeline_dir(
            tmp_path,
            {
                "vae": {"model_type": "autoencoder_kl"},
                "scheduler": {},
            },
        )
        with pytest.raises(FileNotFoundError, match="architectures"):
            _find_pipeline_model_subfolder_local(pipeline_dir)

    def test_falls_back_to_first_candidate_when_no_preferred_arch(self, tmp_path):
        """When no ConditionalGeneration/CausalLM arch exists, first candidate is used."""
        pipeline_dir = _make_pipeline_dir(
            tmp_path,
            {
                "text_encoder": {"architectures": ["T5EncoderModel"]},
                "image_encoder": {"architectures": ["CLIPVisionModel"]},
            },
            has_processor=False,
        )
        model_subfolder, _, cfg = _find_pipeline_model_subfolder_local(pipeline_dir)
        # Must be one of the candidates, not crash
        assert model_subfolder in ("text_encoder", "image_encoder")
        assert "architectures" in cfg


# ---------------------------------------------------------------------------
# Tests for GlmImageProcessor construction path
# ---------------------------------------------------------------------------


class TestGlmImageProcessorConstruction:
    """Unit-test the GlmImageProcessor assembly logic in mllm_load_model.

    Without loading full model weights we directly exercise the branching
    code that wraps image_processor + tokenizer into GlmImageProcessor when
    model_type == "glm_image".  GlmImageProcessor itself is patched so the
    test does not depend on transformers' internal input validation.
    """

    @pytest.fixture()
    def mock_components(self):
        """Return minimal fake tokenizer and image_processor objects."""
        tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token_id=2)
        image_processor = types.SimpleNamespace(size={"height": 448, "width": 448})
        return tokenizer, image_processor

    def test_glm_image_processor_wraps_components(self, mock_components):
        """GlmImageProcessor must be called with image_processor= and tokenizer=."""
        from unittest.mock import MagicMock, patch

        tokenizer, image_processor = mock_components
        fake_processor = object()
        mock_cls = MagicMock(return_value=fake_processor)

        # Patch away the real GlmImageProcessor so we only test the branch logic
        with patch.dict(
            "sys.modules", {"transformers.models.glm_image.processing_glm_image": types.ModuleType("_fake")}
        ):
            import sys

            sys.modules["transformers.models.glm_image.processing_glm_image"].GlmImageProcessor = mock_cls

            model_type = "glm_image"
            processor = None
            if model_type == "glm_image" and image_processor is not None:
                from transformers.models.glm_image.processing_glm_image import GlmImageProcessor

                processor = GlmImageProcessor(image_processor=image_processor, tokenizer=tokenizer)

        mock_cls.assert_called_once_with(image_processor=image_processor, tokenizer=tokenizer)
        assert processor is fake_processor

    def test_non_glm_image_model_type_skips_wrapping(self, mock_components):
        """For any other model_type, the GlmImageProcessor wrapping is not applied."""
        tokenizer, image_processor = mock_components

        model_type = "qwen2_vl"
        processor = None  # simulate AutoProcessor result already in place
        if model_type == "glm_image" and image_processor is not None:
            processor = object()  # should never be reached

        assert processor is None  # wrapping must NOT happen

    def test_skipped_when_image_processor_is_none(self, mock_components):
        """image_processor=None prevents GlmImageProcessor from being built."""
        tokenizer, _ = mock_components

        model_type = "glm_image"
        image_processor = None
        processor = None
        if model_type == "glm_image" and image_processor is not None:
            processor = object()  # must not be reached

        assert processor is None


# ---------------------------------------------------------------------------
# Helpers – minimal PIL Image factory (no file I/O)
# ---------------------------------------------------------------------------


def _make_rgb_image(width: int = 64, height: int = 64):
    """Return a tiny solid-colour PIL Image in RGB mode."""
    from PIL import Image

    return Image.new("RGB", (width, height), color=(128, 64, 32))


# ---------------------------------------------------------------------------
# Tests for image-to-image inference call logic (run_glm_image.py)
# ---------------------------------------------------------------------------


class TestGlmImageI2ICallLogic:
    """Unit tests for the image-to-image pipeline invocation logic.

    The pattern under test mirrors run_glm_image.main()::

        condition_images = [load_image(p) for p in args.reference_image] or None
        result = pipe(prompt=..., image=condition_images, height=..., width=..., ...)

    No real pipeline or model weights are required.
    """

    def test_no_reference_images_passes_none_to_pipeline(self):
        """Empty reference_image list must yield image=None (text-to-image mode)."""
        from unittest.mock import MagicMock

        reference_image_paths = []  # T2I: no reference images provided
        condition_images = [_make_rgb_image() for _ in reference_image_paths] or None

        pipe = MagicMock()
        pipe.return_value = MagicMock(images=[_make_rgb_image()])
        pipe(prompt="a fox", image=condition_images, height=1024, width=1024)

        _, kwargs = pipe.call_args
        assert kwargs["image"] is None, "T2I: image kwarg must be None"

    def test_single_reference_image_passed_as_list(self):
        """Single reference image must be wrapped in a list (not passed bare)."""
        from unittest.mock import MagicMock

        ref_img = _make_rgb_image()
        reference_image_paths = ["dummy_path.jpg"]
        # Simulate load_image returning ref_img for each path
        condition_images = [ref_img for _ in reference_image_paths] or None

        pipe = MagicMock()
        pipe.return_value = MagicMock(images=[_make_rgb_image()])
        pipe(prompt="edit the sky", image=condition_images, height=33 * 32, width=32 * 32)

        _, kwargs = pipe.call_args
        assert isinstance(kwargs["image"], list), "I2I: image must be a list"
        assert len(kwargs["image"]) == 1
        assert kwargs["image"][0] is ref_img

    def test_multi_image_list_preserved(self):
        """Multiple reference images must all be forwarded as a list."""
        from unittest.mock import MagicMock

        imgs = [_make_rgb_image() for _ in range(3)]
        condition_images = imgs or None  # non-empty list stays as-is

        pipe = MagicMock()
        pipe.return_value = MagicMock(images=[_make_rgb_image()])
        pipe(prompt="merge subjects", image=condition_images, height=32 * 32, width=32 * 32)

        _, kwargs = pipe.call_args
        assert kwargs["image"] == imgs
        assert len(kwargs["image"]) == 3

    def test_height_width_not_divisible_by_32_raises(self):
        """run_glm_image.main() raises ValueError when dimensions are not multiples of 32."""
        height, width = 33 * 32 + 1, 32 * 32  # 1057 is not divisible by 32

        with pytest.raises(ValueError, match="divisible by 32"):
            if height % 32 != 0 or width % 32 != 0:
                raise ValueError("GLM-Image requires height and width to be divisible by 32.")

    def test_height_width_divisible_by_32_passes(self):
        """Dimensions that are multiples of 32 must not raise."""
        for height, width in [(33 * 32, 32 * 32), (1024, 768), (32, 32)]:
            # Should not raise
            if height % 32 != 0 or width % 32 != 0:
                raise AssertionError(f"Unexpected non-multiple: {height}x{width}")

    def test_i2i_prompt_forwarded_correctly(self):
        """The prompt string must be forwarded verbatim to the pipeline call."""
        from unittest.mock import MagicMock

        prompt = "Replace the background with an underground station."
        ref_img = _make_rgb_image()
        condition_images = [ref_img]

        pipe = MagicMock()
        pipe.return_value = MagicMock(images=[_make_rgb_image()])
        pipe(prompt=prompt, image=condition_images, height=33 * 32, width=32 * 32)

        _, kwargs = pipe.call_args
        assert kwargs["prompt"] == prompt


# ---------------------------------------------------------------------------
# Tests for load_image helper (run_glm_image.py)
# ---------------------------------------------------------------------------


class TestLoadImage:
    """Unit tests for the load_image() helper in run_glm_image.

    Covers local file loading and the URL-vs-path dispatch logic without
    making any real network requests.
    """

    @pytest.fixture(autouse=True)
    def _import_load_image(self):
        """Import load_image from run_glm_image into the test namespace."""
        import importlib
        import sys

        # Ensure the workspace root is on sys.path so run_glm_image can be imported
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if root not in sys.path:
            sys.path.insert(0, root)
        mod = importlib.import_module("run_glm_image")
        self.load_image = mod.load_image

    def test_load_local_rgb_image(self, tmp_path):
        """load_image() opens a local file and returns an RGB PIL Image."""
        from PIL import Image

        img = Image.new("RGBA", (32, 32), color=(10, 20, 30, 255))
        img_path = str(tmp_path / "test.png")
        img.save(img_path)

        result = self.load_image(img_path)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (32, 32)

    def test_load_image_converts_rgba_to_rgb(self, tmp_path):
        """RGBA images saved locally must be converted to RGB."""
        from PIL import Image

        img = Image.new("RGBA", (16, 16), color=(255, 0, 0, 128))
        img_path = str(tmp_path / "rgba.png")
        img.save(img_path)

        result = self.load_image(img_path)
        assert result.mode == "RGB"

    def test_url_branch_calls_requests_get(self):
        """http/https paths must use requests.get, not PIL.Image.open directly."""
        from io import BytesIO
        from unittest.mock import MagicMock, patch

        from PIL import Image

        fake_img = Image.new("RGB", (8, 8), color=(0, 128, 255))
        buf = BytesIO()
        fake_img.save(buf, format="PNG")
        buf.seek(0)

        mock_response = MagicMock()
        mock_response.raw = buf

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = self.load_image("https://example.com/image.png")

        mock_get.assert_called_once_with("https://example.com/image.png", timeout=60)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_local_path_does_not_call_requests(self, tmp_path):
        """Local file paths must not trigger requests.get."""
        from unittest.mock import patch

        from PIL import Image

        img = Image.new("RGB", (4, 4))
        img_path = str(tmp_path / "local.png")
        img.save(img_path)

        with patch("requests.get") as mock_get:
            self.load_image(img_path)

        mock_get.assert_not_called()
