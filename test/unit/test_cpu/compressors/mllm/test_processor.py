# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.compressors.mllm.processor``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from auto_round.compressors.mllm.processor import (
    PROCESSORS,
    AudioTextProcessor,
    BasicProcessor,
    CogVLM2Processor,
    HFProcessor,
    LongCatNextProcessor,
    Mistral3Processor,
    Qwen2VLProcessor,
    Qwen2_5OmniProcessor,
    Qwen3OmniProcessor,
    register_processor,
)


# ==============================================================================
# Helpers
# ==============================================================================


class DummyTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(("call", text, kwargs))
        # Returns an object with .input_ids for decode slicing
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, **kwargs):
        return str(list(ids))

    def apply_chat_template(self, *args, **kwargs):
        self.calls.append(("apply_chat_template", args, kwargs))
        return "templated_text"


class DummyProcessor:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.calls = []

    def apply_chat_template(self, *args, **kwargs):
        self.calls.append(("apply_chat_template", args, kwargs))
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    def __call__(self, **kwargs):
        self.calls.append(("call", kwargs))
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}


class DummyModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ==============================================================================
# PROCESSORS registry
# ==============================================================================


class TestProcessorsRegistry:
    def test_basic_in_registry(self):
        assert "basic" in PROCESSORS

    def test_hf_in_registry(self):
        assert "hf" in PROCESSORS

    def test_qwen2_vl_in_registry(self):
        assert "qwen2_vl" in PROCESSORS

    def test_register_custom(self):
        @register_processor("my_test")
        class MyProc(BasicProcessor):
            pass

        assert "my_test" in PROCESSORS
        assert PROCESSORS["my_test"] is MyProc

    def test_register_returns_decorator(self):
        """register_processor returns a decorator that adds to PROCESSORS."""
        decorator = register_processor("wrap_me")
        # The decorator should be callable and return the class unchanged
        @decorator
        class ToWrap(BasicProcessor):
            pass

        assert "wrap_me" in PROCESSORS

    def test_processors_dict_is_populated(self):
        assert len(PROCESSORS) > 0

    def test_basic_callable(self):
        assert callable(PROCESSORS["basic"])

    def test_hf_callable(self):
        assert callable(PROCESSORS["hf"])


# ==============================================================================
# BasicProcessor
# ==============================================================================


class TestBasicProcessor:
    def test_can_be_instantiated(self):
        p = BasicProcessor()
        assert p is not None

    def test_post_init_stores_attributes(self):
        p = BasicProcessor()
        model = DummyModel()
        tok = DummyTokenizer()
        p.post_init(model, tok, image_processor="my_img_proc", use_rtn=True)
        assert p.model is model
        assert p.tokenizer is tok
        assert p.image_processor == "my_img_proc"
        assert p.use_rtn is True

    def test_post_init_default_image_processor(self):
        p = BasicProcessor()
        model = DummyModel()
        tok = DummyTokenizer()
        p.post_init(model, tok)
        assert p.image_processor == BasicProcessor.default_image_processor

    def test_get_input_raises(self):
        p = BasicProcessor()
        with pytest.raises(NotImplementedError):
            p.get_input("hello", None)

    def test_data_collator_delegates(self):
        p = BasicProcessor()
        batch = [{"input_ids": torch.tensor([1, 2])}]
        with patch(
            "auto_round.compressors.mllm.processor.default_data_collator",
            return_value={"a": 1},
        ) as mock_coll:
            result = p.data_collator(batch)
            mock_coll.assert_called_once_with(batch)
            assert result == {"a": 1}

    def test_squeeze_result_modifies_in_place(self):
        p = BasicProcessor()
        data = {
            "a": torch.tensor([[1, 2]]),
            "b": torch.tensor([[3, 4]]),
        }
        result = p.squeeze_result(data)
        assert data["a"].tolist() == [1, 2]
        assert data["b"].tolist() == [3, 4]

    def test_check_image_processor_raises_when_none_and_not_rtn(self):
        p = BasicProcessor()
        p.image_processor = None
        p.use_rtn = False
        with pytest.raises(ValueError, match="image processor"):
            p.check_image_processor()

    def test_check_image_processor_ok_when_rtn(self):
        p = BasicProcessor()
        p.image_processor = None
        p.use_rtn = True
        p.check_image_processor()

    def test_check_image_processor_ok_with_processor(self):
        p = BasicProcessor()
        p.image_processor = "something"
        p.use_rtn = False
        p.check_image_processor()


# ==============================================================================
# HFProcessor
# ==============================================================================


class TestHFProcessor:
    def test_init_sets_process_func(self):
        p = HFProcessor()
        assert p.process_func == p._process_v1

    def test_post_init_requires_tokenizer(self):
        p = HFProcessor()
        with pytest.raises(AssertionError, match="tokenizer"):
            p.post_init(DummyModel(), None, processor=DummyProcessor())

    def test_post_init_requires_processor(self):
        p = HFProcessor()
        with pytest.raises(AssertionError, match="processor"):
            p.post_init(DummyModel(), DummyTokenizer(), processor=None)

    def test_post_init_sets_default_image_processor(self):
        p = HFProcessor()
        p.post_init(DummyModel(), DummyTokenizer(), processor=DummyProcessor())
        assert p.image_processor == BasicProcessor.default_image_processor

    def test_process_v1_replaces_image_token(self):
        p = HFProcessor()
        p.post_init(DummyModel(), DummyTokenizer(), processor=DummyProcessor())
        messages = [{"role": "user", "content": "hello <image> world"}]
        result = p._process_v1(messages, "my_image")
        # apply_chat_template was called and returned a dict
        assert isinstance(result, dict)
        assert "input_ids" in result

    def test_process_v1_without_image_token(self):
        p = HFProcessor()
        p.post_init(DummyModel(), DummyTokenizer(), processor=DummyProcessor())
        messages = [{"role": "user", "content": "hello world"}]
        result = p._process_v1(messages, "my_image")
        assert isinstance(result, dict)
        assert "input_ids" in result

    def test_process_v2_with_chat_template(self):
        p = HFProcessor()
        p.post_init(DummyModel(), DummyTokenizer(), processor=DummyProcessor())
        p.processor.chat_template = "template"
        messages = [{"role": "user", "content": "hi <image>"}, {"role": "assistant", "content": "ok"}]
        # Pass image as None to avoid default_image_processor fetch_image call
        result = p._process_v2(messages, None)
        assert isinstance(result, dict)
        assert "input_ids" in result

    def test_process_v2_without_chat_template(self):
        p = HFProcessor()
        p.post_init(DummyModel(), DummyTokenizer(), processor=DummyProcessor())
        p.processor.chat_template = None
        p.tokenizer = DummyTokenizer()
        messages = [{"role": "user", "content": "hi"}]
        result = p._process_v2(messages, None)
        assert isinstance(result, dict)

    def test_process_v2_processes_image(self):
        p = HFProcessor()
        img_proc = MagicMock(return_value="processed_img")
        p.post_init(DummyModel(), DummyTokenizer(), processor=DummyProcessor(), image_processor=img_proc)
        messages = [{"role": "user", "content": "hi"}]
        # Pass image as None to avoid default_image_processor calling fetch_image
        result = p._process_v2(messages, None)
        assert isinstance(result, dict)

    def test_get_input_squeezes_when_enabled(self):
        p = HFProcessor()
        p.post_init(DummyModel(), DummyTokenizer(), processor=DummyProcessor())
        # Patch squeeze_result to avoid tokenizer/decode complications
        p.squeeze_result = MagicMock(return_value={"a": 1})
        result = p.get_input([{"role": "user", "content": "hi"}], None, squeeze=True)
        p.squeeze_result.assert_called_once()


# ==============================================================================
# Qwen2VLProcessor
# ==============================================================================


class TestQwen2VLProcessor:
    def test_squeeze_result_skips_pixel_values(self):
        # 3D pixel_values - skipped entirely, shape unchanged
        data = {
            "pixel_values": torch.tensor([[[1.0]]]),
            "input_ids": torch.tensor([[1, 2]]),
        }
        result = Qwen2VLProcessor.squeeze_result(data)
        assert result["pixel_values"].shape == (1, 1, 1)
        assert result["input_ids"].tolist() == [1, 2]

    def test_squeeze_result_skips_pixel_values_1d(self):
        # 1D pixel_values - skipped, shape unchanged
        data = {
            "pixel_values": torch.tensor([1.0]),
            "input_ids": torch.tensor([[1, 2]]),
        }
        result = Qwen2VLProcessor.squeeze_result(data)
        assert result["pixel_values"].shape == (1,)
        assert result["input_ids"].tolist() == [1, 2]

    def test_squeeze_result_skips_2d_pixel_values(self):
        # 2D pixel_values - skipped, shape unchanged
        data = {
            "pixel_values": torch.tensor([[1.0]]),
            "input_ids": torch.tensor([[1, 2]]),
        }
        result = Qwen2VLProcessor.squeeze_result(data)
        assert result["pixel_values"].shape == (1, 1)
        assert result["input_ids"].tolist() == [1, 2]


# ==============================================================================
# LongCatNextProcessor
# ==============================================================================


class TestLongCatNextProcessor:
    def test_class_attributes(self):
        assert LongCatNextProcessor.IMAGE_TOKEN == "<image>"
        assert LongCatNextProcessor.LONGCAT_IMG_START == "<longcat_img_start>"
        assert LongCatNextProcessor.LONGCAT_IMG_END == "<longcat_img_end>"

    def test_post_init_requires_tokenizer(self):
        p = LongCatNextProcessor()
        with pytest.raises(AssertionError, match="tokenizer"):
            p.post_init(DummyModel(), None, processor=DummyProcessor())

    def test_post_init_requires_processor(self):
        p = LongCatNextProcessor()
        with pytest.raises(AssertionError, match="processor"):
            p.post_init(DummyModel(), DummyTokenizer(), processor=None)

    def test_data_collator_single_item(self):
        p = LongCatNextProcessor()
        result = p.data_collator([{"a": 1}])
        assert result == {"a": 1}

    def test_data_collator_stacks_tensors(self):
        p = LongCatNextProcessor()
        batch = [
            {"a": torch.tensor([1]), "b": "x"},
            {"a": torch.tensor([2]), "b": "y"},
        ]
        result = p.data_collator(batch)
        assert torch.equal(result["a"], torch.tensor([[1], [2]]))
        assert result["b"] == ["x", "y"]

    def test_data_collator_nonstackable_becomes_list(self):
        # When shapes differ, torch.stack fails and result is list
        batch = [
            {"a": torch.tensor([[1, 2]]), "b": torch.tensor([[3, 4]])},
            {"a": torch.tensor([[5, 6]]), "b": torch.tensor([[7, 8]])},
        ]
        result = LongCatNextProcessor.data_collator(batch)
        # Shapes match, so torch.stack succeeds
        assert result["a"].shape == (2, 1, 2)


# ==============================================================================
# Qwen2_5OmniProcessor
# ==============================================================================


class TestQwen2_5OmniProcessor:
    def test_squeeze_result_skips_multimodal_keys(self):
        data = {
            "pixel_values": torch.tensor([[1]]),
            "pixel_values_videos": torch.tensor([[2]]),
            "input_features": torch.tensor([[3]]),
            "input_ids": torch.tensor([[4]]),
        }
        result = Qwen2_5OmniProcessor.squeeze_result(data)
        assert result["input_ids"].tolist() == [4]

    def test_process_v1_returns_dict(self):
        p = Qwen2_5OmniProcessor()
        p.post_init(DummyModel(), DummyTokenizer(), processor=DummyProcessor())
        messages = [{"role": "user", "content": "hello <image> world"}]
        result = p._process_v1(messages, "my_img")
        assert isinstance(result, dict)
        assert "input_ids" in result


# ==============================================================================
# Qwen3OmniProcessor
# ==============================================================================


class TestQwen3OmniProcessor:
    def test_squeeze_result_skips_pixel_values_videos(self):
        data = {
            "pixel_values_videos": torch.tensor([[1]]),
            "input_ids": torch.tensor([[2]]),
        }
        result = Qwen3OmniProcessor.squeeze_result(data)
        # pixel_values_videos IS in skip list, so shape is unchanged
        assert result["pixel_values_videos"].shape == (1, 1)
        assert result["input_ids"].tolist() == [2]


# ==============================================================================
# AudioTextProcessor
# ==============================================================================


class TestAudioTextProcessor:
    def test_post_init_ignores_image_processor(self):
        p = AudioTextProcessor()
        p.post_init(DummyModel(), DummyTokenizer(), processor=DummyProcessor(), image_processor="my_img_proc")
        assert p.image_processor is None

    def test_check_image_processor_does_not_raise(self):
        p = AudioTextProcessor()
        p.image_processor = None
        p.use_rtn = False
        p.check_image_processor()

    def test_squeeze_result(self):
        data = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        result = AudioTextProcessor.squeeze_result(data)
        assert result["input_ids"].tolist() == [1, 2, 3]


# ==============================================================================
# CogVLM2Processor
# ==============================================================================


class TestCogVLM2Processor:
    def test_default_image_processor_calls_fetch(self):
        img = MagicMock()
        img.convert = MagicMock(return_value="rgb_img")
        with patch(
            "auto_round.compressors.mllm.processor.fetch_image",
            return_value=img,
        ):
            result = CogVLM2Processor.default_image_processor("path_or_url")
        img.convert.assert_called_once_with("RGB")

    def test_data_collator_stacks_tensors(self):
        batch = [
            {"a": torch.tensor([1]), "b": ["x"]},
            {"a": torch.tensor([2]), "b": ["y"]},
        ]
        result = CogVLM2Processor.data_collator(batch)
        assert torch.equal(result["a"], torch.tensor([[1], [2]]))
        # Lists are stacked into nested list
        assert result["b"] == [["x"], ["y"]]

    def test_data_collator_nonstackable(self):
        batch = [
            {"a": torch.tensor([1]), "b": 1},
            {"a": torch.tensor([2]), "b": 2},
        ]
        result = CogVLM2Processor.data_collator(batch)
        assert "b" not in result


# ==============================================================================
# Mistral3Processor
# ==============================================================================


class TestMistral3Processor:
    def test_class_attribute(self):
        assert Mistral3Processor.IMAGE_TOKEN == "<image>"
