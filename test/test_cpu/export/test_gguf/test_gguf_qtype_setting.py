import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ....helpers import get_model_path


class TestGGUFQTypeSetting:

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_qtype_setting(self):
        # Qwen2.5-0.5B-Instruct no output, token_embed q6_k fallbakc to q8_0 336M
        # Qwen3-0.6B output q6_k, token_embed q4_0  448M
        # Qwen3-8B output q6_k, token_embed q4_0 4.5G
        # Llama-3.2-1B-Instruct o output, token_embed q6_k 736M
        from auto_round.compressors.utils import set_layer_config
        from auto_round.export.export_to_gguf.config import ModelType

        model_name = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
        ar = AutoRound(model=model_name, scheme="gguf:q4_0", iters=0)
        ar.formats = ["gguf:q4_0"]
        ar.layer_config, _, _ = set_layer_config(
            ar.model,
            ar.layer_config,
            ar.scheme,
            ar.scale_dtype,
            ar.supported_types,
            ar.inner_supported_types,
            ar.quant_block_list,
            ar.ignore_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        assert ar.layer_config["model.embed_tokens"]["bits"] == 8
        assert "lm_head" not in ar.layer_config

        model_name = "Qwen/Qwen3-0.6B"
        ar = AutoRound(model=model_name, scheme="gguf:q4_0", iters=0)
        ar.formats = ["gguf:q4_0"]
        ar.layer_config, _, _ = set_layer_config(
            ar.model,
            ar.layer_config,
            ar.scheme,
            ar.scale_dtype,
            ar.supported_types,
            ar.inner_supported_types,
            ar.quant_block_list,
            ar.ignore_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        assert ar.layer_config["model.embed_tokens"]["bits"] == 4
        assert ar.layer_config["lm_head"]["bits"] == 6 and ar.layer_config["lm_head"]["super_bits"] == 8

        layer_config = {
            "model.embed_tokens": {"bits": 6, "super_bits": 8},
            "lm_head": {"bits": 4},
        }
        ar = AutoRound(model=model_name, scheme="gguf:q4_0", iters=0, layer_config=layer_config)
        ar.formats = ["gguf:q4_0"]
        ar.layer_config, _, _ = set_layer_config(
            ar.model,
            ar.layer_config,
            ar.scheme,
            ar.scale_dtype,
            ar.supported_types,
            ar.inner_supported_types,
            ar.quant_block_list,
            ar.ignore_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        assert (
            ar.layer_config["lm_head"]["bits"] == 4
            and ar.layer_config["model.embed_tokens"]["bits"] == 6
            and ar.layer_config["model.embed_tokens"]["super_bits"] == 8
        )
