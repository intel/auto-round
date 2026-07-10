from types import SimpleNamespace

import pytest

from auto_round.compressors.utils import _resolve_gguf_n_layers
from auto_round.export.export_to_gguf.gguf_dtype import (
    GGUFDTypeSelector,
    gguf_format_to_ftype,
    select_llama_cpp_compatible_qtype,
)
from auto_round.utils import LazyImport
from auto_round.utils.model import ModelType

gguf = LazyImport("gguf")


@pytest.mark.parametrize("layer_id", [0, 28])
def test_attn_v_dtype_uses_same_rule_for_main_and_mtp_layers(layer_id):
    hparams = {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 4}

    qtype = select_llama_cpp_compatible_qtype(
        f"blk.{layer_id}.attn_v.weight",
        gguf.LlamaFileType.MOSTLY_Q2_K_S,
        hparams,
        n_dims=2,
    )

    assert qtype == gguf.GGMLQuantizationType.Q4_K


def test_mtp_ffn_down_dtype_follows_llama_cpp_mixed_rule():
    hparams = {"num_hidden_layers": 28, "mtp_num_hidden_layers": 1, "num_attention_heads": 16, "num_key_value_heads": 4}

    qtype = select_llama_cpp_compatible_qtype(
        "blk.28.ffn_down.weight",
        gguf.LlamaFileType.MOSTLY_Q4_K_M,
        hparams,
        n_dims=2,
    )

    assert qtype == gguf.GGMLQuantizationType.Q6_K


def test_mtp_attn_output_dtype_follows_llama_cpp_mixed_rule():
    hparams = {"num_hidden_layers": 28, "mtp_num_hidden_layers": 1, "num_attention_heads": 16, "num_key_value_heads": 4}

    qtype = select_llama_cpp_compatible_qtype(
        "blk.28.attn_output.weight",
        gguf.LlamaFileType.MOSTLY_Q3_K_M,
        hparams,
        n_dims=2,
    )

    assert qtype == gguf.GGMLQuantizationType.Q4_K


def test_one_dim_mtp_tensors_stay_f32():
    hparams = {"num_hidden_layers": 28}

    qtype = select_llama_cpp_compatible_qtype(
        "blk.28.ffn_norm.weight",
        gguf.LlamaFileType.MOSTLY_Q4_K_M,
        hparams,
        n_dims=1,
    )

    assert qtype == gguf.GGMLQuantizationType.F32


def test_selector_tracks_attn_v_counter_for_q3_k_m():
    hparams = {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 4}
    selector = GGUFDTypeSelector(hparams, gguf_format_to_ftype("gguf:q3_k_m"))

    qtypes = [selector.select_qtype(f"blk.{idx}.attn_v.weight", n_dims=2) for idx in range(3)]

    assert qtypes == [
        gguf.GGMLQuantizationType.Q5_K,
        gguf.GGMLQuantizationType.Q5_K,
        gguf.GGMLQuantizationType.Q4_K,
    ]


def test_selector_returns_gguf_type_string_for_compressor_config():
    hparams = {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 4}
    selector = GGUFDTypeSelector(hparams, gguf_format_to_ftype("gguf:q2_k_s"))

    assert selector.select_gguf_type("blk.28.attn_v.weight", n_dims=2) == "gguf:q4_k"


def test_attn_qkv_matches_llama_cpp_attn_v_like_rule():
    hparams = {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 4}
    selector = GGUFDTypeSelector(hparams, gguf_format_to_ftype("gguf:q3_k_m"))

    qtype = selector.select_qtype("blk.0.attn_qkv.weight", n_dims=2)

    assert qtype == gguf.GGMLQuantizationType.Q5_K
    assert selector.i_attention_wv == 1


def test_selector_accepts_explicit_layer_count_from_compressor():
    hparams = {"num_hidden_layers": 2, "num_attention_heads": 16, "num_key_value_heads": 4}
    selector = GGUFDTypeSelector(hparams, gguf_format_to_ftype("gguf:q4_k_m"), n_layer=28)

    assert selector.select_gguf_type("blk.1.attn_v.weight", n_dims=2) == "gguf:q6_k"


def test_selector_uses_cpp_integer_layer_division():
    hparams = {"num_hidden_layers": 2, "num_attention_heads": 16, "num_key_value_heads": 4}
    selector = GGUFDTypeSelector(hparams, gguf_format_to_ftype("gguf:q4_k_m"), n_layer=2)

    assert selector.select_gguf_type("blk.1.ffn_down.weight", n_dims=2) == "gguf:q6_k"


def test_attn_v_more_bits_uses_attention_counter_not_layer_id():
    hparams = {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 4}
    selector = GGUFDTypeSelector(hparams, gguf_format_to_ftype("gguf:q4_k_m"), n_attention_wv=2)

    assert selector.select_gguf_type("blk.0.attn_v.weight", n_dims=2) == "gguf:q4_k"
    assert selector.select_gguf_type("blk.1.attn_v.weight", n_dims=2) == "gguf:q6_k"


def test_q3_k_m_ffn_down_uses_q4_k_for_non_falcon_like_llama_cpp():
    # llama.cpp: new_type = i_layer < n_layer/16 ? Q5_K
    #          : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? Q4_K : Q3_K
    # so for non-Falcon archs every ffn_down beyond the first n_layer/16 layers is Q4_K.
    hparams = {"num_hidden_layers": 33, "num_attention_heads": 16, "num_key_value_heads": 4}
    selector = GGUFDTypeSelector(hparams, gguf_format_to_ftype("gguf:q3_k_m"))

    qtypes = {i: selector.select_qtype(f"blk.{i}.ffn_down.weight", n_dims=2) for i in range(33)}

    assert qtypes[0] == qtypes[1] == gguf.GGMLQuantizationType.Q5_K
    assert all(qtypes[i] == gguf.GGMLQuantizationType.Q4_K for i in range(2, 33))


def test_q4_0_ffn_down_upgrade_requires_imatrix_like_llama_cpp():
    hparams = {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 4}
    selector = GGUFDTypeSelector(hparams, gguf_format_to_ftype("gguf:q4_0"), has_imatrix=False)

    assert selector.select_gguf_type("blk.0.ffn_down.weight", n_dims=2) == "gguf:q4_0"


def test_n_layer_resolution_reads_nested_text_config():
    # multimodal configs like Qwen3.5 expose num_hidden_layers only under text_config;
    # failing to resolve it silently disabled the whole llama.cpp k-quant mixture.
    config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=32),
        vision_config=SimpleNamespace(depth=24),
    )

    assert _resolve_gguf_n_layers(config, ModelType.TEXT) == (32, None)
    assert _resolve_gguf_n_layers(config, ModelType.MMPROJ) == (32, 24)


def test_n_layer_resolution_prefers_text_config_for_multimodal():
    config = SimpleNamespace(
        num_hidden_layers=48,
        text_config=SimpleNamespace(num_hidden_layers=32),
        vision_config=SimpleNamespace(depth=24),
    )

    assert _resolve_gguf_n_layers(config, ModelType.MMPROJ) == (32, 24)
    assert _resolve_gguf_n_layers(config, ModelType.TEXT) == (48, None)
