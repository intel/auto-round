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

from __future__ import annotations

from enum import Enum

from auto_round.utils import LazyImport

gguf = LazyImport("gguf")


_GGUF_TYPE_TO_QTYPE_NAME = {
    "gguf:fp16": "F16",
    "gguf:f16": "F16",
    "gguf:bf16": "BF16",
    "gguf:f32": "F32",
    "gguf:q2_k": "Q2_K",
    "gguf:q3_k": "Q3_K",
    "gguf:q4_k": "Q4_K",
    "gguf:q5_k": "Q5_K",
    "gguf:q6_k": "Q6_K",
    "gguf:q4_0": "Q4_0",
    "gguf:q4_1": "Q4_1",
    "gguf:q5_0": "Q5_0",
    "gguf:q5_1": "Q5_1",
    "gguf:q8_0": "Q8_0",
}


_QTYPE_NAME_TO_GGUF_TYPE = {qtype: gguf_type for gguf_type, qtype in _GGUF_TYPE_TO_QTYPE_NAME.items()}
_QTYPE_NAME_TO_GGUF_TYPE["F16"] = "gguf:fp16"


class TensorCategory(Enum):
    TOKEN_EMBD = "token_embd"
    ATTENTION_Q = "attn_q"
    ATTENTION_V = "attn_v"
    ATTENTION_K = "attn_k"
    ATTENTION_QKV = "attn_qkv"
    ATTENTION_KV_B = "attn_kv_b"
    ATTENTION_OUTPUT = "attn_output"
    FFN_UP = "ffn_up"
    FFN_GATE = "ffn_gate"
    FFN_DOWN = "ffn_down"
    OUTPUT = "output"
    OTHER = "other"


def _tensor_category(name: str) -> TensorCategory:
    if name == "output.weight":
        return TensorCategory.OUTPUT
    if name in ("token_embd.weight", "per_layer_token_embd.weight"):
        return TensorCategory.TOKEN_EMBD
    if "attn_qkv.weight" in name:
        return TensorCategory.ATTENTION_QKV
    if "attn_kv_b.weight" in name:
        return TensorCategory.ATTENTION_KV_B
    if "attn_v.weight" in name:
        return TensorCategory.ATTENTION_V
    if "attn_k.weight" in name:
        return TensorCategory.ATTENTION_K
    if "attn_q.weight" in name:
        return TensorCategory.ATTENTION_Q
    if "attn_output.weight" in name:
        return TensorCategory.ATTENTION_OUTPUT
    if "ffn_up" in name:
        return TensorCategory.FFN_UP
    if "ffn_gate" in name:
        return TensorCategory.FFN_GATE
    if "ffn_down" in name:
        return TensorCategory.FFN_DOWN
    return TensorCategory.OTHER


def _is_attn_v_like(category: TensorCategory) -> bool:
    return category in (TensorCategory.ATTENTION_V, TensorCategory.ATTENTION_QKV, TensorCategory.ATTENTION_KV_B)


def _use_more_bits(i_layer: int, n_layer: int) -> bool:
    return i_layer < n_layer // 8 or i_layer >= 7 * n_layer // 8 or (i_layer - n_layer // 8) % 3 == 2


def _get_layer_id(name: str, fallback: int) -> int:
    parts = name.split(".")
    if len(parts) > 1 and parts[0] == "blk" and parts[1].isdigit():
        return int(parts[1])
    return fallback


def _default_qtype(ftype):
    if ftype == gguf.LlamaFileType.ALL_F32:
        return gguf.GGMLQuantizationType.F32
    if ftype == gguf.LlamaFileType.MOSTLY_F16:
        return gguf.GGMLQuantizationType.F16
    if ftype == gguf.LlamaFileType.MOSTLY_BF16:
        return gguf.GGMLQuantizationType.BF16
    if ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
        return gguf.GGMLQuantizationType.Q8_0
    if ftype == gguf.LlamaFileType.MOSTLY_Q4_0:
        return gguf.GGMLQuantizationType.Q4_0
    if ftype == gguf.LlamaFileType.MOSTLY_Q4_1:
        return gguf.GGMLQuantizationType.Q4_1
    if ftype == gguf.LlamaFileType.MOSTLY_Q5_0:
        return gguf.GGMLQuantizationType.Q5_0
    if ftype == gguf.LlamaFileType.MOSTLY_Q5_1:
        return gguf.GGMLQuantizationType.Q5_1
    if ftype in (gguf.LlamaFileType.MOSTLY_Q2_K_S, gguf.LlamaFileType.MOSTLY_Q2_K):
        return gguf.GGMLQuantizationType.Q2_K
    if ftype in (
        gguf.LlamaFileType.MOSTLY_Q3_K_S,
        gguf.LlamaFileType.MOSTLY_Q3_K_M,
        gguf.LlamaFileType.MOSTLY_Q3_K_L,
    ):
        return gguf.GGMLQuantizationType.Q3_K
    if ftype in (gguf.LlamaFileType.MOSTLY_Q4_K_S, gguf.LlamaFileType.MOSTLY_Q4_K_M):
        return gguf.GGMLQuantizationType.Q4_K
    if ftype in (gguf.LlamaFileType.MOSTLY_Q5_K_S, gguf.LlamaFileType.MOSTLY_Q5_K_M):
        return gguf.GGMLQuantizationType.Q5_K
    if ftype == gguf.LlamaFileType.MOSTLY_Q6_K:
        return gguf.GGMLQuantizationType.Q6_K
    raise ValueError(f"Unknown file type: {ftype.name}")


def gguf_format_to_ftype(gguf_format: str):
    format_name = gguf_format.lower()
    if format_name == "gguf:q2_k_mixed":
        format_name = "gguf:q2_k_s"
    mapping = {
        "gguf:f32": gguf.LlamaFileType.ALL_F32,
        "gguf:fp16": gguf.LlamaFileType.MOSTLY_F16,
        "gguf:f16": gguf.LlamaFileType.MOSTLY_F16,
        "gguf:bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "gguf:q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
        "gguf:q4_1": gguf.LlamaFileType.MOSTLY_Q4_1,
        "gguf:q5_0": gguf.LlamaFileType.MOSTLY_Q5_0,
        "gguf:q5_1": gguf.LlamaFileType.MOSTLY_Q5_1,
        "gguf:q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "gguf:q2_k": gguf.LlamaFileType.MOSTLY_Q2_K,
        "gguf:q2_k_s": gguf.LlamaFileType.MOSTLY_Q2_K_S,
        "gguf:q3_k_s": gguf.LlamaFileType.MOSTLY_Q3_K_S,
        "gguf:q3_k_m": gguf.LlamaFileType.MOSTLY_Q3_K_M,
        "gguf:q3_k_l": gguf.LlamaFileType.MOSTLY_Q3_K_L,
        "gguf:q4_k_s": gguf.LlamaFileType.MOSTLY_Q4_K_S,
        "gguf:q4_k_m": gguf.LlamaFileType.MOSTLY_Q4_K_M,
        "gguf:q5_k_s": gguf.LlamaFileType.MOSTLY_Q5_K_S,
        "gguf:q5_k_m": gguf.LlamaFileType.MOSTLY_Q5_K_M,
        "gguf:q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
        "gguf:q6_k_s": gguf.LlamaFileType.MOSTLY_Q6_K,
    }
    if format_name not in mapping:
        raise ValueError(f"Unknown GGUF format: {gguf_format}")
    return mapping[format_name]


def qtype_to_gguf_type(qtype) -> str:
    return _QTYPE_NAME_TO_GGUF_TYPE[qtype.name]


class GGUFDTypeSelector:
    def __init__(
        self,
        hparams: dict,
        ftype,
        model_arch=None,
        n_layer: int | None = None,
        n_attention_wv: int | None = None,
        has_imatrix: bool = False,
        has_tied_embeddings: bool = False,
    ):
        self.hparams = hparams
        self.ftype = ftype
        self.model_arch = model_arch
        self.n_layer = n_layer
        self.n_attention_wv = n_attention_wv
        self.has_imatrix = has_imatrix
        self.has_tied_embeddings = has_tied_embeddings
        self.i_attention_wv = 0
        self.i_ffn_down = 0

    def select_qtype(self, name: str, n_dims: int, fallback_index: int = 0):
        """Select the GGML tensor dtype used by llama.cpp quantization.

        `name` must be the final GGUF tensor name. MTP/NextN tensors are already
        remapped to extended `blk.<id>` names by conversion classes, so they use
        the same category and layer-index rules as regular transformer blocks.
        """
        if n_dims < 2:
            return gguf.GGMLQuantizationType.F32

        qtype = _default_qtype(self.ftype)
        category = _tensor_category(name)
        i_layer = _get_layer_id(name, fallback_index)
        if self.n_layer is None:
            n_layer = int(
                self.hparams.get("num_hidden_layers", self.hparams.get("n_layer", self.hparams.get("num_layers", 1)))
            )
            n_layer += int(
                self.hparams.get("mtp_num_hidden_layers", self.hparams.get("num_nextn_predict_layers", 0)) or 0
            )
        else:
            n_layer = self.n_layer
        n_layer = max(n_layer, i_layer + 1)
        n_gqa = 1
        n_head = self.hparams.get("num_attention_heads", self.hparams.get("n_head"))
        n_head_kv = self.hparams.get("num_key_value_heads", self.hparams.get("n_head_kv"))
        if n_head and n_head_kv:
            n_gqa = max(1, int(n_head) // int(n_head_kv))
        n_expert = int(
            self.hparams.get(
                "num_experts", self.hparams.get("num_local_experts", self.hparams.get("n_routed_experts", 0))
            )
            or 0
        )
        # llama.cpp: output & token_embd handling (llama_tensor_get_type outer wrapper)
        if category == TensorCategory.OUTPUT or (self.has_tied_embeddings and category == TensorCategory.TOKEN_EMBD):
            # llama.cpp: if new_type != Q8_0, upgrade to Q6_K
            # Skip F32/F16/BF16 (non-quantized types that won't reach here in practice)
            if qtype not in (
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.BF16,
                gguf.GGMLQuantizationType.Q8_0,
            ):
                qtype = gguf.GGMLQuantizationType.Q6_K
        elif category == TensorCategory.TOKEN_EMBD:
            pass

        if _is_attn_v_like(category):
            if self.ftype == gguf.LlamaFileType.MOSTLY_Q2_K:
                qtype = gguf.GGMLQuantizationType.Q4_K if n_gqa >= 4 else gguf.GGMLQuantizationType.Q3_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q2_K_S and n_gqa >= 4:
                qtype = gguf.GGMLQuantizationType.Q4_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_M:
                qtype = gguf.GGMLQuantizationType.Q5_K if self.i_attention_wv < 2 else gguf.GGMLQuantizationType.Q4_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_L:
                qtype = gguf.GGMLQuantizationType.Q5_K
            elif self.ftype in (gguf.LlamaFileType.MOSTLY_Q4_K_M, gguf.LlamaFileType.MOSTLY_Q5_K_M) and _use_more_bits(
                self.i_attention_wv, self.n_attention_wv or n_layer
            ):
                qtype = gguf.GGMLQuantizationType.Q6_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q4_K_S and self.i_attention_wv < 4:
                qtype = gguf.GGMLQuantizationType.Q5_K
            if n_expert == 8:
                qtype = gguf.GGMLQuantizationType.Q8_0
            self.i_attention_wv += 1
        elif category == TensorCategory.ATTENTION_K and n_expert == 8:
            qtype = gguf.GGMLQuantizationType.Q8_0
        elif category == TensorCategory.FFN_DOWN:
            if self.ftype == gguf.LlamaFileType.MOSTLY_Q2_K:
                qtype = gguf.GGMLQuantizationType.Q3_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q2_K_S and i_layer < n_layer // 8:
                qtype = gguf.GGMLQuantizationType.Q4_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_M:
                if i_layer < n_layer // 16:
                    qtype = gguf.GGMLQuantizationType.Q5_K
                elif self.model_arch != gguf.MODEL_ARCH.FALCON or _use_more_bits(i_layer, n_layer):
                    qtype = gguf.GGMLQuantizationType.Q4_K
                else:
                    qtype = gguf.GGMLQuantizationType.Q3_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_L:
                qtype = (
                    gguf.GGMLQuantizationType.Q4_K
                    if self.model_arch == gguf.MODEL_ARCH.FALCON
                    else gguf.GGMLQuantizationType.Q5_K
                )
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q4_K_M:
                if self.model_arch == gguf.MODEL_ARCH.FALCON:
                    if i_layer < n_layer // 16:
                        qtype = gguf.GGMLQuantizationType.Q6_K
                    elif _use_more_bits(i_layer, n_layer):
                        qtype = gguf.GGMLQuantizationType.Q5_K
                elif _use_more_bits(i_layer, n_layer):
                    qtype = gguf.GGMLQuantizationType.Q6_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q5_K_M and _use_more_bits(i_layer, n_layer):
                qtype = gguf.GGMLQuantizationType.Q6_K
            elif (
                self.ftype == gguf.LlamaFileType.MOSTLY_Q4_K_S
                and self.model_arch != gguf.MODEL_ARCH.FALCON
                and i_layer < n_layer // 8
            ):
                qtype = gguf.GGMLQuantizationType.Q5_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q4_0 and self.has_imatrix and i_layer < n_layer // 8:
                qtype = gguf.GGMLQuantizationType.Q4_1
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q5_0 and self.has_imatrix and i_layer < n_layer // 8:
                qtype = gguf.GGMLQuantizationType.Q5_1
            self.i_ffn_down += 1
        elif category == TensorCategory.ATTENTION_OUTPUT:
            if self.model_arch != gguf.MODEL_ARCH.FALCON:
                if n_expert == 8:
                    if self.ftype in (
                        gguf.LlamaFileType.MOSTLY_Q2_K,
                        gguf.LlamaFileType.MOSTLY_Q3_K_S,
                        gguf.LlamaFileType.MOSTLY_Q3_K_M,
                        gguf.LlamaFileType.MOSTLY_Q4_K_S,
                        gguf.LlamaFileType.MOSTLY_Q4_K_M,
                    ):
                        qtype = gguf.GGMLQuantizationType.Q5_K
                elif self.ftype == gguf.LlamaFileType.MOSTLY_Q2_K:
                    qtype = gguf.GGMLQuantizationType.Q3_K
                elif self.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_M:
                    qtype = gguf.GGMLQuantizationType.Q4_K
                elif self.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_L:
                    qtype = gguf.GGMLQuantizationType.Q5_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_L:
                qtype = gguf.GGMLQuantizationType.Q4_K
        elif category == TensorCategory.ATTENTION_QKV:
            if self.ftype in (gguf.LlamaFileType.MOSTLY_Q3_K_M, gguf.LlamaFileType.MOSTLY_Q3_K_L):
                qtype = gguf.GGMLQuantizationType.Q4_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q4_K_M:
                qtype = gguf.GGMLQuantizationType.Q5_K
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q5_K_M:
                qtype = gguf.GGMLQuantizationType.Q6_K

        return qtype

    def select_gguf_type(self, name: str, n_dims: int, fallback_index: int = 0) -> str:
        return qtype_to_gguf_type(self.select_qtype(name, n_dims, fallback_index=fallback_index))


def select_llama_cpp_compatible_qtype(name: str, ftype, hparams: dict, n_dims: int, fallback_index: int = 0):
    selector = GGUFDTypeSelector(hparams, ftype)
    selector.i_attention_wv = fallback_index
    return selector.select_qtype(name, n_dims, fallback_index=fallback_index)
