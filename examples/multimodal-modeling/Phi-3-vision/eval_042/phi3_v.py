from auto_gptq.modeling._base import BaseGPTQForCausalLM
from auto_gptq.utils.import_utils import compare_transformers_version
from auto_gptq.modeling._const import SUPPORTED_MODELS
from auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP

class Phi3VGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "Phi3DecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.final_layernorm"] # "model.vision_embed_tokens"?
    inside_layer_modules = [
        ["self_attn.o_proj"],
        ["self_attn.qkv_proj"],
        ["mlp.gate_up_proj"],
        ["mlp.down_proj"],
    ]

__all__ = ["Phi3VGPTQForCausalLM"]

if compare_transformers_version("v4.41.0", op="ge"):
    SUPPORTED_MODELS.append("phi3_v")
GPTQ_CAUSAL_LM_MODEL_MAP["phi3_v"] = Phi3VGPTQForCausalLM   
