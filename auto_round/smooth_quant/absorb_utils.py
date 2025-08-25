#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Intel Corporation
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
import torch

from auto_round.smooth_quant.utils import get_module

SUPPORTED_TORCH_MODULE = [
    "Linear",
    "Conv2d",
    "ConvTranspose2d",
    "LayerNorm",
    "BatchNorm2d",
    "GroupNorm",
    "InstanceNorm2d",
    "LlamaRMSNorm",
    "T5LayerNorm",
    "LPLayerNorm",
    "RMSNorm",
    "Qwen2RMSNorm",
    "WrapperWALayer",
]

GET_ABSORB_LAYERS = {}


def register_absorb_func(model_type):
    def register(func):
        if isinstance(model_type, list):
            model_types = model_type
        else:
            model_types = [model_type]
        for name in model_types:
            GET_ABSORB_LAYERS[name] = func
        return func

    return register


def _check_valid_conv(module):
    """Remove group conv except depthwise conv
    :param module:

    :return:
    """
    if not isinstance(module, torch.nn.Conv2d):
        return True
    if module.groups > 1:
        if module.in_channels == module.out_channels and module.groups == module.in_channels:
            return True
        else:
            return False
    return True


def remove_unsupported_layers(model, absorb_to_layer, no_absorb_layers):
    res = {}
    for key in absorb_to_layer.keys():
        absorb_layer = get_module(model, key)
        layer_type = absorb_layer.__class__.__name__
        if layer_type not in SUPPORTED_TORCH_MODULE:
            no_absorb_layers.extend(absorb_to_layer[key])
            continue
        supported = True
        for layer_name in absorb_to_layer[key]:
            layer = get_module(model, layer_name)
            layer_type = layer.__class__.__name__
            if (layer_type not in SUPPORTED_TORCH_MODULE) or not _check_valid_conv(layer):
                supported = False
                no_absorb_layers.extend(absorb_to_layer[key])
                break
        if supported:
            res[key] = absorb_to_layer[key]
    return res


@register_absorb_func("opt")
def get_opt_absorb_layers(model):
    model_layer_name = "model.decoder.layers"
    absorb_to_layer = {}
    for idx in range(len(model.model.decoder.layers)):
        # attention input
        absorb_to_layer[f"{model_layer_name}.{idx}.self_attn_layer_norm"] = [
            f"{model_layer_name}.{idx}.self_attn.q_proj",
            f"{model_layer_name}.{idx}.self_attn.k_proj",
            f"{model_layer_name}.{idx}.self_attn.v_proj",
        ]

        # attention out
        # no_absorb_layers.append(f"{model_layer_name}.{idx}.self_attn.out_proj")
        absorb_to_layer[f"{model_layer_name}.{idx}.v_proj"] = [
            f"{model_layer_name}.{idx}.self_attn.out_proj",
        ]

        # linear 1
        absorb_to_layer[f"{model_layer_name}.{idx}.final_layer_norm"] = [
            f"{model_layer_name}.{idx}.fc1",
        ]

        # linear 2
        absorb_to_layer[f"{model_layer_name}.{idx}.fc1"] = [
            f"{model_layer_name}.{idx}.fc2",
        ]

    # final layer
    # absorb_to_layer["model.decoder.final_layer_norm"] = ['lm_head']

    return absorb_to_layer


# @register_absorb_func('llama')
# def get_llama_absorb_layers(model):
#     model_layer_name = "model.layers"
#     absorb_to_layer = {}

#     for idx in range(len(model.model.layers)):
#         # attention input
#         absorb_to_layer[f"{model_layer_name}.{idx}.input_layernorm"] = [
#             f"{model_layer_name}.{idx}.self_attn.q_proj",
#             f"{model_layer_name}.{idx}.self_attn.k_proj",
#             f"{model_layer_name}.{idx}.self_attn.v_proj",
#         ]

#         # attention out
#         module = model.model.layers[idx]
#         if hasattr(module.self_attn.v_proj, "orig_layer"):
#             v_proj_shape = module.self_attn.v_proj.orig_layer.weight.shape
#             o_proj_shape = module.self_attn.o_proj.orig_layer.weight.shape
#         else:
#             v_proj_shape = module.self_attn.v_proj.weight.shape
#             o_proj_shape = module.self_attn.o_proj.weight.shape
#         if v_proj_shape == o_proj_shape:
#             absorb_to_layer[f"{model_layer_name}.{idx}.v_proj"] = [
#                 f"{model_layer_name}.{idx}.self_attn.o_proj",
#             ]

#         # linear 1
#         absorb_to_layer[f"{model_layer_name}.{idx}.post_attention_layernorm"] = [
#             f"{model_layer_name}.{idx}.mlp.gate_proj",
#             f"{model_layer_name}.{idx}.mlp.up_proj",
#         ]

#         # linear 2
#         absorb_to_layer[f"{model_layer_name}.{idx}.mlp.up_proj"] = [
#             f"{model_layer_name}.{idx}.mlp.down_proj",
#         ]

#     # final layer
#     # absorb_to_layer["model.norm"] = ['lm_head']

#     return absorb_to_layer


@register_absorb_func("mistral")
def get_mistral_absorb_layers(model):
    model_layer_name = "model.layers"
    absorb_to_layer = {}
    for idx in range(len(model.model.layers)):
        # attention input
        absorb_to_layer[f"{model_layer_name}.{idx}.input_layernorm"] = [
            f"{model_layer_name}.{idx}.self_attn.q_proj",
            f"{model_layer_name}.{idx}.self_attn.k_proj",
            f"{model_layer_name}.{idx}.self_attn.v_proj",
        ]

        # attention out
        module = model.model.layers[idx]
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            absorb_to_layer[f"{model_layer_name}.{idx}.v_proj"] = [
                f"{model_layer_name}.{idx}.self_attn.o_proj",
            ]

        # linear 1
        absorb_to_layer[f"{model_layer_name}.{idx}.post_attention_layernorm"] = [
            f"{model_layer_name}.{idx}.mlp.gate_proj",
            f"{model_layer_name}.{idx}.mlp.up_proj",
        ]

        # linear 2
        absorb_to_layer[f"{model_layer_name}.{idx}.mlp.up_proj"] = [
            f"{model_layer_name}.{idx}.mlp.down_proj",
        ]

    # final layer
    # absorb_to_layer["model.norm"] = ['lm_head']

    return absorb_to_layer


@register_absorb_func("mixtral")
def get_mixtral_absorb_layers(model):
    model_layer_name = "model.layers"
    absorb_to_layer = {}
    for idx in range(len(model.model.layers)):
        # attention input
        absorb_to_layer[f"{model_layer_name}.{idx}.input_layernorm"] = [
            f"{model_layer_name}.{idx}.self_attn.q_proj",
            f"{model_layer_name}.{idx}.self_attn.k_proj",
            f"{model_layer_name}.{idx}.self_attn.v_proj",
        ]

        # attention out
        module = model.model.layers[idx]
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            absorb_to_layer[f"{model_layer_name}.{idx}.v_proj"] = [
                f"{model_layer_name}.{idx}.self_attn.o_proj",
            ]

        # linear in
        module = get_module(model, f"{model_layer_name}.{idx}.block_sparse_moe.experts")
        absorb_to_layer[f"{model_layer_name}.{idx}.post_attention_layernorm"] = []
        for i in range(len(module)):
            absorb_to_layer[f"{model_layer_name}.{idx}.post_attention_layernorm"].extend(
                [
                    f"{model_layer_name}.{idx}.block_sparse_moe.experts.{i}.w1",
                    f"{model_layer_name}.{idx}.block_sparse_moe.experts.{i}.w3",
                ]
            )

        # linear out
        for i in range(len(module)):
            absorb_to_layer[f"{model_layer_name}.{idx}.block_sparse_moe.experts.{i}.w3"] = [
                f"{model_layer_name}.{idx}.block_sparse_moe.experts.{i}.w2"
            ]

    # final layer
    # absorb_to_layer["model.norm"] = ['lm_head']
    return absorb_to_layer


@register_absorb_func("bloom")
def get_bloom_absorb_layers(model):
    model_layer_name = "transformer.h"
    absorb_to_layer = {}
    for idx in range(len(model.transformer.h)):
        # attention input
        absorb_to_layer[f"{model_layer_name}.{idx}.input_layernorm"] = [
            f"{model_layer_name}.{idx}.self_attention.query_key_value",
        ]

        # linear 1
        absorb_to_layer[f"{model_layer_name}.{idx}.post_attention_layernorm"] = [
            f"{model_layer_name}.{idx}.mlp.dense_h_to_4h",
        ]

        # linear 2
        absorb_to_layer[f"{model_layer_name}.{idx}.mlp.gelu_impl"] = [
            f"{model_layer_name}.{idx}.mlp.dense_4h_to_h",
        ]

    # final layer
    # absorb_to_layer["transformer.ln_f"] = ['lm_head']

    return absorb_to_layer


@register_absorb_func("gptj")
def get_gptj_absorb_layers(model):
    model_layer_name = "transformer.h"
    absorb_to_layer = {}
    for idx in range(len(model.transformer.h)):
        # attention input + linear 1
        absorb_to_layer[f"{model_layer_name}.{idx}.ln_1"] = [
            f"{model_layer_name}.{idx}.attn.q_proj",
            f"{model_layer_name}.{idx}.attn.k_proj",
            f"{model_layer_name}.{idx}.attn.v_proj",
            f"{model_layer_name}.{idx}.mlp.fc_in",
        ]

        # attention out
        absorb_to_layer[f"{model_layer_name}.{idx}.attn.v_proj"] = [
            f"{model_layer_name}.{idx}.attn.out_proj",
        ]

        # linear 2
        absorb_to_layer[f"{model_layer_name}.{idx}.mlp.act"] = [
            f"{model_layer_name}.{idx}.mlp.fc_out",
        ]

    # final layer
    # absorb_to_layer["transformer.ln_f"] = ['lm_head']

    return absorb_to_layer


@register_absorb_func("phi3")
def get_phi3_absorb_layers(model):
    model_layer_name = "model.layers"
    absorb_to_layer = {}
    for idx in range(len(model.model.layers)):
        # attention input
        absorb_to_layer[f"{model_layer_name}.{idx}.input_layernorm"] = [
            f"{model_layer_name}.{idx}.self_attn.qkv_proj",
        ]

        # attention out
        absorb_to_layer[f"{model_layer_name}.{idx}.self_attn.qkv_proj"] = [
            f"{model_layer_name}.{idx}.self_attn.o_proj",
        ]

        # linear 1
        absorb_to_layer[f"{model_layer_name}.{idx}.post_attention_layernorm"] = [
            f"{model_layer_name}.{idx}.mlp.gate_up_proj",
        ]

        # linear 2
        absorb_to_layer[f"{model_layer_name}.{idx}.mlp.gate_up_proj"] = [
            f"{model_layer_name}.{idx}.mlp.down_proj",
        ]

    # final layer
    # absorb_to_layer["model.norm"] = ['lm_head']

    return absorb_to_layer


@register_absorb_func("qwen")
def get_qwen_absorb_layers(model):
    model_layer_name = "transformer.h"
    absorb_to_layer = {}
    for idx in range(len(model.transformer.h)):
        # attention
        absorb_to_layer[f"{model_layer_name}.{idx}.ln_1"] = [f"{model_layer_name}.{idx}.attn.c_attn"]

        # mlp
        absorb_to_layer[f"{model_layer_name}.{idx}.ln_2"] = [
            f"{model_layer_name}.{idx}.mlp.w2",
            f"{model_layer_name}.{idx}.mlp.w1",
        ]

        # linear 2
        absorb_to_layer[f"{model_layer_name}.{idx}.mlp.w1"] = [
            f"{model_layer_name}.{idx}.mlp.c_proj",
        ]

    # final layer
    # absorb_to_layer["transformer.ln_f"] = ['lm_head']

    return absorb_to_layer


@register_absorb_func(["qwen2", "qwen3"])
@register_absorb_func("llama")
def get_defualt_absorb_layers(model):
    model_layer_name = "model.layers"
    absorb_to_layer = {}

    for idx in range(len(model.model.layers)):
        # attention input
        absorb_to_layer[f"{model_layer_name}.{idx}.input_layernorm"] = [
            f"{model_layer_name}.{idx}.self_attn.q_proj",
            f"{model_layer_name}.{idx}.self_attn.k_proj",
            f"{model_layer_name}.{idx}.self_attn.v_proj",
        ]

        # attention out
        module = model.model.layers[idx]
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            absorb_to_layer[f"{model_layer_name}.{idx}.v_proj"] = [
                f"{model_layer_name}.{idx}.self_attn.o_proj",
            ]

        # linear 1
        absorb_to_layer[f"{model_layer_name}.{idx}.post_attention_layernorm"] = [
            f"{model_layer_name}.{idx}.mlp.gate_proj",
            f"{model_layer_name}.{idx}.mlp.up_proj",
        ]

        # linear 2
        absorb_to_layer[f"{model_layer_name}.{idx}.mlp.up_proj"] = [
            f"{model_layer_name}.{idx}.mlp.down_proj",
        ]

    # final layer
    # absorb_to_layer["model.norm"] = ['lm_head']

    return absorb_to_layer


@register_absorb_func("qwen3_moe")
def get_qwen3_moe_absorb_layers(model):
    model_layer_name = "model.layers"
    absorb_to_layer = {}
    for idx in range(len(model.model.layers)):
        # attention input
        absorb_to_layer[f"{model_layer_name}.{idx}.input_layernorm"] = [
            f"{model_layer_name}.{idx}.self_attn.q_proj",
            f"{model_layer_name}.{idx}.self_attn.k_proj",
            f"{model_layer_name}.{idx}.self_attn.v_proj",
        ]

        # attention out
        module = model.model.layers[idx]
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            absorb_to_layer[f"{model_layer_name}.{idx}.v_proj"] = [
                f"{model_layer_name}.{idx}.self_attn.o_proj",
            ]

        if hasattr(module.mlp, "gate"):
            # linear in
            absorb_to_layer[f"{model_layer_name}.{idx}.post_attention_layernorm"] = [
                f"{model_layer_name}.{idx}.mlp.experts.{i}.gate_proj" for i in range(len(module.mlp.experts))
            ]
            absorb_to_layer[f"{model_layer_name}.{idx}.post_attention_layernorm"].extend(
                [f"{model_layer_name}.{idx}.mlp.experts.{i}.up_proj" for i in range(len(module.mlp.experts))]
            )
            breakpoint()

            # linear out
            for i in range(len(module.mlp.experts)):
                absorb_to_layer[f"{model_layer_name}.{idx}.mlp.experts.{i}.up_proj"] = [
                    f"{model_layer_name}.{idx}.mlp.experts.{i}.down_proj",
                ]
        else:
            # linear 1
            absorb_to_layer[f"{model_layer_name}.{idx}.post_attention_layernorm"] = [
                f"{model_layer_name}.{idx}.mlp.gate_proj",
                f"{model_layer_name}.{idx}.mlp.up_proj",
            ]

            # linear 2
            absorb_to_layer[f"{model_layer_name}.{idx}.mlp.up_proj"] = [f"{model_layer_name}.{idx}.mlp.down_proj"]

    # final layer
    # absorb_to_layer["model.norm"] = ['lm_head']
    return absorb_to_layer


def get_absorb_layers(model, skip_unsupported_layers=False):
    model_type = model.config.model_type
    assert model_type in GET_ABSORB_LAYERS, f"Unsupported model type: {model_type}"
    absorb_to_layer = GET_ABSORB_LAYERS[model_type](model)
    no_absorb_layers = []
    # if skip_unsupported_layers:
    #     absorb_to_layer = remove_unsupported_layers(model, absorb_to_layer, no_absorb_layers)
    return absorb_to_layer, no_absorb_layers
