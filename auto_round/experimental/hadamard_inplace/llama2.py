# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import gc
import math
import typing

import torch
import tqdm
import transformers

from auto_round.experimental.hadamard_inplace.hadamard import apply_exact_had_to_linear, random_hadamard_matrix


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model):
    # Embedding fusion
    W = model.model.embed_tokens
    W_ = W.weight.data.double()
    W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = model.model.layers

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
        fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])

    fuse_ln_linear(model.model.norm, [model.lm_head])


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    W = model.model.embed_tokens
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, Q) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, Q) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, Q):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    apply_exact_had_to_linear(
        W, had_dim=-1, output=False
    )  # apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)


@torch.inference_mode()
def rotate_model(model):
    Q = random_hadamard_matrix(model.config.hidden_size, model.device)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    gc.collect()
    torch.cuda.empty_cache()
    layers = model.model.layers
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q)
        rotate_attention_output(layers[idx], Q)
        rotate_mlp_input(layers[idx], Q)
        rotate_mlp_output(layers[idx], Q)
        rotate_ov_proj(layers[idx], num_heads, head_dim)


# Rotate the weights
if args.rotate:
    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model, args)
    utils.cleanup_memory(verbos=True)

    quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if "down_proj" in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = args.fp32_had
        if "o_proj" in name:
            had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
            qlayers[name].online_partial_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads
            qlayers[name].fp32_had = args.fp32_had
else:
    quant_utils.add_actquant(model)  # Add Activation Wrapper to the model as the rest of the code assumes it is present
