# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import gc
import typing

import torch
import tqdm

from auto_round.experimental.hadamard_inplace.hadamard import apply_exact_had_to_linear, random_hadamard_matrix
from auto_round.experimental.hadamard_inplace.utils import register_online_had_hooks


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


def _reset_ln_params(layernorm: torch.nn.Module) -> None:
    """Reset LayerNorm to identity: weight=1, bias=0.

    Must be called after fuse_ln_linear so that the fused LN no longer
    scales/shifts (the parameters are already absorbed into the Linear).
    """
    layernorm.weight.data.fill_(1.0)
    if hasattr(layernorm, "bias") and layernorm.bias is not None:
        layernorm.bias.data.fill_(0.0)


def fuse_layer_norms(model):
    # Embedding mean subtraction — only valid for standard LayerNorm (which
    # subtracts mean internally, so the shift cancels out).  LLaMA uses
    # RMSNorm which does NOT subtract mean, so this step is skipped.
    # W = model.model.embed_tokens
    # W_ = W.weight.data.double()
    # W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = model.model.layers

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
        _reset_ln_params(layer.post_attention_layernorm)

        fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        _reset_ln_params(layer.input_layernorm)

    fuse_ln_linear(model.model.norm, [model.lm_head])
    _reset_ln_params(model.model.norm)


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
    ##apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)


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
        # rotate_ov_proj(layers[idx], num_heads, head_dim)




def allpy_model(model, fp32_had=False):
    """Fuse layer norms, rotate weights, and register online Hadamard hooks.

    Args:
        model: A HuggingFace model (e.g. LLaMA-2).
        fp32_had: Whether to compute the online Hadamard transform in fp32.

    Returns:
        list of hook handles (call ``handle.remove()`` to detach).
    """
    fuse_layer_norms(model)
    rotate_model(model)
    handles = register_online_had_hooks(model, fp32_had=fp32_had)
    return handles

from transformers import AutoTokenizer,AutoModelForCausalLM

if __name__ == "__main__":
    model_name = "/models/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    allpy_model(model)
    model.to("cuda")
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
