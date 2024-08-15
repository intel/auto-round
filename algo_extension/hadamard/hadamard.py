import gc

import torch
import transformers
## adapted from https://github.com/spcl/QuaRot
from .utils import random_hadamard_matrix
from auto_round.utils import get_module


def get_embeddings(model) -> list[torch.nn.Module]:
    if isinstance(model, transformers.models.llama.LlamaForCausalLM):
        return [model.model.embed_tokens]
    # elif model_type == OPT_MODEL:
    #     return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    else:
        raise ValueError(f'Unknown model type')


def get_lm_head(model) -> list[torch.nn.Module]:
    if isinstance(model, transformers.models.llama.LlamaForCausalLM):
        name = "lm_head"
    # elif model_type == OPT_MODEL:
    #     return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    else:
        raise ValueError(f'Unknown model type')
    return get_module(model, name)


def rotate_embeddings(model, Q: torch.Tensor, device="cpu") -> None:
    # Rotate the embeddings.
    for layer in get_embeddings(model):
        dtype = layer.weight.data.dtype
        new_weight = layer.weight.data.to(device=device, dtype=torch.float64)
        layer.weight.data = torch.matmul(new_weight, Q).to(device="cpu", dtype=dtype)


def rotate_head(model, Q: torch.Tensor, device="cpu") -> None:
    # Rotate the head.
    lm_head_layer = get_lm_head(model)
    dtype = lm_head_layer.weight.data.dtype
    W_ = lm_head_layer.weight.data.to(device=device, dtype=torch.float64)
    lm_head_layer.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cpu"):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')


def rotate_attention_inputs(model, layer, Q, device="cpu") -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)



def rotate_attention_output(model,layer, Q, device="cpu") -> None:
    # Rotate output matrix of the self-attention layer.
    if isinstance(model, transformers.models.llama.LlamaForCausalLM):
        W = layer.self_attn.o_proj
    # elif model_type == model_utils.OPT_MODEL:
    #     W = layer.self_attn.out_proj
    # else:
    #     raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_input(model, layer, Q, device="cpu") -> None:
    # Rotate the MLP input weights.
    ##if model_type == model_utils.LLAMA_MODEL:
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    # elif model_type == model_utils.OPT_MODEL:
    #     mlp_inputs = [layer.fc1]
    # else:
    #     raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_output(model, layer, Q, device="cpu"):
    # Rotate the MLP output weights and bias.
    # if model_type == model_utils.LLAMA_MODEL:
    W = layer.mlp.down_proj
    # elif model_type == model_utils.OPT_MODEL:
    #     W = layer.fc2
    # else:
    #     raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    # W.weight.data = torch.matmul(Q.T, W_)
    # W.weight.data = torch.matmul(Q, W.weight.data ).to(device="cpu", dtype=dtype)
    # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output TODO, not add online
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_model(model, rotate_mode="hadamard", device="cpu") -> None:

    from algo_extension.utils import fuse_norm
    fuse_norm(model)##must fuse,including lm-head

    Q = get_orthogonal_matrix(model.config.hidden_size,
                              rotate_mode)
    Q = Q.to(device)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, Q,device)
    rotate_head(model, Q,device)
    gc.collect()
    torch.cuda.empty_cache()
    from auto_round.utils import get_block_names
    block_names = get_block_names(model)[0]
    for block_name in block_names:
        block = get_module(model, block_name)
        rotate_attention_inputs(model,block,Q,device)
        rotate_attention_output(model,block,Q,device)
        rotate_mlp_input(model, block, Q, device)
        rotate_mlp_output(model, block, Q, device)
    #
    # block_name = block_names[0]
    # block = get_module(model, block_name)
    # rotate_mlp_output(model, block, Q, device)
    # block_name = block_names[1]
    # block = get_module(model, block_name)
    # rotate_attention_inputs(model, block, Q, device)
    #rotate_head(model, Q, device)

    # layers = model_utils.get_transformer_layers(model,
    #                                             model_type=model_type)
    # for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
    #     rotate_attention_inputs(layers[idx], Q, model_type)
    #     rotate_attention_output(layers[idx], Q, model_type)
    #     rotate_mlp_input(layers[idx], Q, model_type)
    #     rotate_mlp_output(layers[idx], Q, model_type)
    #     rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)





