# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Hadamard inplace rotation — public API and rotation primitives.

Supports LLaMA-2, LLaMA-3, Qwen-3 (and any model with the same layout).
The entry point is :func:`apply_hadamard_rotation`.
"""

import gc
import typing

import torch
import tqdm

from auto_round.experimental.hadamard_inplace.model_config import (
    RotationMapping,
    _resolve,
    infer_mapping_from_model,
)
from auto_round.experimental.hadamard_inplace.utils import (
    apply_exact_had_to_linear,
    random_hadamard_matrix,
    register_online_had_hooks,
)

# ---------------------------------------------------------------------------
# Low-level primitives (model-agnostic via RotationMapping)
# ---------------------------------------------------------------------------


def _fuse_ln_linear(
    layernorm: torch.nn.Module,
    linear_layers: typing.Iterable[torch.nn.Linear],
) -> None:
    """Fuse the linear operations in LayerNorm into adjacent linear blocks."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        dev = linear.weight.device

        W_ = linear.weight.data.double()
        ln_weight = layernorm.weight.double().to(dev)
        linear.weight.data = (W_ * ln_weight).to(linear_dtype)

        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64, device=dev))
            ln_bias = layernorm.bias.double().to(dev)
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, ln_bias)
            linear.bias.data = linear.bias.data.to(linear_dtype)


def _reset_ln_params(layernorm: torch.nn.Module) -> None:
    """Reset LayerNorm to identity: weight=1, bias=0."""
    layernorm.weight.data.fill_(1.0)
    if hasattr(layernorm, "bias") and layernorm.bias is not None:
        layernorm.bias.data.fill_(0.0)


def _rotate_linear_by_Q(module: torch.nn.Linear, Q: torch.Tensor, side: str) -> None:
    """Apply rotation *Q* to a Linear layer's weight (and bias if present).

    Args:
        side: ``'input'``  →  W = W @ Q   (rotate input side)
              ``'output'`` →  W = Q^T @ W  (rotate output side)
    """
    dtype = module.weight.data.dtype
    dev = module.weight.data.device
    W_ = module.weight.data.to(dtype=torch.float64)
    Q_ = Q.to(device=dev)
    if side == "input":
        module.weight.data = torch.matmul(W_, Q_).to(dtype=dtype)
    else:
        module.weight.data = torch.matmul(Q_.T, W_).to(dtype=dtype)
        if module.bias is not None:
            b = module.bias.data.to(dtype=torch.float64)
            module.bias.data = torch.matmul(Q_.T, b).to(dtype=dtype)


def _untie_word_embeddings(model, mapping: RotationMapping) -> None:
    """Break tied weights between lm_head and embedding if they share the same tensor.

    Auto-detects by comparing data pointers. After rotation, embedding and
    lm_head receive different transforms, so they must have independent storage.
    """
    embedding = _resolve(model, mapping.embedding)
    lm_head = _resolve(model, mapping.lm_head)

    if lm_head.weight.data_ptr() != embedding.weight.data_ptr():
        return  # not tied, nothing to do

    lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
    # Tell HuggingFace config so it won't re-tie later
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False


def _uses_layernorm_with_mean(model, mapping: RotationMapping) -> bool:
    """Check whether the model uses standard LayerNorm (which subtracts mean).

    Returns True for ``nn.LayerNorm``, False for RMSNorm variants.
    Inspects the first layer's attention input LN as a representative.
    """
    layers = _resolve(model, mapping.layers_attr)
    first_ln = _resolve(layers[0], mapping.attn_input_ln)
    return isinstance(first_ln, torch.nn.LayerNorm)


def _bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """Subtract column-wise mean from a Linear layer's weight (and mean from bias).

    After this, the linear layer's output is always zero-mean, which absorbs
    the mean-subtraction step of standard LayerNorm.  This allows LayerNorm
    to be treated as RMSNorm (normalize only, no mean subtraction) and thus
    be fused and rotated just like in RMSNorm models.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = (W_ - W_.mean(dim=-2, keepdim=True)).to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = (b_ - b_.mean()).to(linear_dtype)


def _subtract_embedding_mean(model, mapping: RotationMapping) -> None:
    """Subtract per-row mean from the embedding weight matrix.

    Standard LayerNorm subtracts the mean internally, so the mean component
    of the embedding is cancelled anyway.  By removing it upfront we ensure
    the rotation does not distort the zero-mean assumption.
    """
    W = _resolve(model, mapping.embedding)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(dtype=dtype)

    # Also subtract mean from positional embedding (e.g. OPT's learned pos embed)
    if mapping.positional_embedding is not None:
        P = _resolve(model, mapping.positional_embedding)
        p_dtype = P.weight.data.dtype
        P_ = P.weight.data.to(dtype=torch.float64)
        P.weight.data = (P_ - P_.mean(dim=-1, keepdim=True)).to(dtype=p_dtype)


class _RMSNorm(torch.nn.Module):
    """RMS Normalization (no mean subtraction).

    Replaces ``nn.LayerNorm`` after ``bake_mean_into_linear`` has absorbed the
    mean-subtraction into the linear layers.  This makes the normalization
    commutative with orthogonal rotations.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def _replace_layernorms_with_rmsnorm(model) -> None:
    """Replace all ``nn.LayerNorm`` modules with ``_RMSNorm``.

    After ``bake_mean_into_linear`` has absorbed the mean-subtraction into
    linear layers, the LayerNorm's mean subtraction is redundant and breaks
    rotation commutativity.  Replacing with RMS-only norm fixes this.
    """
    # Collect first to avoid modifying the model while iterating
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            replacements.append((name, module))

    for name, module in replacements:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = _resolve(model, parts[0])
            attr = parts[1]
        else:
            parent = model
            attr = parts[0]
        rms = _RMSNorm(module.normalized_shape[0], eps=module.eps)
        rms = rms.to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(parent, attr, rms)


# ---------------------------------------------------------------------------
# High-level steps driven by RotationMapping
# ---------------------------------------------------------------------------


def _fuse_layer_norms(model, mapping: RotationMapping) -> None:
    """Fuse all LayerNorm parameters into adjacent Linear layers."""
    layers = _resolve(model, mapping.layers_attr)

    for layer in layers:
        # MLP input LN → gate / up projections
        mlp_ln = _resolve(layer, mapping.mlp_input_ln)
        mlp_linears = [_resolve(layer, p) for p in mapping.mlp_in]
        _fuse_ln_linear(mlp_ln, mlp_linears)
        _reset_ln_params(mlp_ln)

        # Attention input LN → Q / K / V projections
        attn_ln = _resolve(layer, mapping.attn_input_ln)
        attn_linears = [
            _resolve(layer, mapping.attn_q),
            _resolve(layer, mapping.attn_k),
            _resolve(layer, mapping.attn_v),
        ]
        _fuse_ln_linear(attn_ln, attn_linears)
        _reset_ln_params(attn_ln)

    # Pre-head norm → lm_head
    pre_head_ln = _resolve(model, mapping.pre_head_ln)
    lm_head = _resolve(model, mapping.lm_head)
    _fuse_ln_linear(pre_head_ln, [lm_head])
    _reset_ln_params(pre_head_ln)


@torch.inference_mode()
def _rotate_weights(model, mapping: RotationMapping, use_fast_had: bool = True) -> None:
    """Apply random Hadamard + exact Hadamard rotations to all weights."""
    config = model.config
    hidden_size = getattr(config, mapping.hidden_size_attr)
    num_heads = getattr(config, mapping.num_heads_attr)
    head_dim = mapping.attn_head_dim or (hidden_size // num_heads)

    Q = random_hadamard_matrix(hidden_size, model.device)

    # Top-level: embedding & lm_head
    embedding = _resolve(model, mapping.embedding)
    dtype = embedding.weight.data.dtype
    dev = embedding.weight.data.device
    W_ = embedding.weight.data.to(dtype=torch.float64)
    embedding.weight.data = torch.matmul(W_, Q.to(dev)).to(dtype=dtype)

    # Positional embedding (e.g. OPT's learned pos embed)
    if mapping.positional_embedding is not None:
        pos_emb = _resolve(model, mapping.positional_embedding)
        pos_dtype = pos_emb.weight.data.dtype
        pos_dev = pos_emb.weight.data.device
        P_ = pos_emb.weight.data.to(dtype=torch.float64)
        pos_emb.weight.data = torch.matmul(P_, Q.to(pos_dev)).to(dtype=pos_dtype)

    lm_head = _resolve(model, mapping.lm_head)
    _rotate_linear_by_Q(lm_head, Q, side="input")

    gc.collect()
    torch.cuda.empty_cache()

    layers = _resolve(model, mapping.layers_attr)
    for layer in tqdm.tqdm(layers, unit="layer", desc="Rotating"):
        # Attention inputs: Q/K/V  ← W @ Q
        for attr in (mapping.attn_q, mapping.attn_k, mapping.attn_v):
            _rotate_linear_by_Q(_resolve(layer, attr), Q, side="input")

        # Attention output: o_proj ← Q^T @ W
        _rotate_linear_by_Q(_resolve(layer, mapping.attn_o), Q, side="output")

        # MLP inputs: gate/up ← W @ Q
        for attr in mapping.mlp_in:
            _rotate_linear_by_Q(_resolve(layer, attr), Q, side="input")

        # MLP output: down_proj ← Q^T @ W
        _rotate_linear_by_Q(_resolve(layer, mapping.mlp_out), Q, side="output")

        # Exact Hadamard on down_proj input
        down_proj = _resolve(layer, mapping.mlp_out)
        apply_exact_had_to_linear(down_proj, had_dim=-1, output=False, use_fast_had=use_fast_had)

        # OV projection: within-head Had on v_proj + full Had on o_proj
        v_proj = _resolve(layer, mapping.attn_v)
        o_proj = _resolve(layer, mapping.attn_o)
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, use_fast_had=use_fast_had)
        apply_exact_had_to_linear(o_proj, had_dim=-1, output=False, use_fast_had=use_fast_had)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_hadamard_rotation(model, fp32_had: bool = False, use_fast_had: bool = True):
    """Fuse layer norms, rotate weights, and register online Hadamard hooks.

    This is the single entry point for applying Hadamard inplace rotation.
    The model architecture is auto-detected via ``model.config.model_type``.

    Args:
        model: A HuggingFace CausalLM model (LLaMA-2/3, Qwen-3, etc.).
        fp32_had: Whether to compute the online Hadamard transform in fp32.
        use_fast_had: If ``True`` use ``fast_hadamard_transform`` library;
            if ``False`` use pure-Python butterfly ``matmul_hadU``.

    Returns:
        list of hook handles (call ``handle.remove()`` to detach).
    """
    if use_fast_had:
        from auto_round.utils import logger

        try:
            import fast_hadamard_transform  # noqa: F401

            logger.warning(
                "fast_hadamard_transform uses a different Hadamard matrix than the "
                "default implementation. Please ensure consistency between training "
                "and inference. This will be refined later."
            )
        except ImportError:
            logger.warning("Importing fast_hadamard_transform failed, " "falling back to default implementation.")
            use_fast_had = False

    mapping = infer_mapping_from_model(model)

    # Untie lm_head ↔ embedding if they share the same weight tensor.
    _untie_word_embeddings(model, mapping)

    # Subtract per-row mean from embedding.  For standard LayerNorm (OPT)
    # this is required; for RMSNorm (LLaMA/Qwen) it is harmless since
    # RMSNorm doesn't subtract mean.  QuaRot does this unconditionally.
    _subtract_embedding_mean(model, mapping)

    # Fuse LayerNorm affine params into adjacent Linear layers
    _fuse_layer_norms(model, mapping)

    # For standard LayerNorm models (e.g. OPT), absorb the mean-subtraction
    # into the linear layers that feed back into the residual stream (out_proj
    # and down_proj/fc2).  Must be done AFTER fuse_ln so that bake operates
    # on the final (fused) weights.  After this, LayerNorm effectively becomes
    # RMSNorm and the rotation is commutative.
    if _uses_layernorm_with_mean(model, mapping):
        layers = _resolve(model, mapping.layers_attr)
        for layer in layers:
            _bake_mean_into_linear(_resolve(layer, mapping.attn_o))
            _bake_mean_into_linear(_resolve(layer, mapping.mlp_out))
        # Replace all nn.LayerNorm with RMS-only norm — after bake_mean has
        # absorbed the mean subtraction, LayerNorm's mean removal is redundant
        # and breaks rotation commutativity.
        _replace_layernorms_with_rmsnorm(model)

    _rotate_weights(model, mapping, use_fast_had=use_fast_had)

    # For v_proj it's within-head; combining with cross-head equals a full Hadamard
    handles = register_online_had_hooks(
        model,
        mapping=mapping,
        fp32_had=fp32_had,
        use_fast_had=use_fast_had,
    )
    return handles


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import auto_round
    from auto_round.experimental.transform.utils.hadamard import _fetch_hadamard_divisor

    K_list = [172, 156, 140, 108, 60, 52, 36, 28, 40, 20, 12]
    for d in K_list:
        fn = getattr(auto_round.experimental.hadamard_inplace.hadamard_matrix, f"get_had{d}")
        quarot_had = fn()
        hadK = _fetch_hadamard_divisor(d, torch.float, torch.device("cpu"))
        torch.equal(quarot_had, hadK)
    print("equal test passed")
    model_name = "/models/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.to("cuda")
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    apply_hadamard_rotation(model, use_fast_had=False)
    model.to("cuda")
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    model_name = "/models/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    apply_hadamard_rotation(model, use_fast_had=False)
    model.to("cuda")
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "/models/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    apply_hadamard_rotation(model, use_fast_had=False)
    model.to("cuda")
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    model_name = "/models/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    apply_hadamard_rotation(model, use_fast_had=False)
    model.to("cuda")
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
