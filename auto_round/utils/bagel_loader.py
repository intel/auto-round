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

"""Loader for BAGEL-7B-MoT (ByteDance-Seed/BAGEL-7B-MoT) model.

BAGEL uses a Qwen2-based LLM with not (Mixture of Transformers) extensions.
Since transformers doesn't natively support the 'bagel' model_type, we construct
the model manually by:
  1. Building a standard Qwen2ForCausalLM from the llm_config
  2. Adding not generation-path modules (mlp_moe_gen, *_moe_gen projections)
  3. Loading all weights from safetensors
  4. Wrapping in BagelForQuantization for auto_round compatibility
"""

import glob
import json
import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PretrainedConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2MLP, Qwen2RMSNorm

from auto_round.logger import logger


class BagelConfig(PretrainedConfig):
    """Configuration for the BAGEL model wrapper."""

    model_type = "bagel"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def _add_mot_extensions(language_model, llm_config):
    """Add not (Mixture of Transformers) generation-path modules to a Qwen2 model.

    Each transformer layer gets additional modules for the generation path:
      - Attention: q_proj_moe_gen, k_proj_moe_gen, v_proj_moe_gen, o_proj_moe_gen
      - Attention norms: q_norm_moe_gen, k_norm_moe_gen (when qk_norm is used)
      - MLP: mlp_moe_gen (full MLP duplicate)
      - LayerNorms: input_layernorm_moe_gen, post_attention_layernorm_moe_gen
    """
    hidden_size = llm_config.hidden_size
    num_heads = llm_config.num_attention_heads
    num_kv_heads = getattr(llm_config, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    rms_norm_eps = llm_config.rms_norm_eps
    use_qk_norm = getattr(llm_config, "qk_norm", False)

    for layer in language_model.model.layers:
        attn = layer.self_attn

        # QK norms for the understanding path (not present in standard Qwen2Attention)
        if use_qk_norm and not hasattr(attn, "q_norm"):
            attn.q_norm = Qwen2RMSNorm(head_dim, eps=rms_norm_eps)
            attn.k_norm = Qwen2RMSNorm(head_dim, eps=rms_norm_eps)

        # not attention projections for generation path
        attn.q_proj_moe_gen = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        attn.k_proj_moe_gen = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        attn.v_proj_moe_gen = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        attn.o_proj_moe_gen = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # not QK norms for generation path
        if use_qk_norm:
            attn.q_norm_moe_gen = Qwen2RMSNorm(head_dim, eps=rms_norm_eps)
            attn.k_norm_moe_gen = Qwen2RMSNorm(head_dim, eps=rms_norm_eps)
        else:
            attn.q_norm_moe_gen = nn.Identity()
            attn.k_norm_moe_gen = nn.Identity()

        # not MLP for generation path (duplicate of understanding MLP)
        layer.mlp_moe_gen = Qwen2MLP(llm_config)

        # not LayerNorms for generation path
        layer.input_layernorm_moe_gen = Qwen2RMSNorm(hidden_size, eps=rms_norm_eps)
        layer.post_attention_layernorm_moe_gen = Qwen2RMSNorm(hidden_size, eps=rms_norm_eps)


def _build_module_from_weights(weight_dict):
    """Build an nn.Module hierarchy from a flat weight dictionary.

    This creates a generic module tree that holds parameters but doesn't
    define forward operations. Used for non-text modules (VAE, ViT, etc.)
    that need to be saved but not executed during quantization.
    """
    root = nn.Module()

    # Group weights by first path component
    children = {}
    for name, tensor in weight_dict.items():
        parts = name.split(".", 1)
        if len(parts) == 1:
            root.register_parameter(parts[0], nn.Parameter(tensor, requires_grad=False))
        else:
            child_name = parts[0]
            if child_name not in children:
                children[child_name] = {}
            children[child_name][parts[1]] = tensor

    for child_name, child_weights in children.items():
        child_module = _build_module_from_weights(child_weights)
        root.add_module(child_name, child_module)

    return root


def _load_safetensors_weights(model_path):
    """Load all weights from safetensors files in the model directory.

    BAGEL stores all weights across ae.safetensors (VAE) and ema.safetensors
    (LLM + other modules), referenced by model.safetensors.index.json.
    """
    from safetensors.torch import load_file

    all_weights = {}

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})

        # Determine which shard files contain non-VAE weights
        # VAE weights: decoder.*, encoder.* (in ae.safetensors)
        lm_shard_files = set()
        vae_only_files = set()
        for weight_name, shard_file in weight_map.items():
            if weight_name.startswith(("decoder.", "encoder.")):
                vae_only_files.add(shard_file)
            else:
                lm_shard_files.add(shard_file)

        # Load all shard files that contain non-VAE weights
        loaded_files = set()
        for shard_file in lm_shard_files:
            if shard_file in loaded_files:
                continue
            sf_path = os.path.join(model_path, shard_file)
            if os.path.exists(sf_path):
                weights = load_file(sf_path, device="cpu")
                # Only keep non-VAE weights from this file
                for name, tensor in weights.items():
                    if not name.startswith(("decoder.", "encoder.")):
                        all_weights[name] = tensor
                loaded_files.add(shard_file)
    else:
        # Fallback: load all safetensors files except ae.safetensors
        for sf_file in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            basename = os.path.basename(sf_file)
            if basename == "ae.safetensors":
                continue
            weights = load_file(sf_file, device="cpu")
            for name, tensor in weights.items():
                if not name.startswith(("decoder.", "encoder.")):
                    all_weights[name] = tensor

    return all_weights


class BagelForQuantization(nn.Module):
    """Wrapper for BAGEL model that's compatible with auto_round quantization.

    Contains the language_model (Qwen2+not) as the primary quantization target,
    plus non-text modules (connector, vit, etc.) stored as generic parameter holders.

    The forward() delegates to language_model for text-only calibration.
    """

    def __init__(self, config, language_model, source_model_path=None):
        super().__init__()
        self.config = config
        self.language_model = language_model
        self._source_model_path = source_model_path

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Forward pass delegating to the language_model for text-only calibration."""
        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=kwargs.get("use_cache", False),
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def save_pretrained(self, output_dir, **kwargs):
        """Save the model in a format compatible with vllm-omni's BagelPipeline.

        Saves:
          - config.json: Original BAGEL config with quantization_config added
          - model weights: All parameters as safetensors

        Note: Auxiliary files (llm_config.json, vit_config.json,
        preprocessor_config.json) and VAE tensors (encoder/decoder) are handled
        by auto_round's _copy_extra_model_files and copy_missing_tensors_from_source.
        """
        from safetensors.torch import save_file

        os.makedirs(output_dir, exist_ok=True)

        # Save config.json with the quantization_config
        config_dict = self.config.to_dict()
        # Remove internal PretrainedConfig fields
        for key in list(config_dict.keys()):
            if key.startswith("_"):
                del config_dict[key]
        config_dict["architectures"] = ["BagelForConditionalGeneration"]
        config_dict["model_type"] = "bagel"
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        # Save all model parameters as safetensors
        state_dict = {}
        for name, param in self.named_parameters():
            state_dict[name] = param.data.contiguous()

        # Remap weight names to match original BAGEL checkpoint format
        # The BagelPipeline expects top-level names like:
        #   language_model.model.layers.0.self_attn.q_proj.weight
        #   connector.fc1.weight
        #   vit_model.vision_model.embeddings...
        #   encoder.*, decoder.* (VAE, but those are in ae.safetensors)
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))


def load_bagel_model(model_path, torch_dtype="auto"):
    """Load a BAGEL model for quantization.

    Args:
        model_path: Path to the BAGEL model directory.
        torch_dtype: Data type for model weights.

    Returns:
        Tuple of (model, tokenizer).
    """
    # Load configs
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        bagel_config_dict = json.load(f)

    llm_config_dict = bagel_config_dict.get("llm_config", {})

    # Check for separate llm_config.json
    llm_config_path = os.path.join(model_path, "llm_config.json")
    if os.path.exists(llm_config_path):
        with open(llm_config_path, "r", encoding="utf-8") as f:
            llm_config_dict = json.load(f)

    from transformers import Qwen2Config

    llm_config = Qwen2Config(**llm_config_dict)
    # BAGEL always uses qk_norm
    llm_config.qk_norm = True

    # Determine torch_dtype
    if torch_dtype == "auto":
        model_dtype_str = bagel_config_dict.get("torch_dtype", "bfloat16")
        if model_dtype_str == "bfloat16":
            resolved_dtype = torch.bfloat16
        elif model_dtype_str == "float16":
            resolved_dtype = torch.float16
        else:
            resolved_dtype = torch.float32
    else:
        resolved_dtype = torch_dtype

    logger.info("Building Qwen2ForCausalLM with not extensions for BAGEL...")

    # Create the language model (Qwen2 + not extensions)
    language_model = Qwen2ForCausalLM(llm_config)
    _add_mot_extensions(language_model, llm_config)

    # Load all weights
    logger.info(f"Loading weights from {model_path}...")
    all_weights = _load_safetensors_weights(model_path)

    # Separate language_model weights from other component weights
    lm_weights = {}
    other_weights = {}
    for name, tensor in all_weights.items():
        if name.startswith("language_model."):
            lm_name = name[len("language_model.") :]
            lm_weights[lm_name] = tensor
        else:
            other_weights[name] = tensor

    # Load language_model weights
    missing, unexpected = language_model.load_state_dict(lm_weights, strict=False)
    if missing:
        logger.warning(f"Missing keys in language_model: {len(missing)} keys")
        for k in missing[:10]:
            logger.warning(f"  Missing: {k}")
    if unexpected:
        logger.warning(f"Unexpected keys in language_model: {len(unexpected)} keys")
        for k in unexpected[:10]:
            logger.warning(f"  Unexpected: {k}")

    # Build the BAGEL config
    bagel_config = BagelConfig(
        **{k: v for k, v in bagel_config_dict.items() if k not in ("llm_config", "architectures")}
    )
    bagel_config.llm_config = llm_config.to_dict()
    bagel_config.architectures = ["BagelForConditionalGeneration"]

    # Create the wrapper model
    model = BagelForQuantization(bagel_config, language_model, source_model_path=model_path)

    # Add non-text modules as parameter holders
    # These won't be quantized but will be saved with the model
    if other_weights:
        non_text_module = _build_module_from_weights(other_weights)
        for child_name, child_module in non_text_module.named_children():
            if not hasattr(model, child_name):
                model.add_module(child_name, child_module)
        # Also add direct parameters
        for param_name, param in non_text_module.named_parameters(recurse=False):
            if not hasattr(model, param_name):
                model.register_parameter(param_name, param)

    # Convert to target dtype
    model = model.to(resolved_dtype)
    model.eval()

    # Set name_or_path for auto_round compatibility
    model.name_or_path = model_path
    model.config._name_or_path = model_path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    logger.info(
        f"BAGEL model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters, "
        f"language_model has {llm_config.num_hidden_layers} layers"
    )

    return model, tokenizer
