# Copyright (c) 2023 Intel Corporation
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


def _normalize_gemma4_per_layer_input(positional_inputs, hidden_states):
    if positional_inputs is None or len(positional_inputs) == 0:
        return positional_inputs

    per_layer_input = positional_inputs[0]
    if not isinstance(per_layer_input, torch.Tensor) or per_layer_input.shape[1] == hidden_states.shape[1]:
        return positional_inputs

    hs_seq = hidden_states.shape[1]
    pl_seq = per_layer_input.shape[1]
    if hs_seq <= pl_seq:
        per_layer_input = per_layer_input[:, :hs_seq, :]
    else:
        pad = per_layer_input[:, -1:, :].expand(-1, hs_seq - pl_seq, -1)
        per_layer_input = torch.cat([per_layer_input, pad], dim=1)

    normalized_inputs = list(positional_inputs)
    normalized_inputs[0] = per_layer_input
    return type(positional_inputs)(normalized_inputs) if isinstance(positional_inputs, tuple) else normalized_inputs


def _get_gemma4_shared_kv_states_global(block):
    """Return the shared KV states dict for Gemma4 block-wise quantization."""
    ref = getattr(block, "_shared_kv_states_global_ref", None)
    if ref is not None:
        return ref
    return {}


def _get_gemma4_rotary_emb(block, default_rotary_emb=None):
    rotary_emb_ref = getattr(block, "_rotary_emb_ref", None)
    if rotary_emb_ref:
        return rotary_emb_ref[0]
    return getattr(block, "_rotary_emb", default_rotary_emb)


def _rebuild_gemma4_attention_mask(config, hidden_states, position_ids, layer_type, past_key_values=None):
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    mask_builder = create_causal_mask if layer_type == "full_attention" else create_sliding_window_causal_mask
    return mask_builder(
        config=config,
        inputs_embeds=hidden_states,
        attention_mask=None,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )


def _prepare_gemma4_replay_inputs(
    block,
    rotary_input,
    *,
    position_ids=None,
    position_embeddings=None,
    attention_mask=None,
    shared_kv_states=None,
    past_key_values=None,
    config=None,
    default_rotary_emb=None,
    default_shared_kv_states=None,
):
    attn = getattr(block, "self_attn", None)
    layer_type = getattr(attn, "layer_type", None)
    head_dim = getattr(attn, "head_dim", None)

    if attn is not None and hasattr(attn, "store_full_length_kv") and shared_kv_states is None:
        if default_shared_kv_states is not None:
            shared_kv_states = default_shared_kv_states
        else:
            shared_kv_states = _get_gemma4_shared_kv_states_global(block)
            if getattr(block, "layer_idx", None) == 0:
                shared_kv_states.clear()

    need_position_embeddings = position_embeddings is None
    if isinstance(position_embeddings, dict):
        cached_position_embeddings = position_embeddings.get(layer_type) if layer_type is not None else None
        need_position_embeddings = cached_position_embeddings is None
    else:
        cached_position_embeddings = position_embeddings

    if (
        not need_position_embeddings
        and head_dim is not None
        and isinstance(cached_position_embeddings, (tuple, list))
        and cached_position_embeddings
    ):
        need_position_embeddings = cached_position_embeddings[0].shape[-1] != head_dim

    if need_position_embeddings and layer_type is not None and position_ids is not None:
        rotary_emb = _get_gemma4_rotary_emb(block, default_rotary_emb)
        if rotary_emb is not None:
            rebuilt_position_embeddings = rotary_emb(rotary_input, position_ids, layer_type)
            if isinstance(position_embeddings, dict):
                position_embeddings = dict(position_embeddings)
                position_embeddings[layer_type] = rebuilt_position_embeddings
            else:
                position_embeddings = rebuilt_position_embeddings

    if config is not None and layer_type is not None and position_ids is not None:
        try:
            # Resolve via package namespace so
            # patch("auto_round.special_model_handler._rebuild_gemma4_attention_mask") keeps working.
            import auto_round.special_model_handler as _smh

            attention_mask = _smh._rebuild_gemma4_attention_mask(
                config,
                hidden_states=rotary_input,
                position_ids=position_ids,
                layer_type=layer_type,
                past_key_values=past_key_values,
            )
        except Exception:
            pass

    return {
        "position_embeddings": position_embeddings,
        "attention_mask": attention_mask,
        "shared_kv_states": shared_kv_states,
    }


def _patch_gemma4_model(model):
    """Patch each Gemma4 decoder layer so it recomputes position_embeddings and
    attention_mask from the cached position_ids when the cached versions have
    wrong dimensions (sliding_attention vs full_attention head_dims differ).

    During auto-round block-wise quantization the cached inputs from block 0
    (always a sliding_attention layer) are reused for every subsequent block.
    Full-attention layers (head_dim=512) would receive position embeddings
    computed for sliding layers (head_dim=256), causing a shape mismatch crash.
    """
    import types as _types

    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    except ImportError:
        return model

    # Find the Gemma4TextModel by traversing the hierarchy
    text_model = None
    for _, submodule in model.named_modules():
        if isinstance(submodule, Gemma4TextModel):
            text_model = submodule
            break

    if text_model is None:
        return model

    rotary_emb = text_model.rotary_emb
    # Shared dict to propagate KV state between anchor/sharer layers (like
    # Gemma4TextModel.forward does in newer transformers versions).
    shared_kv_states_global = {}

    for layer in text_model.layers:
        object.__setattr__(layer, "_autoround_special_replay", "gemma4")
        object.__setattr__(layer, "_gemma4_config_ref", text_model.config)
        original_forward = layer.forward
        layer_type = getattr(getattr(layer, "self_attn", None), "layer_type", None)
        if layer_type is None:
            continue
        head_dim = getattr(getattr(layer, "self_attn", None), "head_dim", None)
        is_full = layer_type == "full_attention"

        def _make_layer_forward(orig_fwd, lt, hd, is_full_attn, re, cfg, skv_global):
            import inspect

            orig_params = list(inspect.signature(orig_fwd).parameters)
            orig_has_shared_kv = "shared_kv_states" in orig_params

            def patched_layer_forward(
                self,
                hidden_states,
                per_layer_input=None,
                shared_kv_states=None,
                position_embeddings=None,
                attention_mask=None,
                position_ids=None,
                **kwargs,
            ):
                # Rebuild Gemma4 layer-specific replay inputs from the minimal
                # shared cache so later layers do not need variable block inputs.
                if shared_kv_states is None and getattr(self, "layer_idx", None) == 0:
                    skv_global.clear()
                prepared_inputs = _prepare_gemma4_replay_inputs(
                    self,
                    hidden_states,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    shared_kv_states=shared_kv_states,
                    past_key_values=kwargs.get("past_key_values"),
                    config=cfg,
                    default_rotary_emb=re,
                    default_shared_kv_states=skv_global,
                )
                position_embeddings = prepared_inputs["position_embeddings"]
                attention_mask = prepared_inputs["attention_mask"]
                shared_kv_states = prepared_inputs["shared_kv_states"]

                # per_layer_input is token-specific but is cached as shared positional
                # input (only 1st batch stored). Truncate/pad to match hidden_states seq_len.
                if per_layer_input is not None and per_layer_input.shape[1] != hidden_states.shape[1]:
                    hs_seq = hidden_states.shape[1]
                    pl_seq = per_layer_input.shape[1]
                    if hs_seq <= pl_seq:
                        per_layer_input = per_layer_input[:, :hs_seq, :]
                    else:
                        pad = per_layer_input[:, -1:, :].expand(-1, hs_seq - pl_seq, -1)
                        per_layer_input = torch.cat([per_layer_input, pad], dim=1)

                if orig_has_shared_kv:
                    return orig_fwd(
                        hidden_states,
                        per_layer_input,
                        shared_kv_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **kwargs,
                    )
                return orig_fwd(
                    hidden_states,
                    per_layer_input,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **kwargs,
                )

            return patched_layer_forward

        layer.forward = _types.MethodType(
            _make_layer_forward(
                original_forward, layer_type, head_dim, is_full, rotary_emb, text_model.config, shared_kv_states_global
            ),
            layer,
        )

    return model


def _attach_gemma4_rotary_emb(model):
    """Attach ``_rotary_emb`` to each Gemma4 decoder layer.

    For transformers >= 5.6 the per-layer forward patch is unnecessary, but
    ``block_forward`` still needs access to ``rotary_emb`` (which lives on the
    parent ``Gemma4TextModel``) to recompute ``position_embeddings`` when the
    cached version from block 0 has the wrong dimension.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    except ImportError:
        return

    text_model = None
    for _, submodule in model.named_modules():
        if isinstance(submodule, Gemma4TextModel):
            text_model = submodule
            break

    if text_model is None:
        return

    # Create a single shared dict to propagate KV state between anchor/sharer layers.
    # Gemma4TextModel.forward in newer transformers uses the same pattern.
    shared_kv_states_global = {}

    for layer in text_model.layers:
        # Store in a plain list to prevent nn.Module from registering these
        # as child submodules (which would cause meta-tensor errors during .to(device)).
        object.__setattr__(layer, "_rotary_emb_ref", [text_model.rotary_emb])
        object.__setattr__(layer, "_shared_kv_states_global_ref", shared_kv_states_global)
        object.__setattr__(layer, "_autoround_special_replay", "gemma4")
        object.__setattr__(layer, "_gemma4_config_ref", text_model.config)


def _attach_gemma4_unified_rotary_emb(model):
    """Attach ``_rotary_emb`` to each Gemma4 decoder layer.

    For transformers >= 5.6 the per-layer forward patch is unnecessary, but
    ``block_forward`` still needs access to ``rotary_emb`` (which lives on the
    parent ``Gemma4TextModel``) to recompute ``position_embeddings`` when the
    cached version from block 0 has the wrong dimension.
    """
    try:
        from transformers.models.gemma4_unified import Gemma4UnifiedTextModel
    except ImportError:
        return

    text_model = None
    for _, submodule in model.named_modules():
        if isinstance(submodule, Gemma4UnifiedTextModel):
            text_model = submodule
            break

    if text_model is None:
        return

    # Create a single shared dict to propagate KV state between anchor/sharer layers.
    # Gemma4TextModel.forward in newer transformers uses the same pattern.
    shared_kv_states_global = {}

    for layer in text_model.layers:
        # Store in a plain list to prevent nn.Module from registering these
        # as child submodules (which would cause meta-tensor errors during .to(device)).
        object.__setattr__(layer, "_rotary_emb_ref", [text_model.rotary_emb])
        object.__setattr__(layer, "_shared_kv_states_global_ref", shared_kv_states_global)
        object.__setattr__(layer, "_autoround_special_replay", "gemma4")
        object.__setattr__(layer, "_gemma4_config_ref", text_model.config)
