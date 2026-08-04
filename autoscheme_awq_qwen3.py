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

"""Combine AutoRound's *AutoScheme* (per-layer mixed-bit allocation) with a
SFMP-style *AWQ* (activation-aware weight quantization) for Qwen3.

* AutoScheme (from ``auto_round``) is used **only** to decide the bit-width of
  every linear layer given a target average bit-width.
* The actual quantization is a standalone AWQ implementation adapted from
  ``SFMP/AWQ/awq.py`` (activation-aware scale search + weight clipping),
  **without** the sensitivity-file / reorder / real-pack machinery, because the
  per-layer bit allocation now comes from AutoScheme instead.
* Only **Qwen3** is supported and only **fake** (simulated) quantization is done
  the weights are quantize-dequantized in place and the model is saved as a
  normal fp16/bf16 checkpoint.

Example
-------
python autoscheme_awq_qwen3.py \
    --model Qwen/Qwen3-8B \
    --avg_bits 3.0 \
    --options "W2A16,W4A16" \
    --group_size 128 \
    --device 0 \
    --save_dir ./qwen3-awq-autoscheme
"""

from __future__ import annotations

import argparse
import functools
import gc
import time
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from auto_round import AutoScheme
from auto_round.auto_scheme.gen_auto_scheme import GenScheme
from auto_round.calib_dataset import get_dataloader
from auto_round.data_type.register import register_dtype
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad
from auto_round.schemes import QuantizationScheme, preset_name_to_scheme


# ---------------------------------------------------------------------------
# AWQ quantization formula -- the single source of truth.
#
# We register it into AutoRound's data-type registry under the ``int_awq`` name
# (with rtn_/opt_rtn_ aliases so every AutoScheme dispatch path resolves to it).
# This makes AutoScheme score/allocate per-layer bits using *exactly* the AWQ
# quantizer, i.e. auto-round aligns to AWQ (not the other way around).
# ---------------------------------------------------------------------------
def _awq_asym_qdq(tensor, bits=4, group_size=-1, **kwargs):
    """AWQ asymmetric (zero-point) group-wise quantize-dequantize."""
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    max_int = 2**bits - 1
    max_val = tensor.amax(dim=-1, keepdim=True)
    min_val = tensor.amin(dim=-1, keepdim=True)
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    q = torch.clamp(torch.round(tensor / scales) + zeros, 0, max_int)
    qdq = ((q - zeros) * scales).to(tensor.dtype)
    qdq = revert_tensor_by_pad(qdq, orig_shape=orig_shape, pad_len=pad_len)
    return qdq, scales, zeros


def _awq_sym_qdq(tensor, bits=4, group_size=-1, q_scale_thresh=1e-5, **kwargs):
    """Full-range symmetric group-wise quantize-dequantize.

    Matches AutoRound's ``quant_tensor_rtn_sym`` (llama.cpp "full range")
    instead of the restricted-range SFMP formula. This matters a lot at low
    bits: restricted range (``scale = absmax / (2**(bits-1) - 1)``) wastes the
    most-negative level, e.g. 2-bit collapses to only 3 effective levels
    ``{-1, 0, 1}``, roughly halving the resolution and badly hurting accuracy.
    Full range uses all ``2**bits`` levels (``{-2, -1, 0, 1}`` for 2-bit).
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** (bits - 1)
    wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
    wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    wmin_abs = -wmin_tmp
    wmax_abs = wmax_tmp
    # Pick the larger magnitude side, keeping its sign (so the level with the
    # extra step is placed where the weights actually reach).
    max_v = (2 * (wmax_abs < wmin_abs).int() - 1) * torch.max(wmax_abs, wmin_abs)
    scale = max_v / maxq
    scale = torch.where(scale < 0, scale.clamp(max=-q_scale_thresh), scale.clamp(min=q_scale_thresh))
    scale = scale.unsqueeze(dim=-1)
    int_w = torch.round(tensor / scale).clamp(-maxq, maxq - 1)
    qdq = (int_w * scale).to(tensor.dtype)
    qdq = revert_tensor_by_pad(qdq, orig_shape=orig_shape, pad_len=pad_len)
    return qdq, scale, None


# Register under every prefix AutoScheme's ``get_quant_func`` may probe
# (plain / rtn / opt_rtn) so the AWQ formula is used regardless of the path.
register_dtype(["int_awq_asym", "rtn_int_awq_asym", "opt_rtn_int_awq_asym"])(_awq_asym_qdq)
register_dtype(["int_awq_sym", "rtn_int_awq_sym", "opt_rtn_int_awq_sym"])(_awq_sym_qdq)

# The data_type name AutoScheme options should carry so scoring uses AWQ's quant.
AWQ_DATA_TYPE = "int_awq"


# ---------------------------------------------------------------------------
# Fake (simulated) group-wise weight quantization
# ---------------------------------------------------------------------------
@torch.no_grad()
def pseudo_quantize_tensor(w: torch.Tensor, n_bit: int, q_group_size: int = 128, sym: bool = True) -> torch.Tensor:
    """Group-wise fake quantize-dequantize a weight tensor.

    Uses the **same** AWQ quant formula (``_awq_sym_qdq`` / ``_awq_asym_qdq``)
    that is registered into AutoRound's data-type registry, so the AWQ quantizer
    and the AutoScheme scoring quantizer are guaranteed identical.

    Args:
        w: weight tensor of shape ``[out_features, in_features]``.
        n_bit: integer bit-width (2/3/4/8 ...).
        q_group_size: group size along the input-feature dim; ``-1`` = per-channel.
        sym: symmetric (True) or asymmetric/zero-point (False) quantization.
    """
    gs = q_group_size if (q_group_size is not None and q_group_size > 0) else -1
    quant_fn = _awq_sym_qdq if sym else _awq_asym_qdq
    qdq, _, _ = quant_fn(w, bits=n_bit, group_size=gs)
    return qdq.to(w.dtype)


@torch.no_grad()
def get_act_scale(x: torch.Tensor) -> torch.Tensor:
    return x.abs().view(-1, x.shape[-1]).mean(0)


# ---------------------------------------------------------------------------
# Scale / clip application helpers (adapted from SFMP/AWQ/awq.py)
# ---------------------------------------------------------------------------
@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]
    scales = scales.to(ln.weight.device)
    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    scales = scales.to(fc1.weight.device)
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))
    fc2.weight.mul_(scales.view(1, -1))


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


def get_op_name(module, op):
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError("Cannot find op in module")


def get_op_by_name(module, op_name):
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module")


# ---------------------------------------------------------------------------
# AutoScheme: decide the bit-width of every linear layer
# ---------------------------------------------------------------------------
def build_layer_bits(model, tokenizer, avg_bits, options, group_size, dataset, device_map, sym_override=None):
    """Run AutoScheme and return a dict: full_layer_name -> resolved scheme dict.

    ``sym_override`` keeps AutoScheme's scoring quantizer consistent with the AWQ
    quantizer: when the user forces asymmetric (or symmetric) AWQ, the candidate
    options fed to AutoScheme are rewritten to the same symmetry, so the per-layer
    bit allocation is decided under the *same* quantization the weights will
    actually receive.
    """
    # Only quantize linear layers inside the decoder blocks (exclude lm_head etc.)
    quant_layer_names = [
        name for name, m in model.named_modules() if isinstance(m, nn.Linear) and name.startswith("model.layers.")
    ]

    # Rewrite every candidate option so AutoScheme scores/allocates bits using the
    # AWQ quantizer (data_type=int_awq). ``sym_override`` (when set) additionally
    # forces the symmetry to match the AWQ run; otherwise the preset's own sym is
    # kept. Either way the scoring quantizer == the AWQ quantizer.
    resolved_options = []
    for opt in options:
        scheme = preset_name_to_scheme(opt.upper()) if isinstance(opt, str) else opt.copy()
        if not isinstance(scheme, QuantizationScheme):
            scheme = QuantizationScheme.from_dict(dict(scheme))
        if sym_override is not None:
            scheme.sym = sym_override
        scheme.data_type = AWQ_DATA_TYPE
        resolved_options.append(scheme)
    options = resolved_options

    auto_scheme = AutoScheme(
        avg_bits=avg_bits,
        options=options,
        # Count only weight bits so the target avg_bits means the same thing for
        # symmetric and asymmetric quantization.
        ignore_scale_zp_bits=False,
    )
    # AutoScheme copies its own device_map / dataset onto GenScheme.
    auto_scheme.device_map = device_map
    auto_scheme.dataset = dataset

    gen = GenScheme(
        auto_scheme,
        model,
        quant_layer_names,
        fixed_layer_scheme={},
        dataset=dataset,
        device_map=device_map,
        tokenizer=tokenizer,
        enable_torch_compile=False,
    )
    layer_config = gen.get_layer_config()

    # Normalize into a plain {name: {"bits":.., "group_size":.., "sym":..}} dict.
    per_layer = {}
    for name, cfg in layer_config.items():
        cfg = dict(cfg)
        bits = int(cfg.get("bits", 16))
        gs = cfg.get("group_size", group_size)
        sym = bool(cfg.get("sym", True))
        per_layer[name] = {"bits": bits, "group_size": gs if gs is not None else group_size, "sym": sym}
    return per_layer


# ---------------------------------------------------------------------------
# AWQ (SFMP-style, Qwen3 only) with per-layer bits from AutoScheme
# ---------------------------------------------------------------------------
class AWQQwen3:
    def __init__(
        self,
        model,
        tokenizer,
        per_layer_bits,
        device="cuda",
        n_grid=20,
        apply_clip=True,
        sym_override=None,
        duo_scaling=True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.per_layer_bits = per_layer_bits
        self.dev = device
        self.n_grid = n_grid
        self.apply_clip = apply_clip
        self.duo_scaling = duo_scaling
        # None -> follow the per-layer scheme from AutoScheme;
        # True/False -> force symmetric / asymmetric quantization for every layer.
        self.sym_override = sym_override

    def _bits_for(self, layer_idx, fc_name):
        key = f"model.layers.{layer_idx}.{fc_name}"
        cfg = self.per_layer_bits.get(key, {"bits": 16, "group_size": 128, "sym": True})
        if self.sym_override is not None:
            cfg = {**cfg, "sym": self.sym_override}
        return cfg

    def _quant(self, w, layer_idx, fc_name):
        cfg = self._bits_for(layer_idx, fc_name)
        if cfg["bits"] >= 16:
            return w
        return pseudo_quantize_tensor(w, cfg["bits"], cfg["group_size"], cfg["sym"])

    @torch.no_grad()
    def _catch_first_inputs(self, samples):
        layers = self.model.model.layers
        inps, layer_kwargs = [], {}

        layers[0] = layers[0].to(self.dev)
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.dev)
        self.model.model.norm = self.model.model.norm.to(self.dev)
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.to(self.dev)

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps.append(inp)
                layer_kwargs.update(kwargs)
                raise ValueError

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.module, name)

        layers[0] = Catcher(layers[0])
        try:
            self.model(samples.to(self.dev))
        except ValueError:
            pass
        layers[0] = layers[0].module
        inps = inps[0]

        # Disable KV-cache in the captured kwargs: we call blocks repeatedly during
        # the AWQ scale grid-search, and a live cache would accumulate KV states and
        # corrupt the attention shapes. A cache-free prefill is exactly what
        # calibration needs.
        layer_kwargs.pop("past_key_value", None)
        layer_kwargs.pop("past_key_values", None)
        layer_kwargs["use_cache"] = False

        layers[0] = layers[0].cpu()
        self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
        self.model.model.norm = self.model.model.norm.cpu()
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        return inps, layer_kwargs

    @torch.no_grad()
    def _search_scale(self, block, linears2scale: dict, x, kwargs, layer_idx):
        x = x.to(next(block.parameters()).device)
        org_out = block(x, **kwargs)
        if isinstance(org_out, tuple):
            org_out = org_out[0]

        x_mean = get_act_scale(x)
        # duo_scaling additionally balances by the weight magnitude, matching
        # AutoRound/AutoAWQ default (scales = x_mean^ratio / w_mean^(1-ratio)).
        # Activation-only scaling (SFMP) is noticeably worse at low bits.
        if self.duo_scaling:
            fcs = list(linears2scale.values())
            first_name = next(iter(linears2scale))
            gs = self._bits_for(layer_idx, first_name)["group_size"]
            w_mean = self._weight_mean(fcs, gs).to(x.device)
        best_error, best_scales = float("inf"), None
        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}

        for i in range(self.n_grid):
            ratio = i / self.n_grid
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4).view(-1)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1
            for fc_name, fc in linears2scale.items():
                w_dtype = fc.weight.dtype
                scale_row = scales.view(1, -1).to(fc.weight.device)
                fc.weight.mul_(scale_row.to(w_dtype))
                # Keep the weight in its original dtype: scales may be float32
                # (w_mean is computed in float) and dividing a bf16/fp16 weight by
                # a float32 tensor would silently promote it, breaking the block's
                # matmul dtype on the next forward.
                fc.weight.data = (self._quant(fc.weight.data, layer_idx, fc_name) / scale_row).to(w_dtype)

            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            loss = (org_out - out).float().pow(2).mean().item()
            if loss < best_error:
                best_error, best_scales = loss, scales
            block.load_state_dict(org_sd)

        return best_scales.view(-1).detach().cpu()

    @staticmethod
    def _weight_mean(layers, group_size):
        """Per-input-channel mean of group-normalized abs weights across balance layers.

        Mirrors AutoRound's ``AWQTransform._compute_layer_means``.
        """
        weight = torch.cat([m.weight.detach().float() for m in layers], dim=0)
        org_shape = weight.shape
        gs = group_size if (group_size and group_size > 0 and org_shape[1] % group_size == 0) else org_shape[1]
        w = weight.reshape(-1, gs)
        w_scale = w.abs() / (w.abs().amax(dim=1, keepdim=True) + 1e-6)
        w_scale = w_scale.reshape(org_shape)
        return w_scale.mean(0)

    @torch.no_grad()
    def _auto_scale_block(self, module, module_kwargs, input_feat, layer_idx):
        scales_list = []
        module_kwargs = dict(module_kwargs)
        module_kwargs.pop("use_cache", None)

        def _get(prev_op, layers, inp, module2inspect=None, kwargs={}):
            if module2inspect is None:
                module2inspect = list(layers.values())[0]
            scales = self._search_scale(module2inspect, layers, inp, kwargs, layer_idx)
            return (
                get_op_name(module, prev_op),
                tuple(get_op_name(module, m) for m in layers.values()),
                scales,
            )

        # attention input: input_layernorm -> q/k/v
        scales_list.append(
            _get(
                prev_op=module.input_layernorm,
                layers={
                    "self_attn.q_proj": module.self_attn.q_proj,
                    "self_attn.k_proj": module.self_attn.k_proj,
                    "self_attn.v_proj": module.self_attn.v_proj,
                },
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out: v_proj -> o_proj (only when shapes match, i.e. no GQA mismatch)
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _get(
                    prev_op=module.self_attn.v_proj,
                    layers={"self_attn.o_proj": module.self_attn.o_proj},
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        # mlp: post_attention_layernorm -> gate/up
        scales_list.append(
            _get(
                prev_op=module.post_attention_layernorm,
                layers={
                    "mlp.gate_proj": module.mlp.gate_proj,
                    "mlp.up_proj": module.mlp.up_proj,
                },
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )
        # mlp: up_proj -> down_proj
        scales_list.append(
            _get(
                prev_op=module.mlp.up_proj,
                layers={"mlp.down_proj": module.mlp.down_proj},
                inp=input_feat["mlp.down_proj"],
            )
        )
        return scales_list

    @torch.no_grad()
    def _apply_scale(self, module, scales_list, input_feat_dict):
        for prev_op_name, layer_names, scales in scales_list:
            prev_op = get_op_by_name(module, prev_op_name)
            layers = [get_op_by_name(module, n) for n in layer_names]

            prev_op.to(self.dev)
            for layer in layers:
                layer.to(self.dev)
            scales = scales.to(self.dev)

            if isinstance(prev_op, nn.Linear):
                scale_fc_fc(prev_op, layers[0], scales)
            elif isinstance(prev_op, (nn.LayerNorm, Qwen3RMSNorm)) or "RMSNorm" in type(prev_op).__name__:
                scale_ln_fcs(prev_op, layers, scales)
            else:
                raise NotImplementedError(f"prev_op {type(prev_op)} not supported")

            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

            prev_op.cpu()
            for layer in layers:
                layer.cpu()

    @torch.no_grad()
    def _auto_clip_block(self, module, input_feat, layer_idx, max_shrink=0.5, n_sample_token=512):
        named_linears = {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}
        clip_list = []
        for name, fc in named_linears.items():
            # Only q_proj/k_proj are skipped (qk bmm makes clipping imprecise).
            # o_proj IS clipped here even though its v->o smoothing is skipped
            # under GQA (v_proj/o_proj shape mismatch) -- clipping is independent
            # of smoothing and still reduces o_proj's quantization error.
            if any(t in name for t in ["q_proj", "k_proj"]):
                continue
            cfg = self._bits_for(layer_idx, name)
            if cfg["bits"] >= 16:
                continue
            fc.to(self.dev)
            max_val, min_val = self._clip_layer(
                fc.weight, input_feat[name].to(self.dev), cfg, layer_idx, name, max_shrink, n_sample_token
            )
            clip_list.append((name, max_val, min_val))
            fc.cpu()
        return clip_list

    @torch.no_grad()
    def _clip_layer(self, w, input_feat, cfg, layer_idx, name, max_shrink, n_sample_token):
        sym = cfg["sym"]
        gs = cfg["group_size"] if cfg["group_size"] and cfg["group_size"] > 0 else w.shape[1]
        if w.shape[1] % gs != 0:
            gs = w.shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, gs)
        step = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step]

        w = w.reshape(w.shape[0], 1, -1, gs)
        oc_bs = 256 if w.shape[0] % 256 == 0 else 64
        if w.shape[0] % oc_bs != 0:
            oc_bs = w.shape[0]

        best_max_all, best_min_all = [], []
        for i_b in range(w.shape[0] // oc_bs):
            wb = w[i_b * oc_bs : (i_b + 1) * oc_bs]
            org_out = (input_feat * wb).sum(dim=-1)
            if sym:  # symmetric: shrink |max|, min = -max
                org_max = wb.abs().amax(dim=-1, keepdim=True)
                org_min = -org_max
            else:  # asymmetric: shrink actual max & min together (SFMP-style)
                org_max = wb.amax(dim=-1, keepdim=True)
                org_min = wb.amin(dim=-1, keepdim=True)
            best_max = org_max.clone()
            best_min = org_min.clone()
            min_errs = torch.ones_like(org_max) * 1e9
            for i_s in range(int(max_shrink * self.n_grid)):
                f = 1 - i_s / self.n_grid
                max_val = org_max * f
                min_val = org_min * f
                cur_w = torch.minimum(torch.maximum(wb, min_val), max_val)
                q_w = pseudo_quantize_tensor(cur_w.reshape(oc_bs, -1), cfg["bits"], gs, sym).reshape(cur_w.shape)
                cur_out = (input_feat * q_w).sum(dim=-1)
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                cur_best = err < min_errs
                min_errs[cur_best] = err[cur_best]
                best_max[cur_best] = max_val[cur_best]
                best_min[cur_best] = min_val[cur_best]
            best_max_all.append(best_max)
            best_min_all.append(best_min)
        return torch.cat(best_max_all, dim=0).squeeze(1), torch.cat(best_min_all, dim=0).squeeze(1)

    @torch.no_grad()
    def _apply_clip(self, module, clip_list):
        for name, max_val, min_val in clip_list:
            layer = get_op_by_name(module, name)
            layer.to(self.dev)
            max_val = max_val.to(layer.weight.device)
            min_val = min_val.to(layer.weight.device)
            org_shape = layer.weight.shape
            layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            layer.weight.data = torch.minimum(torch.maximum(layer.weight.data, min_val), max_val)
            layer.weight.data = layer.weight.data.reshape(org_shape)
            layer.cpu()

    @torch.no_grad()
    def run(self, nsamples=128, seqlen=512):
        dataloader = get_dataloader(self.tokenizer, seqlen, nsamples=nsamples, bs=1)
        collected = []
        for d in dataloader:
            if d is None:
                continue
            collected.append(d["input_ids"])
            if len(collected) >= nsamples:
                break
        samples = torch.cat(collected, dim=0)

        inps, layer_kwargs = self._catch_first_inputs(samples)
        layers = self.model.model.layers

        for i in tqdm(range(len(layers)), desc="AWQ (AutoScheme bits)"):
            layer = layers[i].to(self.dev)
            named_linears = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}

            input_feat = defaultdict(list)

            def hook(m, x, y, name):
                input_feat[name].append(x[0].detach().cpu())

            handles = [m.register_forward_hook(functools.partial(hook, name=n)) for n, m in named_linears.items()]
            inps = inps.to(self.dev)
            inps = layer(inps, **layer_kwargs)
            if isinstance(inps, tuple):
                inps = inps[0]
            for h in handles:
                h.remove()
            input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

            scales_list = self._auto_scale_block(layer, layer_kwargs, input_feat, i)
            self._apply_scale(layer, scales_list, input_feat)

            if self.apply_clip:
                clip_list = self._auto_clip_block(layer, input_feat, i)
                self._apply_clip(layer, clip_list)

            # fake-quantize the weights in place using the per-layer bits
            for n, m in named_linears.items():
                m.to(self.dev)
                m.weight.data = self._quant(m.weight.data, i, n)
                m.cpu()

            layers[i] = layer.cpu()
            del input_feat
            gc.collect()
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AutoScheme + SFMP-style AWQ for Qwen3 (fake quant)")
    parser.add_argument("--model", required=True, help="Qwen3 model id or local path")
    parser.add_argument("--avg_bits", type=float, default=3.0, help="Target average bit-width for AutoScheme")
    parser.add_argument(
        "--options",
        type=str,
        default="W2A16,W3A16",
        help="Comma-separated AutoScheme options, e.g. 'W2A16,W4A16'",
    )
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--device", type=str, default="0", help="cuda device index, e.g. '0', or 'cpu'")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="Disable AWQ weight clipping. Clipping is ON by default (it clips all "
        "quantized linears except q_proj/k_proj, including o_proj).",
    )
    parser.add_argument(
        "--no_duo_scaling",
        action="store_true",
        help="Use activation-only AWQ scaling (SFMP style). Default is duo_scaling "
        "(activation + weight aware), matching AutoRound/AutoAWQ, which is more "
        "accurate especially at low bits.",
    )
    parser.add_argument(
        "--quant_sym",
        type=str,
        default="auto",
        choices=["auto", "sym", "asym"],
        help="Weight quantization symmetry: 'auto' follows the AutoScheme option "
        "(W*A16 presets are symmetric), 'sym' forces symmetric, 'asym' forces "
        "asymmetric (zero-point, SFMP-style) for every quantized layer.",
    )
    parser.add_argument("--save_dir", type=str, default="./qwen3-awq-autoscheme")
    args = parser.parse_args()

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    dev = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    # AutoScheme's device_map follows the AutoRound convention: int index for a
    # single GPU, or "cpu".
    auto_scheme_device_map = "cpu" if args.device == "cpu" else int(args.device)
    sym_override = {"auto": None, "sym": True, "asym": False}[args.quant_sym]
    options = [o.strip() for o in args.options.split(",") if o.strip()]

    tick = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 1) AutoScheme: decide per-layer bits (loads its own model instance).
    print("==== Step 1: AutoScheme per-layer bit allocation ====")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.eval()
    per_layer_bits = build_layer_bits(
        model,
        tokenizer,
        avg_bits=args.avg_bits,
        options=options,
        group_size=args.group_size,
        dataset=args.dataset,
        device_map=auto_scheme_device_map,
        sym_override=sym_override,
    )
    for name, cfg in per_layer_bits.items():
        print(f"  {name}: {cfg['bits']} bit (g{cfg['group_size']}, sym={cfg['sym']})")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2) Reload a fresh model on CPU and run SFMP-style AWQ with those bits.
    print("==== Step 2: SFMP-style AWQ quantization (fake) ====")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.eval()

    awq = AWQQwen3(
        model,
        tokenizer,
        per_layer_bits,
        device=dev,
        apply_clip=not args.no_clip,
        sym_override=sym_override,
        duo_scaling=not args.no_duo_scaling,
    )
    awq.run(nsamples=args.nsamples, seqlen=args.seqlen)

    # 3) Save fake-quantized model.
    print(f"==== Step 3: saving to {args.save_dir} ====")
    model = model.cpu()
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    print(f"Done in {time.time() - tick:.1f}s -> {args.save_dir}")


if __name__ == "__main__":
    main()
