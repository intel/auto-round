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
import contextlib
import importlib
import sys
import threading

import torch

from auto_round.special_model_handler.constants import SPECIAL_MULTIMODAL_BLOCK, SPECIAL_SHARED_CACHE_KEYS
from auto_round.utils import logger

def _set_quantizable_bits(module, bits=4):
    """Mark all nn.Linear children as quantizable so AutoRound wraps them."""
    import torch.nn as nn

    for _n, _m in module.named_modules():
        if not isinstance(_m, nn.Linear):
            continue
        _m.bits = getattr(_m, "bits", bits)
        _m.group_size = getattr(_m, "group_size", 128)
        _m.sym = getattr(_m, "sym", True)
        _m.data_type = getattr(_m, "data_type", "int")
        _m.scale_dtype = getattr(_m, "scale_dtype", None)
        _m.act_bits = getattr(_m, "act_bits", 16)
        _m.act_sym = getattr(_m, "act_sym", None)
        _m.act_data_type = getattr(_m, "act_data_type", None)
        _m.act_dynamic = getattr(_m, "act_dynamic", True)

_cosmos3_forward_state = threading.local()


@contextlib.contextmanager
def _cosmos3_forward_mode():
    """Context flag marking a genuine `Cosmos3OmniPipeline.__call__` forward."""
    _cosmos3_forward_state.active = True
    try:
        yield
    finally:
        _cosmos3_forward_state.active = False


def _wrap_cosmos3_layer_dual_mode(layer):
    """Monkey-patch one `Cosmos3VLTextMoTDecoderLayer` in place."""
    if getattr(layer, "_autoround_dual_mode_patched", False):
        return
    orig_forward = layer.forward

    def _dual_mode_forward(self, und_seq, gen_seq, rotary_emb):
        if getattr(_cosmos3_forward_state, "active", False):
            return orig_forward(und_seq, gen_seq, rotary_emb)

        und_len = rotary_emb[0].shape[0]
        if und_seq.shape[0] != und_len:
            combined = und_seq
            und_seq = combined[:und_len]
            gen_seq = combined[und_len:]
        und_out, gen_out = orig_forward(und_seq, gen_seq, rotary_emb)
        return torch.cat([und_out, gen_out], dim=0)

    import types

    layer.forward = types.MethodType(_dual_mode_forward, layer)
    layer._autoround_dual_mode_patched = True


def _get_cosmos3_multimodal_block(model, quant_vision=False):
    """Get block names for Cosmos3."""
    if hasattr(model, "layers"):
        return [[f"layers.{i}" for i in range(len(model.layers))]]
    return []


def _bypass_cosmos3_safety_checker():
    """Patch Cosmos3 safety checker so calibration does not require cosmos_guardrail."""
    # Prefer an already-registered module.  Besides avoiding an unnecessary
    # import, this makes the optional dependency easy to substitute in tests
    # and in environments where diffusers lazily registers pipeline modules.
    module_name = "diffusers.pipelines.cosmos.pipeline_cosmos3_omni"
    pipeline_cosmos3_omni = sys.modules.get(module_name)
    if pipeline_cosmos3_omni is None:
        try:
            pipeline_cosmos3_omni = importlib.import_module(module_name)
        except ImportError:
            return

    safety_checker = getattr(pipeline_cosmos3_omni, "CosmosSafetyChecker", None)
    if safety_checker is None or getattr(safety_checker, "_autoround_patched", False):
        return

    def _patched_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

    safety_checker.__init__ = _patched_init
    safety_checker.to = lambda self, *args, **kwargs: self
    safety_checker.check_text_safety = staticmethod(lambda prompt: True)
    safety_checker.check_video_safety = staticmethod(lambda frames: frames)
    safety_checker.dtype = property(lambda self: torch.float32)
    safety_checker.device = property(lambda self: torch.device("cpu"))
    safety_checker._autoround_patched = True


def _align_cosmos3_quant_config_to_vllm_omni(transformer_dir):
    """Rewrite the fused ``layers`` block name to vllm-omni's split module tree."""
    import json
    import os

    def _rewrite_blocks(blocks):
        if isinstance(blocks, str):
            blocks = [b.strip() for b in blocks.split(",") if b.strip()]
        if not isinstance(blocks, list):
            return blocks
        out = []
        for b in blocks:
            if b == "layers":
                out.extend(["language_model.layers", "gen_layers"])
            else:
                out.append(b)
        return out

    config_path = os.path.join(transformer_dir, "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    qcfg = config_data.get("quantization_config")
    if not isinstance(qcfg, dict) or "block_name_to_quantize" not in qcfg:
        return
    rewritten = _rewrite_blocks(qcfg["block_name_to_quantize"])
    qcfg["block_name_to_quantize"] = rewritten
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, sort_keys=True)

    quant_config_path = os.path.join(transformer_dir, "quantization_config.json")
    if os.path.isfile(quant_config_path):
        with open(quant_config_path, "r", encoding="utf-8") as f:
            quant_config_data = json.load(f)
        if "block_name_to_quantize" in quant_config_data:
            quant_config_data["block_name_to_quantize"] = rewritten
            with open(quant_config_path, "w", encoding="utf-8") as f:
                json.dump(quant_config_data, f, indent=2, sort_keys=True)


def load_cosmos3_diffusion(pretrained_model_name_or_path, device_str):
    """Load Cosmos3 from a diffusers checkpoint for AutoRound quantization.

    Returns a (pipe, model) pair where `model.layers` are the quantizable
    blocks.
    """
    import json
    import os
    import shutil

    from diffusers import Cosmos3OmniPipeline  # pylint: disable=import-error,no-name-in-module

    _bypass_cosmos3_safety_checker()

    diffusers_pipe = Cosmos3OmniPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
    torch.set_grad_enabled(True)
    fused_model = diffusers_pipe.transformer

    for layer in fused_model.layers:
        _wrap_cosmos3_layer_dual_mode(layer)
    _set_quantizable_bits(fused_model.layers)

    # Read transformer config for model dimensions / metadata only.
    config_path = os.path.join(pretrained_model_name_or_path, "transformer", "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    import torch.nn as nn
    from transformers import PretrainedConfig

    class _Cosmos3VllmOmniConfig(PretrainedConfig):
        model_type = "cosmos3_vllm_omni"

        def __init__(self, **kwargs):
            kwargs.pop("model_type", None)
            super().__init__(**kwargs)

    class Cosmos3VllmOmniQuantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cosmos3VllmOmniConfig(**cfg)
            self.config._class_name = "Cosmos3VFMTransformer"
            self.config.architectures = ["Cosmos3VFMTransformer"]
            self.config.name_or_path = pretrained_model_name_or_path
            self.layers = fused_model.layers

        @property
        def dtype(self):
            try:
                return next(self.parameters()).dtype
            except StopIteration:
                return torch.float32

        @property
        def device(self):
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        def save_pretrained(self, save_directory, max_shard_size="5GB", safe_serialization=True, **kwargs):
            os.makedirs(save_directory, exist_ok=True)
            self.config.save_pretrained(save_directory)
            state_dict = {name: tensor.detach().cpu() for name, tensor in self.state_dict().items()}
            if safe_serialization:
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(save_directory, "model.safetensors"))
            else:
                torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))

    model = Cosmos3VllmOmniQuantModel()
    logger.info("Cosmos3: quantizing %d decoder-block layers", len(fused_model.layers))

    class _Cosmos3Pipe:
        _autoround_diffusion_pipe = True

        def __init__(self, transformer, diffusers_pipe, pipe_cfg):
            self.transformer = transformer
            self.model = transformer
            self.diffusers_pipe = diffusers_pipe
            self.config = pipe_cfg
            self.components = {"transformer": transformer}
            self.dtype = torch.bfloat16

        @property
        def device(self):
            return next(self.transformer.parameters()).device

        def to(self, *args, **kwargs):
            self.diffusers_pipe.to(*args, **kwargs)
            for component in vars(self.diffusers_pipe).values():
                keep_fp32 = getattr(component, "_keep_in_fp32_modules", None)
                if not isinstance(component, torch.nn.Module) or not keep_fp32:
                    continue
                for sub_name, sub_module in component.named_modules():
                    if any(sub_name == name or sub_name.startswith(name + ".") for name in keep_fp32):
                        sub_module.to(torch.float32)
            return self

        def save_config(self, save_directory):
            os.makedirs(save_directory, exist_ok=True)
            model_index_path = os.path.join(pretrained_model_name_or_path, "model_index.json")
            with open(model_index_path, "r", encoding="utf-8") as f:
                model_index = json.load(f)

            model_index["_class_name"] = "Cosmos3OmniDiffusersPipeline"
            model_index["_name_or_path"] = pretrained_model_name_or_path
            for component_name, component_spec in model_index.items():
                if (
                    component_name.startswith("_")
                    or component_name == "transformer"
                    or not isinstance(component_spec, list)
                ):
                    continue
                source = os.path.join(pretrained_model_name_or_path, component_name)
                target = os.path.join(save_directory, component_name)
                if os.path.isdir(source):
                    shutil.copytree(source, target, dirs_exist_ok=True)
            with open(os.path.join(save_directory, "model_index.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps(model_index, indent=2, sort_keys=True) + "\n")

            _SKIP_TOPLEVEL = {
                "model_index.json",
                "checkpoint.json",
                "model.safetensors.index.json",
            }
            for entry in os.listdir(pretrained_model_name_or_path):
                src = os.path.join(pretrained_model_name_or_path, entry)
                if not os.path.isfile(src):
                    continue
                if entry in _SKIP_TOPLEVEL:
                    continue
                if entry.endswith((".json", ".txt", ".model")):
                    shutil.copy2(src, os.path.join(save_directory, entry))

            # Align the exported quant metadata with vllm-omni's runtime module tree.
            _align_cosmos3_quant_config_to_vllm_omni(os.path.join(save_directory, "transformer"))

        def __call__(self, prompts=None, guidance_scale=6.0, num_inference_steps=35, generator=None, **kwargs):
            """Calibration forward: drives the actual Cosmos3OmniPipeline."""
            prompt_list = prompts if isinstance(prompts, list) else [prompts]
            with _cosmos3_forward_mode():
                for prompt in prompt_list:
                    self.diffusers_pipe(
                        prompt=prompt,
                        num_frames=1,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        output_type="latent",
                        enable_safety_check=False,
                    )

    pipe = _Cosmos3Pipe(model, diffusers_pipe, dict(diffusers_pipe.config))

    return pipe, model


# Register Cosmos3 block detection.
SPECIAL_MULTIMODAL_BLOCK["cosmos3_vllm_omni"] = _get_cosmos3_multimodal_block
SPECIAL_SHARED_CACHE_KEYS["Cosmos3VllmOmniQuantModel"] = ("rotary_emb",)
