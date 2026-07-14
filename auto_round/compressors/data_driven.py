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
import copy
import gc
import time
from functools import partial
from typing import Any, Optional, Union

import accelerate
import torch
from accelerate.big_modeling import dispatch_model
from tqdm import tqdm

from auto_round.calibration.utils import (
    _update_inputs,
)
from auto_round.compressors.base import BaseCompressor
from auto_round.compressors.utils import (
    _get_quantized_layer_names_outside_blocks,
    immediate_pack,
    is_act_static,
    is_nv_fp,
)
from auto_round.data_type.utils import update_block_global_scale_if_needed
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import materialize_model_
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    clear_memory,
    compress_layer_names,
    convert_module_to_hp_if_necessary,
    flatten_list,
    get_block_names,
    get_lm_head_name,
    get_module,
    global_state,
    is_auto_device_mapping,
    memory_monitor,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
    set_module,
    to_device,
)
from auto_round.utils.device import (
    _force_trim_malloc,
)
from auto_round.utils.device_manager import device_manager
from auto_round.wrapper import WrapperMultiblock


class DataDrivenCompressor(BaseCompressor):  # TODO rename this to Compressor

    def __init__(
        self,
        config: Union[object, list[object]],  # TODO rename wenhuach
        model: Union[torch.nn.Module, str],
        tokenizer: Any = None,
        platform: str = "hf",
        format: Union[str, list, None] = None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **kwargs,
        )
        # Routed to ``self._calibration_state.dataset`` via @property.
        # Set after ``super().__init__()`` because the state object is created there.
        self.dataset = dataset

    @property
    def need_calib(self) -> bool:
        """Whether this compressor instance actually needs calibration data.

        Returns True when imatrix/opt-rtn is enabled, activation calibration is
        needed (e.g. act_dynamic=False with NV FP types), or an AutoScheme is in
        use.  Returns False for pure zero-shot RTN cases.
        """
        # During early __init__ quantize_config may not exist yet — default to True.
        if not hasattr(self, "quantize_config") or self.quantize_config is None:
            return True
        return self._needs_calibration_data()

    def _needs_calibration_data(self) -> bool:
        """Determine whether calibration data is truly required.

        Calibration data IS required when:
        - imatrix optimization is enabled (enable_imatrix=True)
        - Static activation quantization is needed (act_dynamic=False with NV FP)
        - AutoScheme is being used (needs delta-loss evaluation)
        - The quantizer uses iterative optimization (iters > 0, i.e., SignRound)

        Otherwise, zero-shot (RTN) quantization can proceed without data.
        """
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

        qcfg = self.quantize_config
        if qcfg is None:
            return True

        # SignRound always needs data
        if not isinstance(qcfg, RTNConfig):
            return True

        # AutoScheme needs data for delta-loss scheme selection
        if isinstance(self.scheme, AutoScheme):
            return True

        # opt-rtn (imatrix optimization) needs data when enabled
        if isinstance(qcfg, RTNConfig) and not getattr(qcfg, "disable_opt_rtn", False):
            return True

        # Check if activation calibration is needed
        from auto_round.compressors.utils import check_need_act_calibration

        act_bits = getattr(qcfg, "act_bits", None)
        act_data_type = getattr(qcfg, "act_data_type", None)
        act_dynamic = getattr(qcfg, "act_dynamic", None)
        is_act_quantize = act_bits is not None and act_bits <= 8
        if is_act_quantize and check_need_act_calibration(
            act_dynamic,
            act_data_type,
            act_bits if act_bits is not None else 16,
            static_kv_dtype=getattr(self, "static_kv_dtype", None),
            static_attention_dtype=getattr(self, "static_attention_dtype", None),
        ):
            return True

        return False

    def post_init(self) -> None:
        """Run base post-init then attach the registered calibrator strategy.

        Subclasses (MLLM/Diffusion) override ``calib`` directly on the
        Compressor; the calibrator owns ``try_cache_inter_data_gpucpu`` /
        ``cache_inter_data`` orchestration plus the LLM ``calib`` body.
        """
        if self._post_init_done:
            return
        super().post_init()
        if self.need_calib and self.calibration is None:
            from auto_round.calibration import get_calibrator

            kind = self._get_calibrator_kind()
            self.calibration = get_calibrator(kind)(self)

    def _get_calibrator_kind(self) -> str:
        """Return the registry name of the calibrator to use.

        Default ``"llm"``.  ``MLLMMixin`` / ``DiffusionMixin`` override this
        to select ``"mllm"`` / ``"diffusion"``.
        """
        return "llm"

    @torch.no_grad()
    def try_cache_inter_data_gpucpu(  # TODO the following two should have some differences wenhuach
        self,
        block_names: list,
        nsamples: int,
        layer_names: Optional[list] = None,
        last_cache_name: Optional[str] = None,
    ) -> Any:
        """Thin wrapper around ``self.calibration.collect``.

        Public API kept for backward compatibility (entry.py and
        LLM-Compressor integration).
        """
        if self.calibration is None:
            self.post_init()
        return self.calibration.collect(block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name)

    @torch.no_grad()
    def cache_inter_data(
        self,
        block_names: list,
        nsamples: int,
        layer_names: Optional[list] = None,
        last_cache_name: Optional[str] = None,
    ) -> Any:
        """Thin wrapper around ``self.calibration.cache_inter_data``.

        Public API kept for backward compatibility.
        """
        if self.calibration is None:
            self.post_init()
        return self.calibration.cache_inter_data(
            block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name
        )

    @torch.no_grad()
    def calib(self, nsamples: int, bs: int) -> Any:
        """Thin wrapper around ``self.calibration.calib``.

        ``MLLMMixin`` and ``DiffusionMixin`` override this method directly via
        Python MRO; for plain LLM models this routes into ``LLMCalibrator.calib``.
        """
        if self.calibration is None:
            self.post_init()
        return self.calibration.calib(nsamples, bs)

    # ── Hook registration API (explicit register/remove pattern) ──────────────

    def get_preprocessor_fp_hooks(self, block: torch.nn.Module) -> list:
        """Register preprocessor hooks for the FP-input reference forward pass.

        Preprocessor hooks (e.g. AWQ activation-stats collection) run during
        the FP reference forward and must be processed before quantizer hooks.
        Subclasses can override to add custom preprocessor hooks.

        Returns a flat list of hook handles; caller must call h.remove() on each.
        """
        handles = []
        for pre in self.pipeline.preprocessors:
            handles.extend(pre.register_fp_input_forward_hooks(block))
        return handles

    def get_quantizer_fp_hooks(self, block: torch.nn.Module) -> list:
        """Register quantizer hooks for the FP-input reference forward pass.

        Includes act_max hooks (for static activation quantization) and
        quantizer-specific hooks (imatrix, etc.).
        Subclasses can override to add custom quantizer hooks.

        Returns a flat list of hook handles; caller must call h.remove() on each.
        """
        handles = self._register_act_max_hooks(block)
        handles.extend(self.pipeline.block_quantizer.register_fp_input_forward_hooks(block))
        return handles

    def get_preprocessor_qinput_hooks(self, block: torch.nn.Module) -> list:
        """Register preprocessor hooks for the quantized-input forward pass.

        Preprocessor hooks that need statistics from quantized activations
        (e.g. AWQ with quantized input). Subclasses can override to add custom hooks.

        Returns a flat list of hook handles; caller must call h.remove() on each.
        """
        handles = []
        for pre in self.pipeline.preprocessors:
            if hasattr(pre, "register_qinput_forward_hooks"):
                handles.extend(pre.register_qinput_forward_hooks(block))
        return handles

    def get_quantizer_qinput_hooks(self, block: torch.nn.Module) -> list:
        """Register quantizer hooks for the quantized-input forward pass.

        Includes act_max hooks and quantizer-specific qinput hooks.
        Used when enable_quanted_input=True to collect statistics from quantized
        activations. Subclasses can override to add custom hooks.

        Returns a flat list of hook handles; caller must call h.remove() on each.
        """
        handles = self._register_act_max_hooks(block)
        handles.extend(self.pipeline.block_quantizer.register_qinput_forward_hooks(block))
        return handles

    # ── Activation-calibration hook infrastructure ───────────────────────────────

    def _register_act_max_hooks(self, model: torch.nn.Module) -> list:
        """Register per-module act_max tracking hooks for static activation quantization.

        Returns a list of hook handles that the caller must remove when done.
        """
        from auto_round.data_type.utils import reshape_pad_tensor_by_group_size

        quantizer = self.pipeline.block_quantizer
        is_act_nv_fp = getattr(quantizer.config, "is_act_nv_fp", False)
        layer_config = quantizer.layer_config or {}

        def collect_act_max(module, input, output):
            input = input[0] if isinstance(input, (tuple, list)) else input
            if input.numel() == 0:
                return
            input, _, _ = reshape_pad_tensor_by_group_size(input, module.act_group_size)
            act_max = torch.max(torch.abs(input), dim=-1).values
            if not hasattr(module, "act_max") or module.act_max.numel() == 0:
                module.act_max = act_max
                if is_act_nv_fp:
                    max_val = act_max.max()
                    module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
                return

            act_max = act_max.to(module.act_max.device)
            if is_act_nv_fp:
                max_val = torch.max(act_max.max(), module.act_max.max())
                module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
            else:
                module.act_max = torch.max(act_max, module.act_max)

        def should_collect(name, module):
            from auto_round.compressors.utils import check_need_act_calibration
            from auto_round.utils import SUPPORTED_LAYER_TYPES, check_to_quantized

            if isinstance(module, tuple(SUPPORTED_LAYER_TYPES)):
                return (
                    hasattr(module, "act_dynamic")
                    and check_need_act_calibration(module.act_dynamic, module.act_data_type, module.act_bits)
                    and check_to_quantized(module)
                )
            if name in layer_config:
                config = layer_config[name]
                act_dynamic = config.get("act_dynamic", True)
                act_data_type = config.get("act_data_type", None)
                act_bits = config.get("act_bits", 16)
                return (
                    config["bits"] <= 8
                    and check_need_act_calibration(act_dynamic, act_data_type, act_bits)
                    and check_to_quantized(config)
                )
            return False

        handles = []
        if should_collect("", model):
            handles.append(model.register_forward_hook(collect_act_max))
            return handles
        for name, module in model.named_modules():
            if name and should_collect(name, module):
                handles.append(module.register_forward_hook(collect_act_max))
        return handles

    def _preprocess_block_inputs(self, inputs, first_input_name="input_ids"):
        # Thin wrapper around auto_round.calibration.inputs.preprocess_block_inputs.
        from auto_round.calibration.inputs import preprocess_block_inputs

        return preprocess_block_inputs(
            inputs,
            model_context=self.model_context,
            compress_context=self.compress_context,
            first_input_name=first_input_name,
        )

    def _quantize_blocks(
        self,
        model: torch.nn.Module,
        inputs: dict,
        block_names: list,
        q_input: torch.Tensor | None = None,
        nblocks: int = 1,
        pbar: tqdm | None = None,
        input_others_extra_blocks: dict | None = None,
    ):
        """Quantize and dequantize the weights of the specified blocks in the model.

        Args:
        model: The PyTorch model to be quantized.
        inputs: The input data for quantization.
        block_names: The names of the blocks to be quantized and dequantized.
        nblocks: The number of blocks to quantize and dequantize.
        device: The device for quantization and dequantization.

        Returns:
        None
        """
        clear_memory()
        for n, m in model.named_parameters():
            m.requires_grad_(False)

        input_ids, input_others = self._preprocess_block_inputs(inputs)

        if pbar is None:
            pbar = tqdm(range(0, len(block_names), nblocks))

        for i in range(0, len(block_names), nblocks):
            if input_others_extra_blocks and block_names[i] in input_others_extra_blocks:
                input_others = input_others_extra_blocks[block_names[i]]
                _, input_others = self._preprocess_block_inputs(input_others)
                input_others_extra_blocks.pop(block_names[i])
            if i != 0:
                pbar.update(1)
            if nblocks == 1:
                n = block_names[i]
                pbar.set_description(f"Quantizing {n}")
                m = get_module(model, n)
            else:
                names = block_names[i : min(i + nblocks, len(block_names))]
                pbar.set_description(f"Quantizing [{i + 1}-{min(i + nblocks, len(block_names))}]/{len(block_names)}")
                modules = [get_module(model, n) for n in names]
                m = WrapperMultiblock(modules)

            if self.compress_context.low_cpu_mem_usage:
                if nblocks == 1:
                    self._offloader.reload(model, n)
                else:
                    self._offloader.reload(model, names)

            block_name_or_names = n if nblocks == 1 else names

            # ── Infrastructure: materialize, dtype convert, device placement ──
            materialize_model_(m)
            convert_module_to_hp_if_necessary(m, self.model_context.amp_dtype, device_manager.device)

            m, _, _ = self.pipeline.dispatch_block(m, input_ids, input_others)

            # ── Pipeline lifecycle: per-block setup ───────────────────────────
            from auto_round.algorithms.pipeline import BlockContext

            current_block_names = (
                block_name_or_names if isinstance(block_name_or_names, list) else [block_name_or_names]
            )
            current_block_name = current_block_names[0] if len(current_block_names) == 1 else str(block_name_or_names)
            # bs = self.quantizer.batch_size * self.quantizer.infer_bs_coeff #TODO change to calib wenhuach
            bs = self.batch_size  # TODO add infer_bs_coeff

            ctx = BlockContext(
                model=model,
                block=m,
                block_names=current_block_names,
                block_name=current_block_name,
                block_index=i,
                bs=bs,
                device=device_manager.device,
                is_mllm=self.model_context.is_mllm,
                is_diffusion=self.model_context.is_diffusion,
                pbar=pbar,
            )

            # ── Step 1: Preprocessor calibration (e.g. AWQ activation stats) ──
            with torch.no_grad():
                pre_hooks = self.get_preprocessor_fp_hooks(m)
                try:
                    if pre_hooks:
                        self.block_forward(m, input_ids, input_others)
                finally:
                    for h in pre_hooks:
                        h.remove()

                # Preprocessor qinput hooks: use q_input if available, otherwise fp input
                pre_q_hooks = self.get_preprocessor_qinput_hooks(m)
                try:
                    if pre_q_hooks:
                        self.block_forward(m, q_input if q_input is not None else input_ids, input_others)
                finally:
                    for h in pre_q_hooks:
                        h.remove()

            # ── Step 2: pre_quantize_block (preprocessor stats consolidation + weight transforms) ──
            for pre in self.pipeline.preprocessors:
                pre.pre_quantize_block(ctx)

            # ── Step 3: Quantizer calibration (act_max, imatrix, etc.) ────────
            with torch.no_grad():
                quant_hooks = self.get_quantizer_fp_hooks(m)
                try:
                    reference_output = self.block_forward(m, input_ids, input_others)
                finally:
                    for h in quant_hooks:
                        h.remove()

                # Quantizer qinput hooks: use q_input if available, otherwise fp input
                if self.pipeline.block_quantizer.enable_quanted_input:
                    q_hooks = self.get_quantizer_qinput_hooks(m)
                    try:
                        if q_hooks:
                            self.block_forward(m, q_input if q_input is not None else input_ids, input_others)
                    finally:
                        for h in q_hooks:
                            h.remove()

            # ── Infrastructure: swap q_input ──────────────────────────────────
            if q_input is not None:
                if input_ids is not q_input:
                    clear_memory(input_ids)
                else:
                    clear_memory()
                input_ids = q_input

            # ── MoE scale alignment for FP8 dispatch efficiency ────────────────
            if is_nv_fp(self.act_data_type) or not self.act_dynamic:
                set_amax_for_all_moe_layers(m, attr_name="act_max")

            update_block_global_scale_if_needed(model, self.data_type, self.group_size)

            # ── Pure algorithm: block_quantizer.quantize_block ────────────────
            self.pipeline.block_quantizer.quantize_block(m, input_ids, input_others, reference_output, q_input, ctx)

            # ── Pipeline lifecycle: post_quantize_block ───────────────────────
            for pre in self.pipeline.preprocessors:
                pre.post_quantize_block(ctx)

            # ── Infrastructure: collect q_outputs if needed ───────────────────
            if self.pipeline.block_quantizer.enable_quanted_input:
                with torch.no_grad():
                    q_input = self.block_forward(m, input_ids, input_others)
            else:
                q_input = None

            # ── Infrastructure: hook removal, device cleanup, logging ─────────
            if len(device_manager.device_list) > 1 and not self.model_context.is_diffusion:
                accelerate.hooks.remove_hook_from_submodules(m)
            mv_module_from_gpu(m)
            # if self.enable_torch_compile:
            #     torch._dynamo.reset()
            #     self.quantizer._invalidate_block_forward_cache()
            # Keep old-arch semantics: the next block's FP reference input comes
            # from the current block's reference output, while q_input (when
            # enabled) is only used as the quantized-input companion for the
            # next block.
            next_input_ids = reference_output
            clear_memory(input_ids if input_ids is not next_input_ids else None)
            memory_monitor.log_summary()

            # ── Infrastructure: immediate_pack / shard write ──────────────────
            if self.compress_context.is_immediate_packing:
                for _n, _mod in m.named_modules():
                    if hasattr(_mod, "bits") and check_to_quantized(_mod):
                        from auto_round.compressors.utils import immediate_pack as _immediate_pack

                        module_name = getattr(_mod, "global_name", None)
                        if module_name is None and nblocks == 1 and _n:
                            module_name = f"{n}.{_n}"
                        if module_name is None:
                            continue
                        _immediate_pack(module_name, self.quantizer.layer_config)

            input_ids = next_input_ids

            if self.compress_context.is_immediate_saving:
                self.shard_writer.write(m, is_finalize=False)

            if self.compress_context.low_cpu_mem_usage and not self.compress_context.is_immediate_saving:
                if nblocks == 1:
                    self._offloader(model, n, overwrite=True)
                else:
                    for name in names:
                        self._offloader(model, name, overwrite=True)
        if pbar is not None:
            pbar.update(1)

        if not self.compress_context.is_immediate_saving:
            self.model = mv_module_from_gpu(self.model)
        for n, m in self.model.named_modules():
            if hasattr(m, "name"):
                delattr(m, "name")

        del q_input
        del input_ids
        del input_others
        del inputs

        clear_memory()

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """
        self.post_init()

        if not self.need_calib:
            return self._quantize_zero_shot()

        return self._quantize_data_driven()

    @torch.no_grad()
    def _quantize_zero_shot(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Zero-shot (RTN) quantization path — no calibration data needed.

        This replaces the standalone ``ZeroShotCompressor.quantize()`` method.
        Block-wise RTN quantization without any input data.
        """
        from auto_round.algorithms.pipeline import BlockContext

        formats = self.formats if isinstance(self.formats, list) else []
        if not (any(fmt.is_gguf() for fmt in formats) or self.super_bits is not None):
            self.quantizer.quantize_embedding_layer()  # leave to gguf itself to handle

        # Release memory
        clear_memory()

        # In RTN mode (iters == 0), force blockwise quantization to avoid
        # full-model materialization and linear CPU RAM growth.
        use_blockwise_quantization = True
        logger.info("Zero-shot mode (no calibration data needed): using blockwise quantization.")

        tied_weights_keys = getattr(self.model, "_tied_weights_keys", [])
        if tied_weights_keys is None:
            tied_weights_keys = []
        if isinstance(tied_weights_keys, dict):
            tied_weights_values = list(tied_weights_keys.values())
        else:
            tied_weights_values = list(tied_weights_keys)
        tied_weights_layers = [".".join(val.split(".")[:-1]) for val in tied_weights_values]  # rm weight/bias
        # In fact, we should detect whether it is is_separate_lm_head, to simplify, we don't do it
        if getattr(self, "formats", None) and self.formats[0].is_gguf():
            lm_head_name = get_lm_head_name(self.model)
            if lm_head_name is not None:
                tied_weights_layers.append(lm_head_name)

        all_blocks = self.quant_block_list or get_block_names(self.model)
        pbar = tqdm(range(sum(len(block) for block in all_blocks)))
        for block_names in all_blocks:
            for block_name in block_names:
                pbar.set_description(f"Quantizing {block_name}")
                block = get_module(self.model, block_name)

                # ── Infrastructure: materialize ───────────────────────────
                materialize_model_(block)

                # ── Pure algorithm ────────────────────────────────────────
                ctx = BlockContext(
                    model=self.model_context.model,
                    block=block,
                    block_names=[block_name],
                    block_name=block_name,
                    block_index=0,
                    device=device_manager.device,
                )
                # ── MoE scale alignment for FP8 dispatch efficiency ────────────────
                if is_nv_fp(self.act_data_type) or not self.act_dynamic:
                    set_amax_for_all_moe_layers(block, attr_name="act_max")

                update_block_global_scale_if_needed(block, self.data_type, self.group_size)
                self.quantizer.quantize_block(block, None, {}, None, None, ctx)
                if self.compress_context.is_immediate_packing:
                    for _n, _mod in block.named_modules():
                        if hasattr(_mod, "bits") and check_to_quantized(_mod):
                            from auto_round.compressors.utils import immediate_pack as _immediate_pack

                            module_name = getattr(_mod, "global_name", None)
                            if module_name is None and self.nblocks == 1 and _n:
                                module_name = f"{block.global_name}.{_n}"
                            if module_name is None:
                                continue
                            _immediate_pack(module_name, self.quantizer.layer_config)

                # ── Infrastructure: shard write / device cleanup ──────────
                if self.compress_context.is_immediate_saving:
                    # Save non-quantized leaf modules (e.g. norms, embeddings in block).
                    for _n, m in block.named_modules():
                        if (
                            not any(m.children())
                            and len(m.state_dict()) > 0
                            and hasattr(m, "global_name")
                            and m.global_name not in tied_weights_layers
                            and not check_to_quantized(m)
                        ):
                            set_module(self.model, m.global_name, copy.deepcopy(m))
                            self.shard_writer.write(name=m.global_name)
                            get_module(self.model, m.global_name).to("meta")
                            m.to("meta")
                    # Write at block scope for any remaining params/buffers.
                    self.shard_writer.write(name=block_name)
                    block.to("meta")
                else:
                    mv_module_from_gpu(block)
                    if self.low_cpu_mem_usage:
                        self._offloader(self.model, block_name)

                clear_memory()
                memory_monitor.log_summary()
                pbar.update(1)

        cnt = 1
        remain_layer_names = []
        block_name_set = set(name for block in all_blocks for name in block)
        for n, m in self.model_context.model.named_modules():
            if not check_to_quantized(m):
                continue
            # Skip if this layer is part of any block (by prefix match)
            if any(n == block_name or n.startswith(f"{block_name}.") for block_name in block_name_set):
                continue
            remain_layer_names.append(n)
        for name in remain_layer_names:
            logger.info(f"Quantizing remaining layer {name} on CPU.")
            self.quantizer.quantize_layer_outside_block(name)
            cnt += 1
            if cnt % 10 == 0:
                clear_memory()
                memory_monitor.log_summary()

        # Convert remaining fp8
        convert_module_to_hp_if_necessary(self.model, self.amp_dtype, self.device)
        if self.low_cpu_mem_usage:
            self._offloader.reload(self.model)
        if self.compress_context.is_immediate_saving:
            self.shard_writer.write(is_finalize=True)

        self.model_context.quantized = True
        return self.model, self.layer_config

    def _quantize_data_driven(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Data-driven quantization path — uses calibration data for optimization."""

        # Reclaim heap fragmentation from init/post_init before the memory-intensive quantize loop.
        gc.collect()
        _force_trim_malloc()

        self._check_compatibility()

        if bool(self.quant_block_list):
            all_blocks = self.quant_block_list
        else:
            all_blocks = get_block_names(self.model_context.model)

        if len(all_blocks) == 0:
            logger.warning("could not find blocks, exit with original model")
            return self.model_context.model, self.layer_config

        layer_names = _get_quantized_layer_names_outside_blocks(
            model=self.model_context.model,
            layer_config=self.layer_config,
            supported_types=SUPPORTED_LAYER_TYPES,
            quant_block_list=self.quant_block_list,
        )
        if not self.has_variable_block_shape:
            to_cache_block_names = [block[0] for block in all_blocks]
        else:
            to_cache_block_names = flatten_list(all_blocks)
        _last_cache_name = to_cache_block_names[-1] if len(to_cache_block_names) > 1 else None
        to_cache_layer_names = layer_names
        if self.super_group_size is not None:
            to_cache_layer_names = []
        if len(layer_names) > 0:
            logger.info(
                "Starting to cache block inputs. This may be slow due to external block layers: %s", layer_names
            )
        else:
            logger.info("start to cache block inputs")
        all_inputs = self.try_cache_inter_data_gpucpu(
            to_cache_block_names,
            self.nsamples,
            to_cache_layer_names,
            last_cache_name=_last_cache_name,
        )
        self.inputs = all_inputs
        is_quantized_embedding = self.quantizer.quantize_embedding_layer()
        clear_memory(device_list=device_manager.device_list)
        all_q_inputs = None
        if is_quantized_embedding:
            all_inputs = copy.deepcopy(self.inputs)
            clear_memory(self.inputs, device_list=device_manager.device_list)
            all_q_inputs = self.try_cache_inter_data_gpucpu(
                to_cache_block_names, self.nsamples, to_cache_layer_names, last_cache_name=_last_cache_name
            )
        # Remove accelerate dispatch hooks before moving parameters.
        # hf_device_map is kept for reference but hooks are no longer needed.
        if hasattr(self.model_context.model, "hf_device_map") and len(self.model_context.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.model_context.model)
        self.model_context.model = mv_module_from_gpu(self.model_context.model)
        clear_memory(device_list=device_manager.device_list)
        logger.info("caching done")
        if self.compress_context.low_cpu_mem_usage:
            if self.model_context.is_model_patched and not self.compress_context.is_immediate_saving:
                self._offloader(
                    self.model_context.model,
                    all_blocks,
                    clear_memory=True,
                    device_list=device_manager.device_list,
                )
                if not self._offloader.enabled:
                    self.compress_context.low_cpu_mem_usage = False
            else:
                self.compress_context.low_cpu_mem_usage = False
        if len(all_blocks) > 1:
            pbar = tqdm(range(0, sum([len(i) for i in all_blocks]), self.nblocks))
        else:
            pbar = tqdm(range(0, len(all_blocks[0]), self.nblocks))  # move the alg warning outside pbar

        start_time = time.time()

        for alg in self.pipeline.all():
            alg.prepare_run(self)

        for block_names in all_blocks:
            inputs = all_inputs[block_names[0]]
            all_inputs.pop(block_names[0])
            q_inputs = None
            if all_q_inputs is not None:
                q_inputs = all_q_inputs[block_names[0]]
                all_q_inputs.pop(block_names[0])

            inputs, q_inputs = _update_inputs(inputs, q_inputs)

            clear_memory(self.inputs, device_list=device_manager.device_list)

            if "input_ids" in inputs.keys():
                total_samples = len(inputs["input_ids"])
                if getattr(self.calibration_state, "batch_size", None):
                    if total_samples < self.batch_size:
                        self.batch_size = total_samples
                        self.calibration_state.batch_size = total_samples
                        logger.warning(f"force the train batch size to {total_samples}")
                else:
                    if total_samples < self.calibration_state.batch_size:
                        self.batch_size = total_samples
                        self.calibration_state.batch_size = total_samples
                        logger.warning(f"force the train batch size to {total_samples}")

            self._quantize_blocks(
                self.model_context.model,
                inputs,
                block_names,
                q_input=q_inputs if q_inputs is not None else None,
                nblocks=self.nblocks,
                pbar=pbar,
                input_others_extra_blocks=all_inputs,
            )
            if self.compress_context.is_immediate_packing and len(self.formats) != 1:
                raise ValueError(
                    f"Expected exactly one packing format when 'immediate_packing' is True, "
                    f"but got {len(self.formats)} formats."
                )

        # ── Pipeline lifecycle: finalize_quantization (model-level teardown) ─
        for alg in self.pipeline.all():
            alg.finalize_run(self)

        pbar.set_description("Quantizing done")
        pbar.close()
        if self.compress_context.low_cpu_mem_usage:
            self._offloader.reload(self.model_context.model)
        self._quantize_layers_outside_blocks(layer_names, all_inputs)

        convert_module_to_hp_if_necessary(
            self.model_context.model, self.model_context.amp_dtype, device_manager.device, to_cpu=True
        )
        if self.compress_context.is_immediate_saving:
            self.shard_writer.write(is_finalize=True)

        end_time = time.time()
        cost_time = end_time - start_time
        logger.info(f"quantization tuning time {cost_time}")

        # Dump a summary
        quantized_layers = []
        unquantized_layers = []
        for n, m in self.model_context.model.named_modules():
            if isinstance(m, tuple(SUPPORTED_LAYER_TYPES)):
                if check_to_quantized(m):
                    quantized_layers.append(n)
                else:
                    unquantized_layers.append(n)
            elif hasattr(m, "scales") or hasattr(m, "scale"):  # packing_immediately
                quantized_layers.append(n)
        summary_info = (
            f"Summary: quantized {len(quantized_layers)}/{len(quantized_layers) + len(unquantized_layers)} in the model"
        )
        if len(unquantized_layers) > 0:
            compressed_unquantized_layers = compress_layer_names(unquantized_layers)
            summary_info += f", unquantized layers: {compressed_unquantized_layers}"
        logger.info(summary_info)

        self.model_context.quantized = True
        return self.model_context.model, self.quantizer.layer_config

    def _immediate_pack_and_save_module(self, module_name):
        from auto_round.compressors.shard_writer import ShardWriter

        shard_writer = ShardWriter.get_shard_writer()
        to_cpu = self.compress_context.low_gpu_mem_usage
        module = get_module(self.model, module_name)
        if self.compress_context.is_immediate_packing:
            immediate_pack(module_name, self.layer_config)
            if to_cpu:
                module = module.to("cpu")
                packed_module = get_module(self.model, module_name)
                set_module(self.model, module_name, packed_module.to("cpu"))
        else:
            if to_cpu:
                module = module.to("cpu")
            set_module(self.model, module_name, module)
        if self.compress_context.is_immediate_saving:
            module = get_module(self.model, module_name)
            module.to("cpu")
            shard_writer.write(module, module_name, False)
            module.to("meta")

    def _quantize_layers_outside_blocks(self, layer_names: list, layer_inputs: dict) -> None:
        """Quantizes specified layers based on inputs and configuration.

        Args:
            layer_names (list): list of layer names to quantize.
            layer_inputs (dict): Dictionary mapping layer names to input data.

        Returns:
            None
        """
        # TODO currently we take all the layers outside blocks as post block layers which is not optimal
        # if there is no input for layer, we use rtn

        for layer_name in copy.deepcopy(layer_names):
            if layer_name not in layer_inputs:
                if self.act_bits < 16 and not self.act_dynamic:
                    if "lm_head" in layer_name:
                        logger.warning_once(
                            "Static activation quantization for lm_head is not fully supported yet. "
                            "If lm_head calibration inputs are missing, activation scale may fall back to unit scale "
                            "or quantization may be skipped."
                        )
                    # Activation quantization requires collected inputs
                    msg_prefix = (
                        f"Activation max hook for layer '{layer_name}' is unavailable due to "
                        f"insufficient collected inputs. "
                    )
                    if "fp8_e5m2" in self.act_data_type:
                        logger.warning(msg_prefix + "Please notes that unit scale is used for this layer.")
                    else:
                        logger.warning(
                            msg_prefix + "Static activation quantization is not supported or ineffective, "
                            "Skipping quantization for this layer."
                        )
                        layer_names.remove(layer_name)
                        continue
                self.quantizer.quantize_layer_outside_block(  # TODO check alg merge
                    layer_name,
                    input_ids=None,
                    device=device_manager.device,
                    disable_opt_rtn=getattr(self, "disable_opt_rtn", False),
                )
                layer_names.remove(layer_name)
                if self.compress_context.is_immediate_packing:
                    immediate_pack(layer_name, self.layer_config)

                if self.compress_context.is_immediate_saving:
                    m = get_module(self.model, layer_name)
                    self.shard_writer.write(m, name=layer_name, is_finalize=False)
        if len(layer_names) == 0:
            memory_monitor.update()
            memory_monitor.log_summary()
            return
        q_layer_inputs = None
        enable_quanted_input = self.enable_quanted_input
        has_gguf = False

        if hasattr(self, "formats") and self.formats is not None:
            has_gguf = any(format_.is_gguf() for format_ in self.formats)
        if has_gguf and self.compress_context.is_immediate_packing:
            enable_quanted_input = False

        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1 and enable_quanted_input:
            dispatch_model(self.model, self.model.hf_device_map)

        if enable_quanted_input:
            logger.info("starting to cache layer inputs for %s, this may be quite slow ", layer_names)
            q_layer_inputs = self.try_cache_inter_data_gpucpu([], self.nsamples, layer_names=layer_names)
            if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                accelerate.hooks.remove_hook_from_submodules(
                    self.model
                )  # self.model.hf_device_map has not been changed
        if not self.compress_context.is_immediate_saving:
            self.model = mv_module_from_gpu(self.model)
        clear_memory()
        quant_layer = self.quantizer.quantize_layer_outside_block
        for layer_name in layer_names:
            layer_input = layer_inputs[layer_name]
            layer_input = to_device(layer_input, self.compress_context.cache_device)
            q_layer_input = q_layer_inputs.get(layer_name, None) if q_layer_inputs is not None else None
            q_layer_input = to_device(q_layer_input, self.compress_context.cache_device)
            quant_layer(layer_name, layer_input, q_layer_input, device=device_manager.device)
            if self.compress_context.is_immediate_packing:
                immediate_pack(layer_name, self.quantizer.layer_config)

            if self.compress_context.is_immediate_saving:
                m = get_module(self.model, layer_name)
                self.shard_writer.write(m, name=layer_name, is_finalize=False)
            del layer_input
            clear_memory(q_layer_input)
            memory_monitor.log_summary()

    def _check_compatibility(self) -> None:
        """Checks compatibility of the configurations and model."""
        # ``seqlen`` clamping is owned by ``CalibrationState``.
        self.calibration_state.clamp_seqlen(self.model_context)

        if self.group_size == 0 and "fp8" not in self.data_type:
            logger.warning("`group_size==0` is not supported for data_type other than fp8 ")

        # if ( # TODO wenhuach add this log
        #     self.bits <= 2
        #     and (self.iters < 1000 or not getattr(self.quantize_config, "enable_alg_ext", False))
        #     and self.super_group_size is None
        # ):
        #     logger.warning(
        #         "for bits <= 2, it is recommended to enable `auto-round-best` " "and turn on `--enable_alg_ext` "
        #     )

    # This is the API for llm-compressor, not used in AutoRound
    def quantize_block(
        self,
        block: torch.nn.Module,
        inputs: Any,
        q_input: Union[torch.Tensor, dict, None] = None,
        device: Union[str, torch.device] = "cpu",  # TODO Delete wenhuach
        auto_offload: bool = True,
    ) -> Any:
        """Quantize a single decoded block of the model (public API for LLM-Compressor).

        This method handles both data-driven and zero-shot (RTN) quantization.
        When calibration data is not needed, ``inputs`` and ``q_input`` are accepted
        for interface compatibility but not used for algorithm purposes.

        Args:
            block: The transformer block (decoder layer) to quantize.
            inputs: Either:

                - the raw decoding-layer inputs captured by
                  LLM-Compressor's hook (list of ``((args, kwargs),)`` tuples),
                  in which case they are normalized via
                  :meth:`normalize_decoding_layer_inputs_`; **or**
                - a :class:`~auto_round.calibration.state.CalibrationState`
                  instance produced by a :class:`~auto_round.calibration.base.Calibrator`,
                  which is bound directly without re-normalization.
            q_input: Optional quantized input from the previous block.  ``None`` on
                the first block.
            device: Target device for quantization (e.g. ``"cuda:0"``).
            auto_offload: When *True*, use the device-map-aware offloading path;
                otherwise move ``block`` directly to ``device``.

        Returns:
            tuple: ``(q_outputs, reference_output)`` where *q_outputs* is the
            block's output after quantization (or ``None`` when
            ``enable_quanted_input`` is ``False``), and *reference_output* is the
            full-precision reference output collected before optimization.
        """

        def normalize_decoding_layer_inputs_(
            self, decoding_layer_inputs: list[tuple[tuple[Any, dict[str, Any]]]]
        ) -> None:
            """Replay captured decoding-layer calls to populate ``self.inputs``.

            Converts the raw ``(args, kwargs)`` tuples captured by LLM-Compressor's
            input hook into the ``self.inputs`` dict format expected by
            :meth:`quantize_block`.  The logic mirrors the old-arch implementation in
            ``compressors/base.py``.

            Args:
                decoding_layer_inputs:
                    A list of entries captured by a forward hook on the decoding layer.
                    Each element is a tuple whose first item is ``(args, kwargs)``.
            """
            first_block_name = self.quant_block_list[0][0]

            class _FakeDecodingLayer(torch.nn.Module):

                def forward(self, *args, **kwargs):
                    return args, kwargs

            fake_layer = _FakeDecodingLayer()
            fake_layer.orig_forward = fake_layer.forward
            fake_layer._true_orig_forward = lambda *a, **kw: (a, kw)
            fake_layer.forward = partial(self.calibration._get_block_forward_func(first_block_name), fake_layer)

            self.inputs = {}
            self.last_cache_name = None
            for step_input in decoding_layer_inputs:
                args, kwargs = step_input[0]
                fake_layer(*args, **kwargs)

        from auto_round.calibration.state import CalibrationState

        if self.diffusion:
            raise NotImplementedError(
                f"Currently, {self.__class__.__name__} does not support quantize_block for diffusion models."
            )

        # Ensure post_init has been called (sets up model_context, compress_context,
        # quantizer, layer_config, etc.).
        if not self._post_init_done:
            self.post_init()

        # ── Zero-shot (RTN) path: no calibration data needed ──────────────────
        if not self.need_calib:
            from auto_round.algorithms.pipeline import BlockContext

            materialize_model_(block)
            convert_module_to_hp_if_necessary(block, self.model_context.amp_dtype, device)
            block = block.to(device)

            ctx = BlockContext(
                model=self.model_context.model,
                block=block,
                block_names=[getattr(block, "global_name", "")],
                block_name=getattr(block, "global_name", ""),
                block_index=0,
                device=device,
            )
            self.quantizer.quantize_block(block, None, {}, None, None, ctx)

            # ── MoE scale alignment for FP8 dispatch efficiency ────────────────
            if is_nv_fp(self.act_data_type) or not self.act_dynamic:
                set_amax_for_all_moe_layers(block, attr_name="act_max")

            mv_module_from_gpu(block)
            return None, None

        if len(self.quant_block_list) != 1 or len(self.quant_block_list[0]) != 1:
            raise ValueError(
                f"{self.__class__.__name__}.quantize_block supports exactly one target block, "
                f"but quant_block_list is {self.quant_block_list!r}. "
                "Use to_quant_block_names to select a single block."
            )
        expected_block_name = self.quant_block_list[0][0]
        actual_block_name = getattr(block, "global_name", None)
        if actual_block_name is not None and actual_block_name != expected_block_name:
            raise ValueError(
                f"quantize_block received block {actual_block_name!r}, but cached inputs are for "
                f"{expected_block_name!r}. Pass the matching block or update to_quant_block_names."
            )

        # When called from LLM-Compressor, `wrapped_model` is a single decoder layer
        # (not the full VL model), so it must not be treated as an MLLM regardless of
        # whether the original model had multimodal assets.  Force is_mllm=False for
        # the duration of this call to stay on the standard LLM quantize_block path.
        orig_is_mllm = self.model_context.is_mllm
        self.model_context.is_mllm = False

        if isinstance(inputs, CalibrationState):
            # Caller already produced a CalibrationState (typically via
            # ``Calibrator.collect``).  Bind it as the authoritative store so
            # the quantizer reads the same ``inputs`` / ``attention_mask`` /
            # ``batch_dim``.
            self.calibration_state = inputs
        else:
            normalize_decoding_layer_inputs_(self, inputs)
        block_inputs = self.inputs[self.quant_block_list[0][0]]
        input_ids, input_others = self._preprocess_block_inputs(block_inputs, "hidden_states")

        # ── Infrastructure: materialize, dtype convert, device placement ──────
        materialize_model_(block)
        convert_module_to_hp_if_necessary(block, self.model_context.amp_dtype, device)

        if auto_offload:
            if (
                is_auto_device_mapping(device_manager.device_map)
                and len(device_manager.device_list) > 1
                and not self.model_context.is_diffusion
            ):
                from auto_round.utils.device import set_auto_device_map_for_block_with_tuning

                card_0_in_high_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                    block,
                    device_manager.device_list,
                    input_ids,
                    self.compress_context.low_gpu_mem_usage,
                    self.calibration_state.batch_size,
                    device,
                )
            else:
                block = block.to(device)
                card_0_in_high_risk, loss_device = False, device
        else:
            card_0_in_high_risk, loss_device = False, device

        if len(device_manager.device_list) > 1 and auto_offload:
            from accelerate.hooks import AlignDevicesHook, add_hook_to_module

            for n, m in block.named_modules():
                if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                    continue
                add_hook_to_module(m, AlignDevicesHook(m.tuning_device, io_same_device=True), True)

        blk_name = self.quant_block_list[0][0]
        # bs = self.batch_size * self.quantizer.infer_bs_coeff
        bs = self.calibration_state.batch_size  # TODO wenhuach add infer_bs_coeff

        from auto_round.algorithms.pipeline import BlockContext

        ctx = BlockContext(
            model=self.model_context.model,
            block=block,
            block_names=[blk_name],
            block_name=blk_name,
            block_index=0,
            bs=bs,
            device=device,
            is_mllm=False,
            is_diffusion=False,
        )

        # ── Step 1: Preprocessor calibration (e.g. AWQ activation stats) ──
        pre_hooks = self.get_preprocessor_fp_hooks(block)
        try:
            if pre_hooks:
                self.block_forward(block, input_ids, input_others)
        finally:
            for h in pre_hooks:
                h.remove()

        # Preprocessor qinput hooks: use q_input if available, otherwise fp input
        pre_q_hooks = self.get_preprocessor_qinput_hooks(block)
        try:
            if pre_q_hooks:
                self.block_forward(block, q_input if q_input is not None else input_ids, input_others)
        finally:
            for h in pre_q_hooks:
                h.remove()

        # ── Step 2: pre_quantize_block (preprocessor stats consolidation + weight transforms) ──
        for pre in self.pipeline.preprocessors:
            pre.pre_quantize_block(ctx)

        # ── Step 3: Quantizer calibration (act_max, imatrix, etc.) ────────
        quant_hooks = self.get_quantizer_fp_hooks(block)
        try:
            reference_output = self.block_forward(block, input_ids, input_others)
        finally:
            for h in quant_hooks:
                h.remove()

        # Quantizer qinput hooks: use q_input if available, otherwise fp input
        if self.pipeline.block_quantizer.enable_quanted_input:
            q_hooks = self.get_quantizer_qinput_hooks(block)
            try:
                if q_hooks:
                    self.block_forward(block, q_input if q_input is not None else input_ids, input_others)
            finally:
                for h in q_hooks:
                    h.remove()

        if q_input is not None:
            if input_ids is not q_input:
                clear_memory(input_ids, device_list=device_manager.device_list)
            else:
                clear_memory(device_list=device_manager.device_list)
            input_ids = q_input

        # ── Pure algorithm: block_quantizer.quantize_block ────────────────────
        self.pipeline.block_quantizer.quantize_block(block, input_ids, input_others, reference_output, q_input, ctx)

        # ── Pipeline lifecycle: post_quantize_block ───────────────────────────
        for pre in self.pipeline.preprocessors:
            pre.post_quantize_block(ctx)

        # ── MoE scale alignment for FP8 dispatch efficiency ────────────────
        if is_nv_fp(self.act_data_type) or not self.act_dynamic:
            set_amax_for_all_moe_layers(block, attr_name="act_max")

        # ── Collect quantized-block outputs ───────────────────────────────────
        if self.pipeline.block_quantizer.enable_quanted_input:
            q_outputs = self.block_forward(block, input_ids, input_others)
        else:
            q_outputs = None

        # ── Cleanup ───────────────────────────────────────────────────────────
        if len(device_manager.device_list) > 1:
            accelerate.hooks.remove_hook_from_submodules(block)
        mv_module_from_gpu(block)
        self.model_context.is_mllm = orig_is_mllm
        return q_outputs, reference_output
