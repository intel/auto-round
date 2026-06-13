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
import traceback
from contextlib import ExitStack
from functools import partial
from typing import Any, Callable, Optional, Union

import accelerate
import torch
from accelerate.big_modeling import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory, get_max_memory
from tqdm import tqdm

from auto_round import envs
from auto_round.calibration.utils import (
    _infer_last_cache_name,
    _split_inputs_diffusion,
    _update_inputs,
)
from auto_round.compressors.base import BaseCompressor
from auto_round.compressors.utils import (
    _get_quantized_layer_names_outside_blocks,
    check_skippable_keywords,
    immediate_pack,
    init_cache,
    is_nv_fp,
    is_static_wfp8afp8,
    reset_params,
)
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import materialize_model_, safe_to_cpu_
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_seqlen_compatible,
    check_to_quantized,
    clear_memory,
    compress_layer_names,
    convert_module_to_hp_if_necessary,
    flatten_list,
    get_block_names,
    get_module,
    hook_ngram_embeddings_on_cpu,
    is_auto_device_mapping,
    is_quantized_input_module,
    memory_monitor,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
    to_device,
    to_dtype,
    wrap_block_forward_positional_to_kwargs,
)
from auto_round.utils.device import (
    _force_trim_malloc,
    parse_available_devices,
)
from auto_round.utils.device_manager import device_manager
from auto_round.wrapper import WrapperMultiblock


class DataDrivenCompressor(BaseCompressor):
    need_calib: bool = True

    def __init__(
        self,
        config: Union[object, list[object]],
        model: Union[torch.nn.Module, str],
        tokenizer: Any = None,
        platform: str = "hf",
        format: Union[str, list, None] = None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        iters: int = 200,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        **kwargs,
    ) -> None:
        if iters is None:
            iters = 200
        self.iters = iters
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
        if iters == 0:
            self.lr = 5e-3

    def post_init(self) -> None:
        """Run base post-init then attach the registered calibrator strategy.

        Subclasses (MLLM/Diffusion) override ``calib`` directly on the
        Compressor; the calibrator owns ``try_cache_inter_data_gpucpu`` /
        ``cache_inter_data`` orchestration plus the LLM ``calib`` body.
        """
        if self._post_init_done:
            return
        super().post_init()
        if self.calibration is None:
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
    def try_cache_inter_data_gpucpu(
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

    @torch.no_grad()
    def _get_block_forward_func(self, name: str) -> Callable:
        """Build the block-forward replacement, then let the calibrator wrap it.

        ``Calibrator.wrap_block_forward`` defaults to passthrough; the
        Diffusion calibrator overrides it to convert positional → kwargs.
        """
        from auto_round.calibration.hooks import make_block_forward_func

        fn = make_block_forward_func(self, name)
        if self.calibration is not None:
            fn = self.calibration.wrap_block_forward(fn)
        return fn

    @torch.no_grad()
    def _get_cache_data_hook_for_layer(self, name):
        """Thin wrapper around ``auto_round.calibration.hooks.make_layer_cache_hook``."""
        from auto_round.calibration.hooks import make_layer_cache_hook

        return make_layer_cache_hook(self, name)

    def _replace_forward(self):
        """Thin wrapper around ``auto_round.calibration.hooks.replace_forward_with_hooks``."""
        from auto_round.calibration.hooks import replace_forward_with_hooks

        replace_forward_with_hooks(self)

    def _should_stop_cache_forward(self, name: str) -> bool:
        """Delegate the early-stop policy to the active calibrator.

        Falls back to the default helper when the calibrator has not been
        constructed yet (very early init code paths).
        """
        if self.calibration is not None:
            return self.calibration.should_stop(name)
        from auto_round.calibration.hooks import should_stop_cache_forward

        return should_stop_cache_forward(self, name)

    def _preprocess_block_inputs(self, inputs, first_input_name="input_ids"):
        # Thin wrapper around auto_round.calibration.inputs.preprocess_block_inputs.
        from auto_round.calibration.inputs import preprocess_block_inputs

        return preprocess_block_inputs(
            inputs,
            model_context=self.model_context,
            compress_context=self.compress_context,
            first_input_name=first_input_name,
        )

    def _split_inputs(self, inputs: dict, first_input_name: str) -> tuple[torch.Tensor, dict]:
        # Thin wrapper around auto_round.calibration.inputs.split_inputs.
        from auto_round.calibration.inputs import split_inputs

        return split_inputs(
            inputs,
            first_input_name,
            is_diffusion=self.model_context.is_diffusion,
            shared_cache_keys=self.model_context.shared_cache_keys,
        )

    def normalize_decoding_layer_inputs_(self, decoding_layer_inputs: list[tuple[tuple[Any, dict[str, Any]]]]) -> None:
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
        fake_layer.forward = partial(self._get_block_forward_func(first_block_name), fake_layer)

        self.inputs = {}
        self.last_cache_name = None
        for step_input in decoding_layer_inputs:
            args, kwargs = step_input[0]
            fake_layer(*args, **kwargs)

    def quantize_block(
        self,
        block: torch.nn.Module,
        inputs: Any,
        q_input: Union[torch.Tensor, dict, None] = None,
        device: Union[str, torch.device] = "cpu",
        auto_offload: bool = True,
    ) -> Any:
        """Quantize a single decoded block of the model (public API for LLM-Compressor).

        This method is the new-arch equivalent of the old ``BaseCompressor.quantize_block``
        (see ``compressors/base.py``).  It is primarily consumed by LLM-Compressor:
        https://github.com/vllm-project/llm-compressor/pull/1994

        The method normalizes the raw decoding-layer inputs provided by LLM-Compressor,
        runs the full infrastructure pipeline (device placement, act-max collection,
        reference-output caching) for the given *block*, delegates the pure-algorithm
        weight optimization to ``self.quantizer.quantize_block``, then returns the
        quantized-block outputs.

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
        from auto_round.calibration.state import CalibrationState

        if self.diffusion:
            raise NotImplementedError(
                f"Currently, {self.__class__.__name__} does not support quantize_block for diffusion models."
            )

        # Ensure post_init has been called (sets up model_context, compress_context,
        # quantizer, layer_config, etc.).
        if not self._post_init_done:
            self.post_init()

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

        try:
            if isinstance(inputs, CalibrationState):
                # Caller already produced a CalibrationState (typically via
                # ``Calibrator.collect``).  Bind it as the authoritative store so
                # the quantizer reads the same ``inputs`` / ``attention_mask`` /
                # ``batch_dim``.
                self.calibration_state = inputs
            else:
                self.normalize_decoding_layer_inputs_(inputs)
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
                        self.quantizer.batch_size,
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
            bs = self.quantizer.batch_size * self.quantizer.infer_bs_coeff
            mid_iter_mem_check = self.compress_context.low_gpu_mem_usage and card_0_in_high_risk

            if not hasattr(self.quantizer, "create_block_io"):
                if q_input is None:
                    hook_handles = self.quantizer.register_calibration_hooks(block)
                    reference_output = self.quantizer._get_block_outputs(block, input_ids, input_others, bs)
                    for h in hook_handles:
                        h.remove()
                else:
                    reference_output = self.quantizer._get_block_outputs(block, input_ids, input_others, bs)
                    hook_handles = self.quantizer.register_calibration_hooks(block)
                    if hook_handles:
                        self.quantizer._get_block_outputs(block, q_input, input_others, bs, save_output=False)
                    for h in hook_handles:
                        h.remove()
                    if input_ids is not q_input:
                        clear_memory(input_ids, device_list=device_manager.device_list)
                    else:
                        clear_memory(device_list=device_manager.device_list)
                    input_ids = q_input

                self.quantizer.quantize_block(
                    block,
                    input_ids,
                    input_others,
                    reference_output,
                    loss_device=loss_device,
                    mid_iter_mem_check=mid_iter_mem_check,
                )

                if is_nv_fp(self.quantizer.act_data_type) or is_static_wfp8afp8(self.quantizer):
                    set_amax_for_all_moe_layers(block, attr_name="act_max")

                if self.quantizer.enable_quanted_input:
                    q_outputs = self.quantizer._get_block_outputs(block, input_ids, input_others, bs)
                else:
                    q_outputs = None

                if len(device_manager.device_list) > 1:
                    accelerate.hooks.remove_hook_from_submodules(block)
                mv_module_from_gpu(block)
                return q_outputs, reference_output

            from auto_round.algorithms.pipeline import BlockContext, InputSource

            ctx = BlockContext(
                model=self.model_context.model,
                block=block,
                block_names=[blk_name],
                block_name=blk_name,
                block_index=0,
                io=self.quantizer.create_block_io(input_ids, input_others, q_input, block),
                bs=bs,
                loss_device=loss_device,
                device=device,
                mid_iter_mem_check=mid_iter_mem_check,
                is_mllm=False,
                is_diffusion=False,
            )
            policy = self.pipeline.get_merged_policy(ctx)

            if policy.source == InputSource.QUANTIZED_INPUT and q_input is not None:
                with ExitStack() as fwd_stack:
                    self.pipeline.enter_preprocessor_hooks(ctx, fwd_stack)
                    reference_output = ctx.collect_reference(fwd_stack)
                with ExitStack() as fwd_stack:
                    quantizer_hooks = self.pipeline.enter_quantizer_hooks(ctx, fwd_stack)
                    if quantizer_hooks:
                        ctx.collect_quantized_stats(fwd_stack)
            else:
                with ExitStack() as fwd_stack:
                    self.pipeline.enter_block_forward_hooks(ctx, fwd_stack)
                    reference_output = ctx.collect_reference(fwd_stack)

            if q_input is not None:
                if input_ids is not q_input:
                    clear_memory(input_ids, device_list=device_manager.device_list)
                else:
                    clear_memory(device_list=device_manager.device_list)
                input_ids = q_input

            # pre_quantize_block: consolidate stats and apply weight transforms.
            for pre in self.pipeline.preprocessors:
                pre.pre_quantize_block(ctx)

            # ── Pure algorithm: block_quantizer.quantize_block ────────────────────
            self.pipeline.block_quantizer.quantize_block(ctx)

            # ── Pipeline lifecycle: post_quantize_block ───────────────────────────
            for pre in self.pipeline.preprocessors:
                pre.post_quantize_block(ctx)

            # ── MoE scale alignment for FP8 dispatch efficiency ────────────────
            if is_nv_fp(self.quantizer.act_data_type) or is_static_wfp8afp8(self.quantizer):
                set_amax_for_all_moe_layers(block, attr_name="act_max")

            # ── Collect quantized-block outputs ───────────────────────────────────
            if self.pipeline.block_quantizer.enable_quanted_input:
                q_outputs = ctx.collect_next_inputs()
            else:
                q_outputs = None

            # ── Cleanup ───────────────────────────────────────────────────────────
            if len(device_manager.device_list) > 1:
                accelerate.hooks.remove_hook_from_submodules(block)
            ctx.finish()
            mv_module_from_gpu(block)
            return q_outputs, reference_output
        finally:
            self.model_context.is_mllm = orig_is_mllm

    def _quantize_blocks(
        self,
        model: torch.nn.Module,
        inputs: dict,
        block_names: list,
        q_input: torch.Tensor = None,
        nblocks: int = 1,
        pbar: tqdm = None,
        input_others_extra_blocks: dict = None,
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
        clear_memory(device_list=device_manager.device_list)
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

            if (
                is_auto_device_mapping(device_manager.device_map)
                and len(device_manager.device_list) > 1
                and not self.model_context.is_diffusion
            ):
                from auto_round.utils.device import set_auto_device_map_for_block_with_tuning

                card_0_in_high_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                    m,
                    device_manager.device_list,
                    input_ids,
                    self.compress_context.low_gpu_mem_usage,
                    self.quantizer.batch_size,
                    device_manager.device,
                )
            else:
                m = m.to(device_manager.device)
                card_0_in_high_risk, loss_device = False, device_manager.device

            if len(device_manager.device_list) > 1 and not self.model_context.is_diffusion:
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                for _n, _mod in m.named_modules():
                    if len(list(_mod.children())) != 0 or not hasattr(_mod, "tuning_device"):
                        continue
                    add_hook_to_module(_mod, AlignDevicesHook(_mod.tuning_device, io_same_device=True), True)

            # ── Pipeline lifecycle: per-block setup ───────────────────────────
            from auto_round.algorithms.pipeline import BlockContext, InputSource

            current_block_names = (
                block_name_or_names if isinstance(block_name_or_names, list) else [block_name_or_names]
            )
            current_block_name = current_block_names[0] if len(current_block_names) == 1 else str(block_name_or_names)
            bs = self.quantizer.batch_size * self.quantizer.infer_bs_coeff
            mid_iter_mem_check = self.compress_context.low_gpu_mem_usage and card_0_in_high_risk

            ctx = BlockContext(
                model=model,
                block=m,
                block_names=current_block_names,
                block_name=current_block_name,
                block_index=i,
                io=self.quantizer.create_block_io(input_ids, input_others, q_input, m),
                bs=bs,
                loss_device=loss_device,
                device=device_manager.device,
                mid_iter_mem_check=mid_iter_mem_check,
                is_mllm=self.model_context.is_mllm,
                is_diffusion=self.model_context.is_diffusion,
                pbar=pbar,
            )

            # ── Infrastructure: collect reference output and act calib ────────
            # All forward hooks (preprocessor stats + act-calib) are active during
            # the reference forward and removed when the ExitStack exits.
            policy = self.pipeline.get_merged_policy(ctx)

            if policy.source == InputSource.QUANTIZED_INPUT and q_input is not None:
                # First: reference forward with FP inputs and preprocessor hooks only.
                with ExitStack() as fwd_stack:
                    self.pipeline.enter_preprocessor_hooks(ctx, fwd_stack)
                    reference_output = ctx.collect_reference(fwd_stack)
                # Second: quantizer stats forward with q_input.
                with ExitStack() as fwd_stack:
                    quantizer_hooks = self.pipeline.enter_quantizer_hooks(ctx, fwd_stack)
                    if quantizer_hooks:
                        ctx.collect_quantized_stats(fwd_stack)
            else:
                # Unified: reference forward with all hooks active (or no hooks).
                with ExitStack() as fwd_stack:
                    self.pipeline.enter_block_forward_hooks(ctx, fwd_stack)
                    reference_output = ctx.collect_reference(fwd_stack)

            # ── Infrastructure: swap q_input ──────────────────────────────────
            if q_input is not None:
                if input_ids is not q_input:
                    clear_memory(input_ids, device_list=device_manager.device_list)
                else:
                    clear_memory(device_list=device_manager.device_list)
                input_ids = q_input

            # ── Pipeline lifecycle: pre_quantize_block (stats consolidation + weight transforms) ──
            for pre in self.pipeline.preprocessors:
                pre.pre_quantize_block(ctx)

            # ── Pure algorithm: block_quantizer.quantize_block ────────────────
            self.pipeline.block_quantizer.quantize_block(ctx)

            # ── Pipeline lifecycle: post_quantize_block ───────────────────────
            for pre in self.pipeline.preprocessors:
                pre.post_quantize_block(ctx)

            # ── MoE scale alignment for FP8 dispatch efficiency ────────────────
            if is_nv_fp(self.quantizer.act_data_type) or is_static_wfp8afp8(self.quantizer):
                set_amax_for_all_moe_layers(m, attr_name="act_max")

            # ── Infrastructure: collect q_outputs if needed ───────────────────
            if self.pipeline.block_quantizer.enable_quanted_input:
                q_input = ctx.collect_next_inputs()
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
            ctx.finish()
            clear_memory(input_ids if input_ids is not next_input_ids else None, device_list=device_manager.device_list)
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

        clear_memory(device_list=device_manager.device_list)

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """
        self.post_init()

        # Reclaim heap fragmentation from init/post_init before the memory-intensive quantize loop.
        gc.collect()
        _force_trim_malloc()

        self._check_compatibility()

        if bool(self.quantizer.quant_block_list):
            all_blocks = self.quantizer.quant_block_list
        else:
            all_blocks = get_block_names(self.model_context.model)

        if len(all_blocks) == 0:
            logger.warning("could not find blocks, exit with original model")
            return self.model_context.model, self.quantizer.layer_config

        layer_names = _get_quantized_layer_names_outside_blocks(
            model=self.model_context.model,
            layer_config=self.quantizer.layer_config,
            supported_types=SUPPORTED_LAYER_TYPES,
            quant_block_list=self.quantizer.quant_block_list,
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

        try:
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
                    if total_samples < self.quantizer.batch_size:
                        self.quantizer.batch_size = total_samples
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
        finally:
            # ── Pipeline lifecycle: finalize_quantization (model-level teardown) ─
            for alg in self.pipeline.all():
                try:
                    alg.finalize_run(self)
                except Exception as _fe:
                    logger.warning("finalize_run error in %s: %s", type(alg).__name__, _fe)

        pbar.set_description("Quantizing done")
        pbar.close()
        if self.compress_context.low_cpu_mem_usage:
            self._offloader.reload(self.model_context.model)
        self._quantize_layers(layer_names, all_inputs)

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

    def _quantize_layers(self, layer_names: list, layer_inputs: dict) -> None:
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
                self.quantizer.quantize_layer_outside_block(
                    layer_name,
                    input_ids=None,
                    device=device_manager.device,
                    disable_opt_rtn=getattr(self, "disable_opt_rtn", False),
                )
                layer_names.remove(layer_name)
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
        clear_memory(device_list=device_manager.device_list)
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
            clear_memory(q_layer_input, device_list=device_manager.device_list)
            memory_monitor.log_summary()

    def _check_compatibility(self) -> None:
        """Checks compatibility of the configurations and model."""
        # ``seqlen`` clamping is owned by ``CalibrationState``.
        self._calibration_state.clamp_seqlen(self.model_context)

        if self.group_size == 0 and "fp8" not in self.data_type:
            logger.warning("`group_size==0` is not supported for data_type other than fp8 ")

        if (
            self.bits <= 2
            and (self.iters < 1000 or not getattr(self.quantize_config, "enable_alg_ext", False))
            and self.super_group_size is None
        ):
            logger.warning(
                "for bits <= 2, it is recommended to enable `auto-round-best` " "and turn on `--enable_alg_ext` "
            )


class CalibratedRTNCompressor(DataDrivenCompressor):
    """DataDrivenCompressor variant for iters=0 RTN that needs calibration data.

    Handles two cases that require forward passes through the model:
      - Weight quantization with imatrix (importance-matrix statistics for
        improved RTN accuracy on INT / weight-only schemes).
      - Activation quantization with static scales (e.g. NVFP4, FP8_STATIC)
        where per-tensor or per-channel scale factors must be collected before
        the actual quantization step.

    Both cases use OptimizedRTNQuantizer and need a calibration dataset,
    which is why they cannot be handled by the zero-shot (no-data) path.
    """

    need_calib: bool = True

    def __init__(
        self,
        config: object,
        model: torch.nn.Module,
        **kwargs,
    ) -> None:
        kwargs["iters"] = 0
        super().__init__(
            config,
            model,
            **kwargs,
        )

    def _quantize_via_rtn_blockwise(self) -> None:
        """Quantize model layers block by block using cached inputs and imatrix."""

        all_blocks = self.quantizer.quant_block_list or get_block_names(self.model)
        if not all_blocks:
            raise ValueError("Could not find any blocks. Check the model or quant_block_list.")

        if not self.has_variable_block_shape:
            to_cache_block_names = [block[0] for block in all_blocks]
        else:
            to_cache_block_names = flatten_list(all_blocks)
        layer_names = _get_quantized_layer_names_outside_blocks(
            model=self.model_context.model,
            layer_config=self.quantizer.layer_config,
            supported_types=SUPPORTED_LAYER_TYPES,
            quant_block_list=self.quantizer.quant_block_list,
        )
        if (
            self.quantize_config.is_act_quantize
            and (not self.quantize_config.act_dynamic or len(layer_names) > 0)
            or self.has_variable_block_shape
        ):
            if len(layer_names) > 0:
                logger.warning(
                    "quantize layers outside blocks for static activation quantizaiton"
                    " will significantly increase calibration time"
                )
            all_inputs = self.try_cache_inter_data_gpucpu(to_cache_block_names, self.nsamples, layer_names)
        else:
            all_inputs = self.cache_inter_data(to_cache_block_names, self.nsamples)

        # Clear hooks for multi-GPU setups
        if hasattr(self.model_context.model, "hf_device_map") and len(self.model_context.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.model_context.model)

        pbar = tqdm(range(sum(len(block) for block in all_blocks)))

        for block_names in all_blocks:
            first_block = block_names[0]
            inputs = all_inputs.pop(first_block)
            input_keys = [k for k in inputs if k.startswith("hidden_state")]
            if len(input_keys) != 1:
                raise RuntimeError(
                    "hidden_states arg mismatch. Please file an issue at https://github.com/intel/auto-round/issues"
                )
            inputs["input_ids"] = inputs.pop(input_keys[0])

            clear_memory(self.inputs, device_list=device_manager.device_list)

            total_samples = len(inputs["input_ids"])
            if total_samples < self.batch_size:
                self.batch_size = total_samples
                logger.warning(f"Forcing batch size to {total_samples}")

            tmp_dtype = self.model_context.amp_dtype if self.model_context.amp else torch.float32

            input_ids = to_device(inputs.pop("input_ids"), self.compress_context.cache_device)
            input_ids = [id_.to(tmp_dtype) for id_ in input_ids]

            def process_input_others(input_others):
                input_others = to_device(input_others, self.compress_context.cache_device)
                # Unwrap single-element list/tuple so they are passed as bare values.
                for key in list(input_others.keys()):
                    val = input_others[key]
                    if isinstance(val, (list, tuple)) and len(val) == 1:
                        input_others[key] = val[0]
                for key, val in input_others.items():
                    if isinstance(val, torch.Tensor) and val.dtype in (torch.float16, torch.bfloat16):
                        input_others[key] = val.to(tmp_dtype)
                    elif isinstance(val, list):
                        input_others[key] = [
                            to_dtype(v, tmp_dtype)
                            for v in val
                            if not (isinstance(v, torch.Tensor) and v.dtype in (torch.int32, torch.int64))
                        ]
                return input_others

            input_others = inputs
            input_others = process_input_others(input_others)
            for block_name in block_names:
                if block_name in all_inputs.keys():
                    input_others = all_inputs[block_name]
                    input_others = process_input_others(input_others)
                    all_inputs.pop(block_name)
                pbar.set_description(f"Quantizing {block_name}")
                block = get_module(self.model_context.model, block_name)

                # ── Infrastructure: materialize, dtype convert, device placement ──
                materialize_model_(block)
                block.to("cpu")
                block = convert_module_to_hp_if_necessary(
                    block, dtype=self.model_context.amp_dtype, device=device_manager.device
                )
                if (
                    is_auto_device_mapping(device_manager.device_map)
                    and len(device_manager.device_list) > 1
                    and not self.model_context.is_diffusion
                ):
                    from auto_round.utils.device import set_auto_device_map_for_block_with_tuning

                    set_auto_device_map_for_block_with_tuning(
                        block,
                        device_manager.device_list,
                        input_ids,
                        self.compress_context.low_gpu_mem_usage,
                        self.quantizer.batch_size,
                        device_manager.device,
                    )
                    if len(device_manager.device_list) > 1:
                        from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                        for _, _mod in block.named_modules():
                            if len(list(_mod.children())) != 0 or not hasattr(_mod, "tuning_device"):
                                continue
                            add_hook_to_module(_mod, AlignDevicesHook(_mod.tuning_device, io_same_device=True), True)
                else:
                    block = block.to(device_manager.device)

                # ── Infrastructure: collect block outputs and hook stats ──
                from auto_round.algorithms.pipeline import BlockContext

                block_input_ids = input_ids
                bs = self.quantizer.batch_size * self.quantizer.infer_bs_coeff
                ctx = BlockContext(
                    model=self.model_context.model,
                    block=block,
                    block_names=[block_name],
                    block_name=block_name,
                    block_index=0,
                    io=self.quantizer.create_block_io(input_ids, input_others, None, block),
                    bs=bs,
                    device=device_manager.device,
                    is_mllm=self.model_context.is_mllm,
                    is_diffusion=self.model_context.is_diffusion,
                )
                with ExitStack() as fwd_stack:
                    self.pipeline.enter_block_forward_hooks(ctx, fwd_stack)
                    input_ids = ctx.collect_reference(fwd_stack)

                if len(device_manager.device_list) > 1:
                    accelerate.hooks.remove_hook_from_submodules(block)

                if self.compress_context.low_gpu_mem_usage:
                    block.to("cpu")
                    self.compress_context.clear_memory()

                # ── Pure algorithm ────────────────────────────────────────────
                ctx.io.seed_reference(fp_inputs=block_input_ids, reference_outputs=input_ids)
                self.quantizer.quantize_block(ctx)
                ctx.finish()

                # ── Infrastructure: cleanup ───────────────────────────────────
                mv_module_from_gpu(block)

                if self.compress_context.low_cpu_mem_usage and not self.compress_context.is_immediate_saving:
                    self._offloader(self.model_context.model, block_name)
                if block_name == block_names[-1]:
                    clear_memory(input_ids, device_list=device_manager.device_list)
                else:
                    clear_memory(device_list=device_manager.device_list)

                memory_monitor.log_summary()
                pbar.update(1)
        pbar.close()
        # Process remaining layers not in blocks
        # Collect names of quantizable layers not belonging to any block
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
            dtype = None
            if self.super_group_size is not None:
                dtype = torch.float32
            self.quantizer.quantize_layer_outside_block(name, dtype=dtype)
            # clear_memory(device_list=device_manager.device_list)
        # if self.compress_context.is_immediate_saving:
        #     shard_writer(self, is_finalize=True)

    def _quant_rtn_with_imatrix(self) -> None:
        logger.info("start to compute imatrix")
        self.quantizer.enable_imatrix = True

        # Dataloader resolution is owned by ``CalibrationState``.
        self._calibration_state.ensure_dataloader(self.model_context, self.seed)

        model = self.model_context.model

        # Dispatch multi-GPU model if necessary
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            dispatch_model(model, model.hf_device_map)

        try:
            if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                import accelerate

                accelerate.hooks.remove_hook_from_submodules(model)
            safe_to_cpu_(model)
            clear_memory(device_list=device_manager.device_list)
            self._quantize_via_rtn_blockwise()
        except torch.OutOfMemoryError:
            cuda_error_msg = traceback.format_exc()
            try:
                logger.error(cuda_error_msg)
                logger.warning(
                    "Fallback to CPU. "
                    "Consider enabling `low_gpu_mem_usage` or using more GPUs via `--device 0,1,2,3`."
                )
                safe_to_cpu_(model)
                clear_memory(device_list=device_manager.device_list)
                if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                    import accelerate

                    accelerate.hooks.remove_hook_from_submodules(model)

                # Fully fall back to CPU: both the compute device (single-sourced
                # from the DeviceManager) and the input cache device are switched,
                # then restored once the CPU pass completes.
                orig_device = device_manager.device
                orig_cache_device = self.compress_context.cache_device
                device_manager.device = "cpu"
                self.compress_context.cache_device = torch.device("cpu")
                self._quantize_via_rtn_blockwise()
                device_manager.device = orig_device
                self.compress_context.cache_device = orig_cache_device
            except Exception as e:
                raise
        finally:
            self.quantizer.enable_imatrix = False

    def quantize(self):
        """Quantize all modules in the model using RTN (Round-To-Nearest) strategy.

        If the target format includes GGUF with `k`, and optimized RTN is enabled,
        blockwise quantization with input caching and imatrix is used.

        Returns:
            tuple[nn.Module, Dict[str, Any]]: The quantized model and the layer configuration.
        """
        # post_init must be called OUTSIDE @torch.inference_mode() because
        # AutoScheme delta-loss selection requires autograd (backward pass).
        self.post_init()
        return self._quantize_impl()

    # Use no_grad instead of inference_mode
    # https://github.com/intel/auto-round/issues/1620
    @torch.no_grad()
    def _quantize_impl(self):

        formats = getattr(self, "formats", None) or []
        if not (any(fmt.is_gguf() for fmt in formats) or self.super_bits is not None):
            self.quantizer.quantize_embedding_layer()  # leave to gguf itself to handle

        # Release memory
        clear_memory(device_list=device_manager.device_list)

        enable_imatrix = False
        if not getattr(self, "disable_opt_rtn", True):
            formats = getattr(self, "formats", None) or []
            has_gguf_k = (
                any(fmt.is_gguf() and "k" in fmt.output_format for fmt in formats) or self.super_bits is not None
            )
            if has_gguf_k:
                enable_imatrix = True
            elif self.data_type == "int" and self.sym and self.bits < 8:
                enable_imatrix = True

        if enable_imatrix:
            self._quant_rtn_with_imatrix()
        else:
            self._quantize_via_rtn_blockwise()

        convert_module_to_hp_if_necessary(
            self.model_context.model,
            self.model_context.amp_dtype,
            device_manager.device,
        )
        if self.compress_context.low_cpu_mem_usage:
            self._offloader.reload(self.model_context.model)
        if self.compress_context.is_immediate_saving:
            self.shard_writer.write(is_finalize=True)

        self.model_context.quantized = True
        return self.model_context.model, self.quantizer.layer_config
