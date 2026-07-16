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
"""LLM (text-only) calibration strategy.

Implements ``try_cache_inter_data_gpucpu`` / ``cache_inter_data`` /
``calib`` for the plain-text path.  Compressor state is accessed via
``self.compressor.X``.
"""

import traceback
from functools import partial
from typing import Callable

import accelerate
import torch
from accelerate.big_modeling import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory, get_max_memory

from auto_round import envs
from auto_round.calibration.base import Calibrator
from auto_round.calibration.register import register_calibrator
from auto_round.calibration.utils import _infer_last_cache_name
from auto_round.compressors.utils import init_cache, check_skippable_keywords, reset_params
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import materialize_model_
from auto_round.utils import (
    check_seqlen_compatible,
    clear_memory,
    flatten_list,
    hook_ngram_embeddings_on_cpu,
    is_quantized_input_module,
    mv_module_from_gpu,
    to_device,
    to_dtype, SUPPORTED_LAYER_TYPES,
)
from auto_round.utils.device import parse_available_devices
from auto_round.utils.device_manager import device_manager


@register_calibrator("llm")
class LLMCalibrator(Calibrator):
    """Calibrator for plain text / LLM models."""

    # ── Public API ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect(self, block_names, nsamples, layer_names=None, last_cache_name=None):
        """Attempts to cache intermediate data on GPU; on OOM, falls back to CPU.

        Verbatim port of the legacy ``DataDrivenCompressor.try_cache_inter_data_gpucpu``.
        """
        if is_quantized_input_module(self.model):#e.g. FP8 model
            layer_names = []
        if layer_names is None:
            layer_names = []

        block_names = flatten_list(block_names)
        self.blocks_requiring_input_ids = [data if isinstance(data, str) else data[0] for data in block_names]

        calibrate_on_cpu = False
        cannot_calibrate_on_cpu = False
        if self.low_gpu_mem_usage or (
            len(block_names) == 1
            and len(layer_names) == 0
            and (last_cache_name is None or last_cache_name in block_names)
        ):
            # low_gpu_mem_usage or calibrate only the embedding layer (also fast on CPU)
            calibrate_on_cpu = True
            try:
                all_inputs = self.cache_inter_data(
                    block_names, nsamples, layer_names=[], last_cache_name=last_cache_name
                )
            except NotImplementedError as error:
                error_msg = str(error)
                if "flash_attn::" in error_msg and "CPU" in error_msg:
                    cannot_calibrate_on_cpu = True
                else:
                    raise error

        if not calibrate_on_cpu or cannot_calibrate_on_cpu:
            try:
                if any(p.device.type == "meta" for p in self.model.parameters()):
                    materialize_model_(self.model)

                if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                    dispatch_model(
                        self.model, device_map=self.model.hf_device_map
                    )
                else:
                    if str(self.model.device) == "cpu" and (not device_manager.device.startswith("hpu")):
                        no_split_modules = list(getattr(self.model, "_no_split_modules", []))
                        devices = parse_available_devices(device_manager.device_map)

                        max_memory = get_max_memory()
                        new_max_memory = {}
                        if "cpu" not in devices:
                            devices.append("cpu")
                        for device in devices:
                            if ":" in device:
                                device = int(device.split(":")[-1])
                            elif device == "cpu":
                                device = "cpu"
                            elif isinstance(device, str):
                                device = 0
                            else:
                                raise ValueError(
                                    f"Unsupported device {device} in device_map: {device_manager.device_map}"
                                )
                            if device not in max_memory:
                                continue
                            new_max_memory[device] = max_memory[device] * 0.9

                        requested_non_cpu = any((d != "cpu") for d in devices)
                        has_non_cpu_memory = any((k != "cpu") for k in new_max_memory)
                        if requested_non_cpu and not has_non_cpu_memory:
                            raise torch.OutOfMemoryError(
                                "No non-CPU device available in accelerate's reported memory. "
                                "Falling back to CPU caching."
                            )

                        has_ngram_embeddings, raw_ngram_embeddings = hook_ngram_embeddings_on_cpu(self.model)
                        new_max_memory = get_balanced_memory(
                            self.model,
                            max_memory=new_max_memory,
                            no_split_module_classes=no_split_modules,
                        )
                        if hasattr(self.model, "tie_weights"):
                            self.model.tie_weights()
                        device_map = infer_auto_device_map(
                            self.model,
                            max_memory=new_max_memory,
                            no_split_module_classes=no_split_modules,
                        )
                        if len(devices) > 1 and "cpu" in device_map.values():
                            logger.warning(
                                "Some layers are offloaded to cpu, which may severely impact calibration speed."
                                " Please consider using more cards."
                            )

                        try:
                            dispatch_model(self.model, device_map=device_map)
                            if has_ngram_embeddings:
                                self.model.model.ngram_embeddings = raw_ngram_embeddings
                        except ValueError as e:
                            if "offload_dir" in e.__str__():
                                logger.warning(
                                    f"Due to insufficient resources, disk is used to store the model."
                                    f" `offload_dir={envs.AR_WORK_SPACE}`"
                                )
                                dispatch_model(
                                    self.model,
                                    device_map=device_map,
                                    offload_dir=envs.AR_WORK_SPACE,
                                )
                            else:
                                raise
                    else:
                        self.model.to(device_manager.device)

                all_inputs = self.cache_inter_data(
                    block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name
                )
                if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                    accelerate.hooks.remove_hook_from_submodules(self.model)

            except torch.OutOfMemoryError as e:
                if cannot_calibrate_on_cpu:
                    raise e
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.info("switch to cpu to cache block inputs")

                    if len(block_names)>1 or len(layer_names)>1:
                        logger.warning(
                            "we recommend using more GPUs in calibration."
                            " Otherwise, some layers may fall back to `rtn` mode, which can affect accuracy."
                        )
                    accelerate.hooks.remove_hook_from_submodules(self.model)
                    mv_module_from_gpu(self.model)
                    clear_memory()
                    # On cpu, we use rtn mode for layers in layer_names (post v0.51).
                    all_inputs = self.cache_inter_data(
                        block_names, nsamples, layer_names=[], last_cache_name=last_cache_name
                    )
                except Exception:
                    logger.error(cuda_error_msg)
                    raise
        return all_inputs

    @torch.no_grad()
    def cache_inter_data(self, block_names, nsamples, layer_names=None, last_cache_name=None):
        """Replace forward, run :meth:`calib`, return cached ``inputs``.

        Verbatim port of the legacy ``DataDrivenCompressor.cache_inter_data``.
        """

        if layer_names is None:
            layer_names = []

        #TODO have a check wenhuach need to pass attetnion_mask
        # if hasattr(c, "quantizer") and hasattr(c.quantizer, "attention_mask"):
        #     c.quantizer.attention_mask = []

        self.inputs = {}
        block_names = flatten_list(block_names)
        self.to_cached_layers = block_names + layer_names

        # tmp_dtype = None  # TODO delete this as most model is not fp32 now
        # # Bug if block name is not the first block
        # if (len(block_names) > 1 or len(layer_names) > 0) and self.low_gpu_mem_usage:
        #     tmp_dtype = self.model.dtype
        #     if c.model_context.amp:
        #         if c.model_context.model.dtype != c.model_context.model.dtype:
        #             c.model_context.model = c.model_context.model.to(torch.bfloat16)
        #     else:
        #         c.model_context.model = c.model_context.model.to(torch.float32)  # model on cpu

        self.last_cache_name = _infer_last_cache_name(block_names, layer_names, last_cache_name)
        self._cache_target_set = set(self.to_cached_layers)
        self._cache_seen_targets = set()
        calib_bs = self.batch_size
        self.hook_handles = []

        self.replace_forward_with_hooks()
        try:
            # Dispatch via the Compressor so that MLLMMixin / DiffusionMixin overrides
            # of ``calib`` are honoured; if neither override applies, the Compressor's
            # ``calib`` thin-wrapper routes back into ``self.calib`` below.
            self.calib(nsamples, calib_bs)
        finally:
            # Use finally to recover_forward and delattr in case calib raises
            # NotImplementedError, e.g. flash_attn on CPU.
            self.recover_forward()
            for attr in ("last_cache_name", "_cache_target_set", "_cache_seen_targets", "to_cached_layers"):
                if hasattr(self, attr):
                    delattr(self, attr)
            # Release calibration dataloader to free tokenized sample tensors.
            if hasattr(self, "dataloader"):
                del self.dataloader
        res = self.inputs
        # if tmp_dtype is not None:
        #     c.model_context.model.to(tmp_dtype)

        return res

    def recover_forward(self, restore_positional_wrapper=None):
        """Recovers the forward function.

        Args:
            restore_positional_wrapper: If True, restores forward to the wrapped version
                (needed for LLM calibration where positional wrapper is used during quantization).
                If False, restores to the true original forward (needed for diffusion).
                If None (default), auto-detects: uses False for diffusion models.
        """

        # Auto-detect for diffusion: when _true_orig_forward is present (set by
        # CalibCompressor._replace_forward), we are in new-arch diffusion mode where
        # the positional wrapper must be fully removed after caching.
        if restore_positional_wrapper is None:
            restore_positional_wrapper = not getattr(self, "_has_true_orig_forward_set", False)
            if not restore_positional_wrapper:
                logger.debug("recover_forward: auto-detected diffusion mode, stripping positional wrapper")

        for n, m in self.model.named_modules():
            if hasattr(m, "orig_forward"):
                true_orig = getattr(m, "_true_orig_forward", m.orig_forward)
                if restore_positional_wrapper:
                    # Restore orig_forward so that any wrapper (e.g. from
                    # wrap_block_forward_positional_to_kwargs) can still access it.
                    # The wrapper holds a closure reference to orig_forward.
                    m.forward = getattr(m, "_wrapped_forward_before_replace", m.orig_forward)
                    m.orig_forward = true_orig
                else:
                    # Full recovery: restore the true original forward.  Used for diffusion
                    # where the positional wrapper must be fully removed after caching.
                    m.forward = true_orig
                    # Keep _true_orig_forward so the wrapped forward's base_hook can
                    # still call it during quantization tuning.
                    m._true_orig_forward = true_orig
                    delattr(m, "orig_forward")
                    if hasattr(m, "_wrapped_forward_before_replace"):
                        delattr(m, "_wrapped_forward_before_replace")
        for hook_handle in self.hook_handles:
            hook_handle.remove()
        self.hook_handles = []

    @torch.no_grad()
    def calib(self, nsamples: int, bs: int) -> None:
        """Drive the model with text data so block hooks fire.

        Verbatim port of the legacy ``DataDrivenCompressor.calib`` (LLM path only).
        """
        from auto_round.calib_dataset import get_dataloader

        need_attention_mask = True
        if isinstance(self.dataset, str):
            need_attention_mask = False  # all supported datasets do not use pad
            dataset = self.dataset.replace(" ", "")  # remove all whitespaces

            # slow here
            self.dataloader = get_dataloader(
                self.tokenizer,
                self.seqlen,
                dataset,
                self.seed,
                bs,
                nsamples,
            )
        else:
            self.dataloader = self.dataset
        total_cnt = 0
        if self.dataloader.__class__.__name__ == "BatchEncoding":
            self.dataloader = [self.dataloader.data]

        for data in self.dataloader:
            if data.__class__.__name__ == "BatchEncoding":
                data = data.data
            if data is None:
                continue
            if isinstance(data, torch.Tensor):
                input_ids = data.to(self.model.device)
                data_new = input_ids
            elif isinstance(data, str):
                if self.tokenizer is None:
                    logger.error("please provide tokenizer for string input")
                    exit(-1)
                data = self.tokenizer(data, truncation=True, max_length=self.seqlen, return_tensors="pt").data
                data_new = {}
                for key in data.keys():
                    data_new[key] = data[key].to(self.model.device)
                input_ids = data_new["input_ids"]
            elif isinstance(data, tuple) or isinstance(data, list):
                data_new = to_device(data, self.model.device)
                input_ids = data_new[0]
            else:
                data_new = {}
                for key in data.keys():
                    data_new[key] = to_device(data[key], self.model.device)
                    if key == "images":
                        data_new[key] = to_dtype(data_new[key], self.model.dtype)
                input_ids = data_new["input_ids"]
            if input_ids.shape[-1] < self.seqlen:
                continue
            if need_attention_mask:
                if (
                    isinstance(data_new, dict)
                    and "attention_mask" in data_new
                    and data_new["attention_mask"] is not None
                ):
                    new_attention_mask = data_new["attention_mask"]
                elif (
                    self.tokenizer is not None
                    and hasattr(self.tokenizer, "pad_token")
                    and self.tokenizer.pad_token is not None
                ):
                    new_attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.long)
                else:
                    # Default all ones
                    new_attention_mask = torch.ones_like(input_ids, dtype=torch.long)

                    # For each sample, check if there are trailing repeated tokens; if
                    # so, set the mask of the last token to 0.
                    batch_size, seq_len = input_ids.shape
                    for i in range(batch_size):
                        last_token = input_ids[i, -1]
                        j = seq_len - 2
                        repeated = False
                        while j >= 0 and input_ids[i, j] == last_token:
                            repeated = True
                            new_attention_mask[i, j] = 0
                            j -= 1
                        if repeated:
                            new_attention_mask[i, -1] = 0

                # Workaround: some models treat an all-1 attention mask as equivalent to
                # None and will internally replace it with None for block inputs, which
                # can cause tensor concatenation / shape-mismatch issues downstream.
                # Force the last token in each sequence to be masked out so the mask is
                # never "all ones".
                new_attention_mask[:, -1] = 0

                #TODO wenhuach pass attention_mask to alg
                # if not hasattr(c.quantizer, "attention_mask"):
                #     c.quantizer.attention_mask = []
                # c.quantizer.attention_mask.extend(list(torch.split(new_attention_mask, 1, dim=0)))
            else:
                new_attention_mask = None
            try:
                kwargs = {"use_cache": False}
                if new_attention_mask is not None and not (isinstance(data_new, dict) and "attention_mask" in data_new):
                    kwargs["attention_mask"] = new_attention_mask

                if isinstance(data_new, torch.Tensor):
                    self.model(data_new, **kwargs)
                elif isinstance(data_new, tuple) or isinstance(data_new, list):
                    self.model(*data_new, **kwargs)
                else:
                    self.model(**data_new, **kwargs)
            except NotImplementedError as error:
                error_msg = str(error)
                # Re-raise to fallback to CUDA when flash_attn does not support CPU.
                if "flash_attn::" in error_msg and "CPU" in error_msg:
                    raise NotImplementedError(
                        "Could not run 'flash_attn::_flash_attn_varlen_forward'"
                        " with arguments from the 'CPU' backend."
                    )
                else:
                    pass
            except RuntimeError as error:
                error_msg = str(error)
                if "The expanded size of the tensor" in str(error_msg) and "must match the existing size" in error_msg:
                    check_seqlen_compatible(c.seqlen, c.model_context.tokenizer, c.model)
                logger.warning(
                    "When quantization encounters tensor shape mismatch error, "
                    "you can try to avoid it with batch_size=1"
                )
                raise error
            except Exception as error:
                raise error

            total_cnt += input_ids.shape[0] if len(input_ids.shape) > 1 else 1
            if total_cnt >= nsamples:
                break
        if total_cnt == 0:
            logger.error(
                f"no data has been cached, please provide more data with sequence length "
                f">={self.seqlen} in the dataset or decease the sequence length"
            )
            exit(-1)
        elif total_cnt < nsamples:
            logger.warning_once(
                f"An insufficient number of samples likely reduces the accuracy of the quantized model. "
                f"Target samples count is {nsamples}, while valid samples count is {total_cnt}"
            )

    def make_block_forward_func(self, name: str) -> Callable:
        """Build a ``forward`` replacement that captures inputs for *block* ``name``.

        Mirrors the legacy ``DataDrivenCompressor._get_block_forward_func`` exactly.
        The returned function expects to be bound as ``module.forward = partial(fn, module)``.
        """

        def post_process_cache_data(batch_size, data, data_name):
            new_data = data
            if data_name in self.shared_cache_keys:
                return None
            if batch_size <= 1:
                return new_data
            if "alibi" in data_name:
                if isinstance(data, torch.Tensor):
                    alibi = data
                    alibi = alibi.reshape(batch_size, -1, alibi.shape[1], alibi.shape[2])
                    new_data = alibi
            return new_data

        def forward_capture(m, hidden_states=None, *positional_inputs, **kwargs):
            if name not in self.inputs:
                self.inputs[name] = {}
                init_cache(positional_inputs, self.inputs[name])

            if self.batch_dim is None:
                self.batch_dim = 0
                if hidden_states is not None and self.batch_size > 1:
                    if hidden_states.shape[0] > self.batch_size:
                        self.batch_dim = 1 # TODO wenhuach this one should pass to algorithm
                        if (
                                len(hidden_states.shape) > 1
                                and hidden_states.shape[1] > self.batch_size
                        ):
                            logger.error(
                                "this model has not been supported, "
                                "please raise an issue in https://github.com/intel/auto-round/issues"
                                " or try to set the `batch_size` to 1 and "
                                "`gradient_accumulate_steps` to your current batch size."
                            )
                            exit(-1)

            if hidden_states is not None:
                kwargs["hidden_states"] = hidden_states

            for key in kwargs.keys():
                if isinstance(kwargs[key], torch.Tensor) or isinstance(kwargs[key], list) or isinstance(kwargs[key],
                                                                                                        tuple):
                    if (
                            self.has_variable_block_shape
                            and name not in self.blocks_requiring_input_ids
                            and key == "hidden_states"
                    ):
                        continue
                    if key not in self.inputs[name].keys():  # initialization
                        data = to_device(kwargs[key], device=torch.device("cpu"))
                        if data is None or key in self.shared_cache_keys:
                            self.inputs[name][key] = data
                            continue
                        if self.batch_size <= 1:
                            self.inputs[name][key] = [data]
                        else:
                            data = post_process_cache_data(self.batch_size, data, key)
                            if isinstance(data, torch.Tensor):
                                self.inputs[name][key] = list(
                                    torch.split(data, 1, dim=self.batch_dim))
                            else:
                                self.inputs[name][key] = [data]
                    else:  # append cache inputs
                        new_data = post_process_cache_data(self.batch_size, kwargs[key], key)
                        if new_data is None:  # shareable args or NoneType
                            if key in self.shared_cache_keys:
                                # Shared keys are normally the same across samples.  However
                                # in VLM visual encoders (e.g. Qwen2-VL) ``position_embeddings``
                                # varies per image because each image has a different patch count.
                                # Upgrade from shared (raw value) to per-sample list storage so
                                # each sample gets its own positional embeddings.
                                raw_new = to_device(kwargs[key], device=torch.device("cpu"))
                                stored = self.inputs[name].get(key)
                                if isinstance(stored, list):
                                    stored.append(raw_new)
                                elif stored is not None:
                                    self.inputs[name][key] = [stored, raw_new]
                            continue
                        new_data = to_device(new_data, device=torch.device("cpu"))
                        # Guard against None-initialized kwargs that arrive as tensors on later samples (#1950).
                        if self.inputs[name][key] is None:
                            self.inputs[name][key] = []
                        if self.batch_size <= 1:
                            self.inputs[name][key].append(new_data)
                        else:
                            if isinstance(new_data, torch.Tensor):
                                self.inputs[name][key].extend(
                                    list(torch.split(new_data, 1, dim=self.batch_dim))
                                )
                            else:
                                self.inputs[name][key].append(new_data)
                elif isinstance(kwargs[key], (str, bool, type(None))):
                    if key not in self.inputs[name].keys():
                        self.inputs[name][key] = kwargs[key]
                else:
                    # Parameters not to be cached
                    if check_skippable_keywords(key):
                        logger.warning_once(
                            f"Please note that '{key}' key" " is not currently used in quantization fine-tuning."
                        )
            reset_params(self.inputs[name])

            if self._should_stop_cache_forward(name):
                raise NotImplementedError
            else:
                if hidden_states is not None:
                    kwargs.pop("hidden_states", None)
                    if positional_inputs:
                        return m.orig_forward(hidden_states=hidden_states, *positional_inputs, **kwargs)
                    else:
                        return m.orig_forward(hidden_states, **kwargs)
                else:
                    # Currently only for Llama-3.2-Vision-Instruct Series
                    return m.orig_forward(*positional_inputs, **kwargs)

        # Apply positional-to-kwargs conversion so positional_inputs get their proper parameter names.
        from auto_round.utils.model import wrap_block_forward_positional_to_kwargs

        return wrap_block_forward_positional_to_kwargs(forward_capture)

    def make_layer_cache_hook(self, name: str) -> Callable:
        """Build a forward-hook that captures inputs for *layer* ``name``.

        Mirrors the legacy ``DataDrivenCompressor._get_cache_data_hook_for_layer`` exactly.
        """

        def cache_input_hook(module, inputs, outputs):
            input = inputs
            if isinstance(inputs, tuple) or isinstance(input, list):
                input = inputs[0]
            if name in self.inputs:
                self.inputs[name].extend(list(torch.split(input.to("cpu"), 1, dim=0)))
            else:
                self.inputs[name] = list(torch.split(input.to("cpu"), 1, dim=0))

            if self._should_stop_cache_forward(name):
                raise NotImplementedError

        return cache_input_hook

    def replace_forward_with_hooks(self) -> None:
        """Install block-forward replacements and layer hooks via ``model_context.replace_forward``.

        Mirrors the legacy ``DataDrivenCompressor._replace_forward`` exactly. The
        ``state`` is expected to expose ``to_cached_layers`` / ``hook_handles`` /
        ``model_context`` and the two factory methods on its class
        (``_get_block_forward_func`` / ``_get_cache_data_hook_for_layer``) so
        that subclass overrides (e.g. ``DiffusionMixin``) still take effect.
        """

        def register_hook(n, m):
            if n in self.to_cached_layers and type(m) not in SUPPORTED_LAYER_TYPES:  # block
                m.orig_forward = m.forward
                m.forward = partial(self._get_block_forward_func(n), m)
            elif n in self.to_cached_layers:  # linear / conv1d layer
                hook_func = self.make_layer_cache_hook(n)
                hook_handle = m.register_forward_hook(hook_func)
                self.hook_handles.append(hook_handle)

        for n, m in self.model.named_modules():
            register_hook(n, m)

    def should_stop_cache_forward(self, name: str) -> bool:
        """Default early-stop policy for block input collection.

        Mirrors the legacy ``DataDrivenCompressor._should_stop_cache_forward`` exactly.
        Subclasses (e.g. ``DiffusionMixin``) override the method on the Compressor
        class to always return ``False``; this helper is only used by the default
        LLM path.
        """
        if name == self.last_cache_name:
            return True

        if self.last_cache_name is not None:
            return False

        if not hasattr(self, "_cache_target_set") or not hasattr(self, "_cache_seen_targets"):
            return False

        if name in self._cache_target_set:
            self._cache_seen_targets.add(name)

        if not self._cache_target_set.issubset(self._cache_seen_targets):
            return False

        # Lock the last cache name after the first full forward pass.
        self.last_cache_name = name
        return True

