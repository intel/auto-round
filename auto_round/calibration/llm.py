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

import accelerate
import torch
from accelerate.big_modeling import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory, get_max_memory

from auto_round import envs
from auto_round.calibration.base import Calibrator
from auto_round.calibration.register import register_calibrator
from auto_round.calibration.utils import _infer_last_cache_name
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
    to_dtype,
)
from auto_round.utils.device import parse_available_devices


@register_calibrator("llm")
class LLMCalibrator(Calibrator):
    """Calibrator for plain text / LLM models."""

    # ── Public API ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect(self, block_names, nsamples, layer_names=None, last_cache_name=None):
        """Attempts to cache intermediate data on GPU; on OOM, falls back to CPU.

        Verbatim port of the legacy ``DataDrivenCompressor.try_cache_inter_data_gpucpu``.
        """
        c = self.compressor
        if is_quantized_input_module(c.model_context.model):
            layer_names = []
        if layer_names is None:
            layer_names = []

        block_names = flatten_list(block_names)
        c.blocks_requiring_input_ids = [data if isinstance(data, str) else data[0] for data in block_names]

        calibrate_on_cpu = False
        cannot_calibrate_on_cpu = False
        if c.compress_context.low_gpu_mem_usage or (
            len(block_names) == 1
            and len(layer_names) == 0
            and not c.quantizer.has_qlayer_outside_block
            and (last_cache_name is None or last_cache_name in block_names)
            and not getattr(c, "mllm", False)
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
                if any(p.device.type == "meta" for p in c.model_context.model.parameters()):
                    materialize_model_(c.model_context.model)

                if hasattr(c.model_context.model, "hf_device_map") and len(c.model_context.model.hf_device_map) > 1:
                    c.model_context.model = dispatch_model(
                        c.model_context.model, device_map=c.model_context.model.hf_device_map
                    )
                else:
                    if str(c.model_context.model.device) == "cpu" and (not c.compress_context.device.startswith("hpu")):
                        no_split_modules = list(getattr(c.model_context.model, "_no_split_modules", []))
                        devices = parse_available_devices(c.compress_context.device_map)

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
                                    f"Unsupported device {device} in device_map: {c.compress_context.device_map}"
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

                        has_ngram_embeddings, raw_ngram_embeddings = hook_ngram_embeddings_on_cpu(c.model_context.model)
                        new_max_memory = get_balanced_memory(
                            c.model_context.model,
                            max_memory=new_max_memory,
                            no_split_module_classes=no_split_modules,
                        )
                        if hasattr(c.model_context.model, "tie_weights"):
                            c.model_context.model.tie_weights()
                        device_map = infer_auto_device_map(
                            c.model_context.model,
                            max_memory=new_max_memory,
                            no_split_module_classes=no_split_modules,
                        )
                        if len(devices) > 1 and "cpu" in device_map.values():
                            logger.warning(
                                "Some layers are offloaded to cpu, which may severely impact calibration speed."
                                " Please consider using more cards."
                            )

                        try:
                            c.model_context.model = dispatch_model(c.model_context.model, device_map=device_map)
                            if has_ngram_embeddings:
                                c.model_context.model.model.ngram_embeddings = raw_ngram_embeddings
                        except ValueError as e:
                            if "offload_dir" in e.__str__():
                                logger.warning(
                                    f"Due to insufficient resources, disk is used to store the model."
                                    f" `offload_dir={envs.AR_WORK_SPACE}`"
                                )
                                c.model_context.model = dispatch_model(
                                    c.model_context.model,
                                    device_map=device_map,
                                    offload_dir=envs.AR_WORK_SPACE,
                                )
                            else:
                                raise
                    else:
                        c.model_context.model = c.model_context.model.to(c.compress_context.device)

                all_inputs = self.cache_inter_data(
                    block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name
                )
                if hasattr(c.model_context.model, "hf_device_map") and len(c.model_context.model.hf_device_map) > 1:
                    accelerate.hooks.remove_hook_from_submodules(c.model_context.model)

            except torch.OutOfMemoryError as e:
                if cannot_calibrate_on_cpu:
                    raise e
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.info("switch to cpu to cache block inputs")
                    c.compress_context.cache_device = torch.device("cpu")
                    if c.quantizer.has_qlayer_outside_block or c.__class__.__name__ == "AutoRoundMLLM":
                        logger.warning(
                            "we recommend using more GPUs in calibration."
                            " Otherwise, some layers may fall back to `rtn` mode, which can affect accuracy."
                        )
                    accelerate.hooks.remove_hook_from_submodules(c.model_context.model)
                    c.model_context.model = mv_module_from_gpu(c.model_context.model)
                    clear_memory(device_list=c.compress_context.device_list)
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
        c = self.compressor
        if layer_names is None:
            layer_names = []

        if not c._post_init_done:
            c.post_init()

        if hasattr(c, "quantizer") and hasattr(c.quantizer, "attention_mask"):
            c.quantizer.attention_mask = []

        c.inputs = {}
        block_names = flatten_list(block_names)
        c.to_cached_layers = block_names + layer_names

        tmp_dtype = None  # TODO delete this as most model is not fp32 now
        ## have bug if block name is not the first block
        if (len(block_names) > 1 or len(layer_names) > 0) and c.compress_context.low_gpu_mem_usage:
            tmp_dtype = c.model_context.model.dtype
            if c.model_context.amp:
                if c.model_context.model.dtype != c.model_context.model.dtype:
                    c.model_context.model = c.model_context.model.to(torch.bfloat16)
            else:
                c.model_context.model = c.model_context.model.to(torch.float32)  # model on cpu

        c.last_cache_name = _infer_last_cache_name(block_names, layer_names, last_cache_name)
        c._cache_target_set = set(c.to_cached_layers)
        c._cache_seen_targets = set()
        calib_bs = c.quantizer.batch_size
        c.hook_handles = []
        c._replace_forward()
        try:
            # Dispatch via the Compressor so that MLLMMixin / DiffusionMixin overrides
            # of ``calib`` are honoured; if neither override applies, the Compressor's
            # ``calib`` thin-wrapper routes back into ``self.calib`` below.
            c.calib(nsamples, calib_bs)
        finally:
            # Use finally to recover_forward and delattr in case calib raises
            # NotImplementedError, e.g. flash_attn on CPU.
            c.model_context.recover_forward()
            for attr in ("last_cache_name", "_cache_target_set", "_cache_seen_targets", "to_cached_layers"):
                if hasattr(c, attr):
                    delattr(c, attr)
            # Release calibration dataloader to free tokenized sample tensors.
            if hasattr(c, "dataloader"):
                del c.dataloader
        res = c.inputs
        if tmp_dtype is not None:
            c.model_context.model = c.model_context.model.to(tmp_dtype)

        return res

    @torch.no_grad()
    def calib(self, nsamples: int, bs: int) -> None:
        """Drive the model with text data so block hooks fire.

        Verbatim port of the legacy ``DataDrivenCompressor.calib`` (LLM path only).
        """
        from auto_round.calib_dataset import get_dataloader

        c = self.compressor
        need_attention_mask = True
        if isinstance(c.dataset, str):
            need_attention_mask = False  # all supported datasets do not use pad
            dataset = c.dataset.replace(" ", "")  # remove all whitespaces

            # slow here
            c.dataloader = get_dataloader(
                c.model_context.tokenizer,
                c.seqlen,
                dataset,
                c.seed,
                bs,
                c.nsamples,
            )
        else:
            c.dataloader = c.dataset
        total_cnt = 0
        if c.dataloader.__class__.__name__ == "BatchEncoding":
            c.dataloader = [c.dataloader.data]

        for data in c.dataloader:
            if data.__class__.__name__ == "BatchEncoding":
                data = data.data
            if data is None:
                continue
            if isinstance(data, torch.Tensor):
                input_ids = data.to(c.model.device)
                data_new = input_ids
            elif isinstance(data, str):
                if c.model_context.tokenizer is None:
                    logger.error("please provide tokenizer for string input")
                    exit(-1)
                data = c.model_context.tokenizer(data, truncation=True, max_length=c.seqlen, return_tensors="pt").data
                data_new = {}
                for key in data.keys():
                    data_new[key] = data[key].to(c.model.device)
                input_ids = data_new["input_ids"]
            elif isinstance(data, tuple) or isinstance(data, list):
                data_new = to_device(data, c.model.device)
                input_ids = data_new[0]
            else:
                data_new = {}
                for key in data.keys():
                    data_new[key] = to_device(data[key], c.model.device)
                    if key == "images":
                        data_new[key] = to_dtype(data_new[key], c.model.dtype)
                input_ids = data_new["input_ids"]
            if input_ids.shape[-1] < c.seqlen:
                continue
            if need_attention_mask:
                if (
                    isinstance(data_new, dict)
                    and "attention_mask" in data_new
                    and data_new["attention_mask"] is not None
                ):
                    new_attention_mask = data_new["attention_mask"]
                elif (
                    c.model_context.tokenizer is not None
                    and hasattr(c.model_context.tokenizer, "pad_token")
                    and c.model_context.tokenizer.pad_token is not None
                ):
                    new_attention_mask = (input_ids != c.model_context.tokenizer.pad_token_id).to(torch.long)
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

                if not hasattr(c.quantizer, "attention_mask"):
                    c.quantizer.attention_mask = []
                c.quantizer.attention_mask.extend(list(torch.split(new_attention_mask, 1, dim=0)))
            else:
                new_attention_mask = None
            try:
                kwargs = {"use_cache": False}
                if new_attention_mask is not None and not (isinstance(data_new, dict) and "attention_mask" in data_new):
                    kwargs["attention_mask"] = new_attention_mask

                if isinstance(data_new, torch.Tensor):
                    c.model(data_new, **kwargs)
                elif isinstance(data_new, tuple) or isinstance(data_new, list):
                    c.model(*data_new, **kwargs)
                else:
                    c.model(**data_new, **kwargs)
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
                f">={c.seqlen} in the dataset or decease the sequence length"
            )
            exit(-1)
        elif total_cnt < nsamples:
            logger.warning_once(
                f"An insufficient number of samples likely reduces the accuracy of the quantized model. "
                f"Target samples count is {nsamples}, while valid samples count is {total_cnt}"
            )
