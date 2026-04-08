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
from typing import Any, Union

import torch
from tqdm import tqdm

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.compressors_new.base import BaseCompressor
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import materialize_model_
from auto_round.utils import (
    check_to_quantized,
    clear_memory,
    convert_module_to_hp_if_necessary,
    flatten_list,
    get_block_names,
    get_lm_head_name,
    get_module,
    global_state,
    memory_monitor,
    mv_module_from_gpu,
    set_module,
)


class ZeroShotCompressor(BaseCompressor):
    need_calib: bool = False

    def __init__(
        self,
        config: Union[AlgConfig, list[AlgConfig]],
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform="hf",
        format=None,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        enable_alg_ext: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        **kwargs,
    ):
        super().__init__(
            config=config,
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            device_map=device_map,
            low_gpu_mem_usage=low_gpu_mem_usage,
            enable_torch_compile=enable_torch_compile,
            enable_alg_ext=enable_alg_ext,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **kwargs,
        )
        self.lr = 5e-3

    def quantize_block(
        self,
        block: torch.nn.Module,
        inputs: tuple,
        q_input: Union[torch.Tensor, dict, None] = None,
        device: Union[str, torch.device] = "cpu",
        auto_offload: bool = True,
    ):
        """Quantize a single block via RTN (public API for LLM-Compressor).

        ZeroShotCompressor does not need calibration data, so ``inputs`` and
        ``q_input`` are accepted for interface compatibility but not used for
        algorithm purposes.  The block is materialized, converted to the target
        dtype, moved to ``device``, and quantized in-place via RTN.

        Returns:
            tuple: ``(None, None)`` — RTN does not produce reference outputs.
        """
        assert not self.mllm and not self.diffusion, (
            f"Currently, {self.__class__.__name__} does not support quantize_block " "for MLLM / diffusion models."
        )

        if not self._post_init_done:
            self.post_init()

        materialize_model_(block)
        convert_module_to_hp_if_necessary(block, self.model_context.amp_dtype, device)
        block = block.to(device)

        self.quantizer.quantize_block(block)

        mv_module_from_gpu(block)
        return None, None

    # Use no_grad instead of inference_mode
    # https://github.com/intel/auto-round/issues/1620
    @torch.no_grad()
    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """

        self.post_init()

        formats = self.formats if isinstance(self.formats, list) else []
        if not (any(fmt.is_gguf() for fmt in formats) or self.super_bits is not None):
            self._quantize_embedding_layer()  # leave to gguf itself to handle

        # Release memory
        clear_memory(device_list=self.device_list)

        # By default, we go with layer-wise way if no replacement happened.
        # In RTN mode (iters == 0), force blockwise quantization to avoid
        # full-model materialization and linear CPU RAM growth.
        use_blockwise_quantization = global_state.replaced_module_count > 0
        if not use_blockwise_quantization:
            logger.info(
                "RTN mode detected (iters=0): force blockwise quantization to avoid "
                "layer-wise full-model materialization."
            )
            use_blockwise_quantization = True
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

        if use_blockwise_quantization:  # The ram usage is a little higher

            all_blocks = self.quant_block_list or get_block_names(self.model)
            pbar = tqdm(range(sum(len(block) for block in all_blocks)))
            for block_names in all_blocks:
                for block_name in block_names:
                    pbar.set_description(f"Quantizing {block_name}")
                    block = get_module(self.model, block_name)

                    # ── Infrastructure: materialize ───────────────────────────
                    materialize_model_(block)

                    # ── Pure algorithm ────────────────────────────────────────
                    self.quantizer.quantize_block(block)

                    # ── Infrastructure: shard write / device cleanup ──────────
                    if self.is_immediate_saving:
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

                    clear_memory(device_list=self.device_list)
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
                self.quantizer.quantize_layer(name)
                cnt += 1
                if cnt % 10 == 0:
                    clear_memory(device_list=self.device_list)
                    memory_monitor.log_summary()
        else:
            all_to_quantized_module_names: list[str] = [
                n for n, m in self.model.named_modules() if check_to_quantized(m)
            ]
            all_to_quantized_module_names = all_to_quantized_module_names
            materialize_model_(self.model)
            self.model.to("cpu")
            block_names_cnt = len(flatten_list(get_block_names(self.model, True)))
            clear_mem_freq = len(all_to_quantized_module_names) // block_names_cnt
            cnt = 0
            pbar = tqdm(all_to_quantized_module_names)

            for n, m in self.model.named_modules():
                if hasattr(m, "global_name") and m.global_name in all_to_quantized_module_names:
                    pbar.set_description(f"Quantizing {m.global_name}")
                    self.quantizer.quantize_layer(m.global_name)
                    cnt += 1
                    pbar.update()
                    if cnt % clear_mem_freq == 0:
                        clear_memory(device_list=self.device_list)
                        memory_monitor.log_summary()

                elif (
                    not any(m.children())
                    and len(m.state_dict()) > 0
                    and n not in tied_weights_layers
                    and self.is_immediate_saving
                ):
                    set_module(self.model, n, copy.deepcopy(m))
                    self.shard_writer.write(name=n)
                    m.to("meta")

        # Convert remaining fp8
        convert_module_to_hp_if_necessary(self.model, self.amp_dtype, self.device)
        if self.low_cpu_mem_usage:
            self._offloader.reload(self.model)
        if self.is_immediate_saving:
            self.shard_writer.write(is_finalize=True)

        self.model_context.quantized = True
        return self.model, self.layer_config
