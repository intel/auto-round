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
"""vLLM calibration strategy.

Inherits the block-input collection infrastructure from ``LLMCalibrator``
but drives calibration through the vLLM engine (``llm.generate()``) instead
of calling the inner model directly.

Why not call the model directly
--------------------------------
vLLM's attention layers read ``attn_metadata`` and ``kv_cache`` from a
*global forward context* that is only populated by the vLLM engine at
inference time.  Calling ``model(input_ids, positions)`` outside the engine
leaves the context empty, causing the attention kernel to crash.

Why ``llm.generate()`` works
-----------------------------
The vLLM engine sets up the full forward context (attention metadata,
KV-cache slot mapping, etc.) before calling the model.  Block-level hooks
installed by ``_replace_forward`` fire normally during the prefill pass,
capturing every block's input hidden-states just as in the LLM path.

Early-stop is disabled (``should_stop`` always returns ``False``) because
the engine must be allowed to complete its own forward pass; raising
``NotImplementedError`` inside a vLLM forward would crash the engine.
All block inputs are therefore collected in a single prefill pass per sample.
"""

from __future__ import annotations

import torch

from auto_round.calibration.llm import LLMCalibrator
from auto_round.calibration.register import register_calibrator
from auto_round.logger import logger


@register_calibrator("vllm")
class VLLMCalibrator(LLMCalibrator):
    """Calibrator for models loaded through the vLLM engine.

    Differences from ``LLMCalibrator``:

    * ``calib()`` drives the model via ``llm.generate()`` so that the vLLM
      engine correctly populates the attention forward-context.
    * ``should_stop()`` always returns ``False`` — the engine must be allowed
      to complete its full forward pass; all block inputs are captured in one
      prefill per sample.
    * CPU fallback on OOM is disabled and surfaces a clear error message.
    """

    # ── Early-stop policy ───────────────────────────────────────────────────

    def should_stop(self, name: str) -> bool:
        """Never early-stop inside a vLLM forward pass.

        Raising ``NotImplementedError`` inside ``llm.generate()`` would crash
        the engine.  All block inputs are captured during the single prefill
        pass; there is no need to stop early.
        """
        return False

    # ── OOM guard ───────────────────────────────────────────────────────────

    def collect(self, block_names, nsamples, layer_names=None, last_cache_name=None):
        """Block-input collection with vLLM-specific OOM guard.

        vLLM models are always resident on GPU and cannot be moved to CPU
        (paged KV-cache, CUDA kernels).  The parent's CPU-fallback path is
        disabled; OOM surfaces as a clear error instead of a silent crash
        deep inside accelerate.
        """
        try:
            return super().collect(block_names, nsamples, layer_names, last_cache_name)
        except torch.OutOfMemoryError as e:
            raise torch.OutOfMemoryError(
                "Out of memory during vLLM calibration. "
                "CPU fallback is not supported for vLLM models. "
                "Try lowering gpu_memory_utilization, using fewer nsamples, "
                "or a shorter seqlen."
            ) from e

    # ── Calibration driver ──────────────────────────────────────────────────

    @torch.no_grad()
    def calib(self, nsamples: int, bs: int) -> None:
        """Drive the model via the vLLM engine so block hooks fire.

        Feeds pre-tokenized prompts to ``llm.generate()`` with
        ``max_tokens=1`` so only a prefill pass runs.  The hooks installed by
        ``_replace_forward`` capture each block's input hidden-states during
        prefill, which is identical to what the LLM path captures.

        For static activation quantization (e.g. NVFP4, FP8_STATIC), also
        registers ``act_max`` collection hooks on the full model before each
        ``llm.generate()`` call.  The block-wise reference forward used by
        the data-driven pipeline cannot run outside the engine (vLLM attention
        needs the engine's ForwardContext), so act_max would otherwise never be
        collected and ``pack_layer`` would fail at the
        ``assert hasattr(layer, "act_max")`` check.
        """
        from vllm import SamplingParams

        from auto_round.calib_dataset import get_dataloader
        from auto_round.compressors.utils import check_need_act_calibration

        c = self.compressor
        llm = c.model_context.llm
        tokenizer = c.model_context.tokenizer

        # vLLM hidden_states are 2-D (num_tokens, hidden_dim) — there is no
        # explicit batch axis.  Pre-set batch_dim=0 to prevent the auto-detect
        # heuristic in forward_capture from misidentifying the hidden_size
        # dimension as the batch dimension and aborting with "unsupported model".
        if hasattr(c, "quantizer") and hasattr(c.quantizer, "batch_dim"):
            c.quantizer.batch_dim = 0

        # vLLM processes one sequence per generate() call.  Each call
        # contributes exactly one "sample" to the cached block inputs, so the
        # quantizer must see batch_size=1.  We temporarily override it here
        # and restore it after calib() returns.
        _orig_batch_size = None
        if hasattr(c, "quantizer") and hasattr(c.quantizer, "batch_size"):
            _orig_batch_size = c.quantizer.batch_size
            c.quantizer.batch_size = 1

        if isinstance(c.dataset, str):
            dataset_name = c.dataset.replace(" ", "")
            c.dataloader = get_dataloader(
                tokenizer,
                c.seqlen,
                dataset_name,
                c.seed,
                bs=1,  # one sequence at a time; vLLM batches internally
                nsamples=nsamples,
            )
        else:
            c.dataloader = c.dataset

        # max_tokens=1: run prefill only, skip decode loop.
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        total_cnt = 0

        # Register act_max collection hooks on the full model for static
        # activation quantization (e.g. NVFP4).  These must be active during
        # llm.generate() because the block-wise reference forward cannot run
        # outside the vLLM engine.
        act_max_handles = []
        quantizer = getattr(c, "quantizer", None)
        if quantizer is not None and hasattr(quantizer, "_register_act_max_hooks"):
            act_bits = getattr(quantizer, "act_bits", 16) or 16
            act_data_type = getattr(quantizer, "act_data_type", None)
            act_dynamic = getattr(quantizer, "act_dynamic", True)
            if check_need_act_calibration(act_dynamic, act_data_type, act_bits):
                logger.info(
                    "VLLMCalibrator: registering act_max hooks on full model "
                    "for static activation calibration (%s).",
                    act_data_type,
                )
                act_max_handles = quantizer._register_act_max_hooks(c.model_context.model)
                if act_max_handles:
                    logger.info(
                        "VLLMCalibrator: registered %d act_max hook(s).",
                        len(act_max_handles),
                    )

        try:
            for data in c.dataloader:
                if data is None:
                    continue

                # Extract input_ids tensor from various dataloader formats.
                if isinstance(data, torch.Tensor):
                    input_ids_batch = data
                elif isinstance(data, dict) or hasattr(data, "keys"):
                    input_ids_batch = data.get("input_ids")
                    if input_ids_batch is None:
                        continue
                else:
                    continue

                if input_ids_batch.shape[-1] < c.seqlen:
                    continue

                # Feed each sequence as a pre-tokenized prompt so we skip the
                # tokenize → detokenize roundtrip.
                for i in range(input_ids_batch.shape[0]):
                    token_ids = input_ids_batch[i, : c.seqlen].tolist()
                    llm.generate(
                        [{"prompt_token_ids": token_ids}],
                        sampling_params,
                        use_tqdm=False,
                    )
                    total_cnt += 1
                    if total_cnt >= nsamples:
                        break

                if total_cnt >= nsamples:
                    break
        finally:
            for h in act_max_handles:
                h.remove()

        if total_cnt == 0:
            logger.error(
                "No calibration data was captured. "
                f"Provide data with sequence length >= {c.seqlen} "
                "or decrease seqlen."
            )
            exit(-1)
        elif total_cnt < nsamples:
            logger.warning_once(
                f"Insufficient calibration samples: target {nsamples}, "
                f"captured {total_cnt}.  Quantization accuracy may be reduced."
            )

        # NOTE: Do NOT restore batch_size here.
        # vLLM block I/O uses a flattened [n_tokens, hidden] format (no explicit
        # batch dimension).  With batch_size > 1, _append_output would call
        # torch.split(output, 1, dim=0) which splits individual tokens rather
        # than samples, producing wrong reference-output shapes for the
        # downstream collect_reference + quantize_block optimization loop.
        # Keeping batch_size=1 for the full vLLM-loading path is correct.
        # Reference: Fix 13 — do not restore _orig_batch_size after calib().
