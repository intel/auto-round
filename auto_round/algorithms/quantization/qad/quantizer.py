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
"""QAD quantizer: whole-network joint distillation from an offline teacher cache."""

from __future__ import annotations

import math
import os

import torch

from auto_round import envs
from auto_round.algorithms.quantization.qad.config import QADConfig
from auto_round.algorithms.quantization.qad.teacher_cache import TeacherTopKCache
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.logger import logger
from auto_round.utils import check_is_cpu, clear_memory, htcore, is_hpex_available
from auto_round.utils.device_manager import device_manager

try:  # CUDA-only fast path; falls back to the chunked PyTorch implementation.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton is optional
    _HAS_TRITON = False


def _streaming_logsumexp(s: torch.Tensor, tau: float, vocab_chunk: int) -> torch.Tensor:
    """Numerically-stable ``logsumexp(s/τ)`` over dim -1, streamed in vocab chunks.

    PyTorch fallback used when Triton / CUDA is unavailable. Keeps a running max
    and running sum so the largest temporary is ``[M, chunk]`` instead of the
    full ``[M, vocab]`` ``exp`` (the online-softmax trick that lets Unsloth's
    chunked kernel handle 256K-vocab models).
    """
    M, V = s.shape
    running_max = torch.full((M,), float("-inf"), device=s.device, dtype=torch.float32)
    running_sum = torch.zeros(M, device=s.device, dtype=torch.float32)
    for c0 in range(0, V, vocab_chunk):
        c1 = min(c0 + vocab_chunk, V)
        sc = s[:, c0:c1].float() / tau  # [M, cw]
        chunk_max = sc.max(dim=-1).values  # [M]
        new_max = torch.maximum(running_max, chunk_max)
        running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(sc - new_max.unsqueeze(1)).sum(dim=-1)
        running_max = new_max
    return running_max + torch.log(running_sum)


_QAD_TRITON_BLOCK = 8192  # register-friendly vocab tile for the Triton kernels


if _HAS_TRITON:

    @triton.jit
    def _qad_logsumexp_kernel(
        logits_ptr,
        logz_ptr,
        stride_b,
        stride_t,
        S1,
        VOCAB,
        INV_TAU,
        BLOCK: tl.constexpr,
    ):
        """One program per shifted row: online ``logsumexp(s/τ)`` over the vocab.

        Row ``r`` maps to token ``(b, t) = (r // S1, r % S1)`` so the kernel reads
        straight from the ``[B, seq, vocab]`` logits without a reshape/copy of the
        shifted slice. The ``exp`` tile is at most ``[BLOCK]`` wide.
        """
        row = tl.program_id(0)
        b = row // S1
        t = row % S1
        base = logits_ptr + b.to(tl.int64) * stride_b + t.to(tl.int64) * stride_t
        m = -float("inf")
        acc = 0.0
        for start in range(0, VOCAB, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            msk = offs < VOCAB
            x = tl.load(base + offs, mask=msk, other=-float("inf")).to(tl.float32) * INV_TAU
            cmax = tl.max(x, 0)
            new_m = tl.maximum(m, cmax)
            acc = acc * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), 0)
            m = new_m
        tl.store(logz_ptr + row, m + tl.log(acc))

    @triton.jit
    def _qad_softmax_kernel(
        logits_ptr,
        out_ptr,
        logz_ptr,
        stride_b,
        stride_t,
        out_stride_b,
        out_stride_t,
        S1,
        VOCAB,
        INV_TAU,
        BLOCK: tl.constexpr,
    ):
        """Write ``softmax(s/τ)`` of one row/vocab-block straight into ``out``.

        ``out`` is the full ``[B, seq, vocab]`` gradient buffer; only the first
        ``S1`` positions per batch are written (the last predicts nothing), so the
        shifted-out position keeps its zero initialization. No full-vocab softmax
        is ever materialized outside this buffer.
        """
        row = tl.program_id(0)
        blk = tl.program_id(1)
        b = row // S1
        t = row % S1
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        msk = offs < VOCAB
        x = tl.load(
            logits_ptr + b.to(tl.int64) * stride_b + t.to(tl.int64) * stride_t + offs,
            mask=msk,
            other=-float("inf"),
        ).to(tl.float32) * INV_TAU
        logz = tl.load(logz_ptr + row)
        y = tl.exp(x - logz)
        tl.store(out_ptr + b.to(tl.int64) * out_stride_b + t.to(tl.int64) * out_stride_t + offs, y, mask=msk)


def _triton_logsumexp(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Per-row ``logsumexp(s/τ)`` over the shifted logits via Triton -> ``[M]``."""
    B, S, V = logits.shape
    S1 = S - 1
    M = B * S1
    logz = torch.empty(M, device=logits.device, dtype=torch.float32)
    sb, st, _ = logits.stride()
    _qad_logsumexp_kernel[(M,)](
        logits, logz, sb, st, S1, V, 1.0 / tau, BLOCK=_QAD_TRITON_BLOCK, num_warps=8
    )
    return logz


def _triton_softmax_into(out: torch.Tensor, logits: torch.Tensor, logz: torch.Tensor, tau: float) -> None:
    """Fill ``out[:, :S1, :]`` with ``softmax(s/τ)`` via Triton (in place)."""
    B, S, V = logits.shape
    S1 = S - 1
    M = B * S1
    n_blk = triton.cdiv(V, _QAD_TRITON_BLOCK)
    sb, st, _ = logits.stride()
    osb, ost, _ = out.stride()
    _qad_softmax_kernel[(M, n_blk)](
        logits, out, logz, sb, st, osb, ost, S1, V, 1.0 / tau, BLOCK=_QAD_TRITON_BLOCK, num_warps=8
    )


class _QADDistillLoss(torch.autograd.Function):
    """Memory-lean forward-KL distillation loss, à la Unsloth's fused cross-entropy.

    The target is the full-precision teacher's (temperature-scaled) distribution
    ``p_T = softmax(t/τ)`` -- *not* the hard next-token label. Per position:

        L = KL(p_T || p_S) = Σ_k p_T(k)·log p_T(k) − Σ_k p_T(k)·log p_S(k)
          = Σ_k p_T log p_T + logsumexp(s/τ) − Σ_k p_T(k)·(s_k/τ)

    using ``log p_S(k) = s_k/τ − logsumexp(s/τ)`` and ``Σ_k p_T(k) = 1``. The
    labels are only used as a validity mask (``-100`` = ignore). As in
    Unsloth's ``Fast_CrossEntropyLoss``:

    * the **forward** keeps only the per-row normalizer ``logsumexp`` (``[M]``),
      computed by a Triton kernel that streams over the vocabulary (CUDA) or by
      the chunked PyTorch fallback;
    * the **backward** recomputes ``softmax(s/τ)`` from the saved logits -- again
      via a Triton kernel on CUDA -- and writes the gradient
      ``(softmax(s/τ) − p_T)`` straight into the (sole) output buffer.

    The one-hot label of plain cross-entropy is simply replaced by the teacher
    distribution ``p_T`` (the teacher Top-K scattered back to vocab, or the full
    distribution when ``teacher_index is None``). No full-vocab log-softmax ever
    enters the autograd graph, so the retained memory is just the logits (the
    model output, unavoidable) plus the ``[M]`` normalizer.
    """

    @staticmethod
    def forward(ctx, logits, teacher_index, teacher_prob, labels, tau, eps, vocab_chunk):
        # logits:        [B, seq, vocab]
        # teacher_prob:  [B, seq, K] (Top-K) or [B, seq, vocab] (full)
        # teacher_index: [B, seq, K] (Top-K) or None (full)
        # labels:        [B, seq] (-100 = ignore)
        B, S, V = logits.shape
        tau = float(tau)
        vocab_chunk = int(vocab_chunk) if vocab_chunk else V
        vocab_chunk = max(1, min(vocab_chunk, V))
        use_triton = _HAS_TRITON and logits.is_cuda

        # The kernels index rows via strides; only the last (vocab) dim must be
        # contiguous. Model LM-head logits already are, but be safe.
        if use_triton and logits.stride(-1) != 1:
            logits = logits.contiguous()

        S1 = S - 1
        M = B * S1
        mask = (labels[:, 1:].reshape(M) != -100)
        n_valid = int(mask.sum().item())

        ctx.shape = (B, S, V)
        ctx.tau = tau
        ctx.eps = float(eps)
        ctx.vocab_chunk = vocab_chunk
        ctx.n_valid = n_valid
        ctx.full_mode = teacher_index is None
        ctx.use_triton = use_triton

        if n_valid == 0:
            ctx.save_for_backward(logits)
            return logits.new_zeros(())

        # Per-row log-normalizer logsumexp(s/τ) -> [M], float32.
        if use_triton:
            logz = _triton_logsumexp(logits, tau)
        else:
            logz = _streaming_logsumexp(logits[:, :-1, :].reshape(M, V), tau, vocab_chunk)

        s3 = logits[:, :-1, :]  # [B, S1, V] view
        logz3 = logz.view(B, S1)
        if teacher_index is None:
            # Full-distribution teacher (already sums to 1). The [B, S1, chunk]
            # product is transient; nothing full-vocab beyond the resident probs.
            p_full = teacher_prob[:, :-1, :].float()  # [B, S1, V] (already resident)
            cross = torch.zeros(B, S1, device=logits.device, dtype=torch.float32)
            neg_ent = torch.zeros(B, S1, device=logits.device, dtype=torch.float32)
            for c0 in range(0, V, vocab_chunk):
                c1 = min(c0 + vocab_chunk, V)
                pc = p_full[:, :, c0:c1]
                cross += (pc * (s3[:, :, c0:c1].float() / tau)).sum(dim=-1)
                neg_ent += torch.xlogy(pc, pc).sum(dim=-1)
            ti = logits.new_empty(0)
            pT = p_full
        else:
            K = teacher_prob.shape[-1]
            ti = teacher_index[:, :-1, :]  # [B, S1, K]
            # Cache already holds softmax(topk(t)/τ) over the K set; the
            # renormalization is only a guard against storage round-off.
            pT = teacher_prob[:, :-1, :].float()
            pT = pT / pT.sum(dim=-1, keepdim=True).clamp_min(ctx.eps)
            s_sel = s3.gather(-1, ti).float() / tau  # [B, S1, K]
            cross = (pT * s_sel).sum(dim=-1)  # [B, S1]
            neg_ent = torch.xlogy(pT, pT).sum(dim=-1)  # Σ p_T log p_T

        # KL(p_T || p_S) = Σ p_T log p_T − Σ p_T log p_S = neg_ent + logz − cross.
        # The teacher-entropy term is constant w.r.t. the student, so the gradient
        # is identical to soft-target CE; including it makes the logged loss a true
        # KL (0 at a perfect match) instead of KL + H(p_T).
        mask3 = mask.view(B, S1)
        loss_row = torch.where(mask3, logz3 - cross + neg_ent, torch.zeros_like(logz3))
        loss = (tau * tau) * loss_row.sum() / n_valid

        ctx.save_for_backward(logits, logz, pT, ti, mask)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        B, S, V = ctx.shape
        if ctx.n_valid == 0:
            (logits,) = ctx.saved_tensors
            return torch.zeros_like(logits), None, None, None, None, None, None

        logits, logz, pT, ti, mask = ctx.saved_tensors
        tau = ctx.tau
        vocab_chunk = ctx.vocab_chunk
        S1 = S - 1

        grad_logits = torch.zeros_like(logits)
        seg = grad_logits[:, :-1, :]  # [B, S1, V] view -> in-place writes land in grad_logits

        # Recompute softmax(s/τ) and write it straight into the gradient buffer
        # (no full-vocab softmax retained anywhere else).
        if ctx.use_triton:
            _triton_softmax_into(grad_logits, logits, logz, tau)
        else:
            logz3 = logz.view(B, S1, 1)
            s3 = logits[:, :-1, :]
            for c0 in range(0, V, vocab_chunk):
                c1 = min(c0 + vocab_chunk, V)
                sc = s3[:, :, c0:c1].float() / tau
                seg[:, :, c0:c1] = torch.exp(sc - logz3).to(grad_logits.dtype)

        # Subtract the teacher distribution: full vector, or Top-K scattered back.
        if ctx.full_mode:
            seg -= pT.view(B, S1, V).to(grad_logits.dtype)
        else:
            K = ti.shape[-1]
            seg.scatter_add_(2, ti.view(B, S1, K), (-pT).view(B, S1, K).to(grad_logits.dtype))

        # dL/ds = (τ/N)·mask·(softmax(s/τ) − p_T); fold in the upstream scalar grad.
        scale = (grad_out * (tau / ctx.n_valid)).to(grad_logits.dtype)
        seg *= mask.view(B, S1, 1).to(grad_logits.dtype)
        seg *= scale

        return grad_logits, None, None, None, None, None, None


@register_pipeline_member(QADConfig)
class QADQuantizer(SignRoundQuantizer):
    """Whole-model QAD: distill the FP teacher's cached Top-K into the student.

    Unlike block-by-block tuning, QAD optimizes the entire network jointly with a
    single full-model forward to the LM head (``whole_model_tuning``), so there is
    no nblocks fusing and no probing of the modules between the last block and the
    LM head:

    * The full-precision teacher is forwarded once (offline); only the Top-K
      ``(index, prob)`` per token is cached to disk.
    * All quantizable layers are wrapped and tuned jointly against a
      temperature-scaled soft-label cross-entropy over the teacher Top-K, so
      neither the teacher model nor full-vocab logits stay resident.
    """

    #: Run a single full-model forward instead of the per-block loop.
    whole_model_tuning = True

    def __init__(self, config: QADConfig) -> None:
        super().__init__(config)
        self.qad_top_k = getattr(config, "topk", 32)
        self.qad_temperature = getattr(config, "lfq_temperature", 1.0)
        self.qad_workspace = getattr(config, "qad_workspace", None)
        # Loss mixing weights: total = kd_weight * KD(teacher) + ce_weight * CE(label).
        # kd_weight=0 => pure hard-label QAT (teacher cache is skipped);
        # ce_weight>0 => mix the ground-truth cross-entropy into the distillation.
        self.qad_kd_weight = float(getattr(config, "qad_kd_weight", 1.0))
        self.qad_ce_weight = float(getattr(config, "qad_ce_weight", 0.0))
        self._qad_use_teacher = self.qad_kd_weight > 0.0
        # Vocab chunk width for the streaming KD loss (caps the softmax temporary
        # at [N, chunk] in both forward and backward). 0/None = whole vocab.
        self.qad_loss_vocab_chunk = getattr(config, "qad_loss_vocab_chunk", 32768)
        # Recompute block activations in backward instead of storing them.
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", True)
        self._teacher_cache: TeacherTopKCache | None = None

    # ── optimizer selection (signround / adamw) ───────────────────────────────
    def _get_optimizer(self, optimizer):
        """Resolve the QAD optimizer: SignSGD ('signround') or AdamW ('adamw')."""
        opt = str(optimizer or "signround").lower()
        self._use_adamw = opt in ("adamw", "adam")
        if self._use_adamw:
            self.momentum = None  # AdamW manages momentum internally
            return torch.optim.AdamW
        return SignSGD

    def _get_scaler(self):
        if not getattr(self, "_use_adamw", False):
            return super()._get_scaler()
        scaler = None
        if self.model_context.amp and not check_is_cpu(device_manager.device):
            from torch.amp import GradScaler

            device_type = torch.device(device_manager.device).type
            scaler = GradScaler(device_type, init_scale=1024, growth_interval=100000)
        return scaler

    def _scale_loss_and_backward(self, scaler, loss):
        if not getattr(self, "_use_adamw", False):
            return super()._scale_loss_and_backward(scaler, loss)
        if scaler is not None:
            loss = scaler.scale(loss)
        loss.backward()
        if is_hpex_available():
            htcore.mark_step()
        return loss

    def _step(self, scaler, optimizer, lr_schedule):
        if not getattr(self, "_use_adamw", False):
            return super()._step(scaler, optimizer, lr_schedule)
        if scaler is not None:
            scaler.step(optimizer)
            optimizer.zero_grad()
            lr_schedule.step()
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad()
            lr_schedule.step()
        if is_hpex_available():
            htcore.mark_step()

    # ── workspace / cache management ──────────────────────────────────────────
    def _resolve_workspace(self) -> str:
        ws = self.qad_workspace
        # Treat None / NaN / empty as "unset" and fall back to AutoRound's workspace.
        unset = (
            ws is None
            or (isinstance(ws, float) and math.isnan(ws))
            or (isinstance(ws, str) and not ws.strip())
        )
        if unset:
            ws = os.path.join(envs.AR_WORK_SPACE, "qad_teacher_cache")
        os.makedirs(ws, exist_ok=True)
        return ws

    @staticmethod
    def _as_batch_ids(ids, device) -> torch.Tensor:
        """Normalize one sample's token ids to a ``[1, seq]`` long tensor on *device*."""
        if not torch.is_tensor(ids):
            ids = torch.as_tensor(ids)
        ids = ids.to(device=device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids.long()

    @staticmethod
    def _strip_accelerate_hooks(model) -> None:
        """Remove accelerate dispatch hooks so a plain full-model forward works."""
        if hasattr(model, "hf_device_map") and len(getattr(model, "hf_device_map", {})) > 0:
            try:
                import accelerate

                accelerate.hooks.remove_hook_from_submodules(model)
            except Exception:  # pragma: no cover - best effort
                pass

    @staticmethod
    def _safe_input_ids(ids: torch.Tensor, vocab_size: int | None) -> torch.Tensor:
        """Return a forward-safe copy of *ids* with out-of-range tokens replaced.

        Calibration ``input_ids`` carry the ``-100`` loss-ignore sentinel at
        trailing positions (pad tokens, repeated tail tokens, the last position).
        Those are invalid embedding indices, so replace anything ``< 0`` or ``>=
        vocab_size`` with ``0``. The original ids are still used as loss labels so
        the masked positions are correctly ignored.
        """
        safe = ids.clone()
        invalid = safe < 0
        if vocab_size is not None:
            invalid = invalid | (safe >= vocab_size)
        if invalid.any():
            safe[invalid] = 0
        return safe

    # @torch.no_grad()
    def _forward_logits(self, model, batch_ids: torch.Tensor, vocab_size: int | None) -> torch.Tensor:
        """Full-model forward to logits with a padding-aware attention mask.

        Mirrors normal tuning: the ``-100`` (trailing pad / ignore) positions are
        excluded via ``attention_mask`` so they neither contribute to nor corrupt
        the valid positions, and the invalid ids are sanitized for the embedding.
        """
        safe_ids = self._safe_input_ids(batch_ids, vocab_size)
        attention_mask = (batch_ids != -100).to(torch.long)
        return model(input_ids=safe_ids, attention_mask=attention_mask, use_cache=False).logits

    @staticmethod
    def _enable_gradient_checkpointing(model) -> None:
        """Turn on HF activation checkpointing (``use_reentrant=False``).

        ``use_reentrant=False`` is required because QAD freezes every non-wrapper
        parameter, so the tensor entering the first decoder block
        (``inputs_embeds``) does not require grad; the reentrant implementation
        would then drop the gradient at the checkpoint boundary and starve the
        wrapper params. The saved-tensor-hooks implementation has no such
        restriction. No-op when the model does not support checkpointing.

        Because the base model is fully frozen, we also call
        ``enable_input_require_grads()`` (the PEFT/LoRA trick): it hooks the
        embedding output to ``requires_grad_(True)`` so the checkpointed region
        has a grad-requiring input and the autograd graph stays connected
        (otherwise ``loss`` ends up with no ``grad_fn``).
        """
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning("QAD: model does not support gradient checkpointing; ignoring the flag.")
            return
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:  # older transformers without the kwargs argument
            model.gradient_checkpointing_enable()
        # Keep the graph connected through the frozen embeddings.
        enable_input_grads = getattr(model, "enable_input_require_grads", None)
        if callable(enable_input_grads):
            enable_input_grads()

    @staticmethod
    def _disable_gradient_checkpointing(model) -> None:
        """Best-effort revert of :meth:`_enable_gradient_checkpointing`."""
        fn = getattr(model, "gradient_checkpointing_disable", None)
        if callable(fn):
            try:
                fn()
            except Exception:  # pragma: no cover - best effort
                pass
        disable_input_grads = getattr(model, "disable_input_require_grads", None)
        if callable(disable_input_grads):
            try:
                disable_input_grads()
            except Exception:  # pragma: no cover - best effort
                pass

    @staticmethod
    def _teacher_cache_signature(model, token_ids) -> dict:
        """Fingerprint of the run so a cache is never reused across models/data.

        Includes the model identity (type + vocab size) and the calibration shape
        (sample count + sequence length). The Top-K width is merged in by the
        cache itself. A mismatch triggers a full rebuild of the teacher files.
        """
        cfg = getattr(model, "config", None)
        seqlen = None
        if token_ids is not None and len(token_ids) > 0:
            first = token_ids[0]
            if torch.is_tensor(first):
                seqlen = int(first.reshape(-1).shape[0])
            else:
                try:
                    seqlen = len(first)
                except TypeError:
                    seqlen = None
        return {
            "model_type": getattr(cfg, "model_type", None),
            "vocab_size": getattr(cfg, "vocab_size", None),
            "name_or_path": getattr(cfg, "_name_or_path", None),
            "nsamples": None if token_ids is None else len(token_ids),
            "seqlen": seqlen,
        }

    @torch.no_grad()
    def _build_teacher_cache_from_model(self, model, token_ids, device) -> None:
        """Forward the FP model to the LM head once per sample and cache the Top-K.

        Uses the real model forward (``model(input_ids).logits``), so no module
        between the last block and the LM head needs to be probed. Samples already
        on disk are skipped; each sample's logits are freed right after caching.
        """
        if self._teacher_cache is None:
            self._teacher_cache = TeacherTopKCache(
                self._resolve_workspace(),
                self.qad_top_k,
                signature=self._teacher_cache_signature(model, token_ids),
                temperature=float(self.qad_temperature),
            )
        cache = self._teacher_cache
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        was_training = model.training
        model.eval()
        n_built = 0
        for sample_idx, ids in enumerate(token_ids):
            batch_ids = self._as_batch_ids(ids, device)
            # Skip only when the cached file was produced by *this* sample; a
            # stale file from another model/dataset is rebuilt (overwritten).
            if cache.is_valid_for(sample_idx, batch_ids):
                continue
            logits = self._forward_logits(model, batch_ids, vocab_size)  # [1, seq, vocab]
            # Store the original ids (with any -100) so they serve as loss labels.
            cache.build_sample(sample_idx, batch_ids, logits)
            del logits
            n_built += 1
        if was_training:
            model.train()
        if n_built:
            logger.info("QAD: cached teacher Top-%d for %d samples at %s", self.qad_top_k, n_built, cache.workspace)
        clear_memory()

    # ── whole-model tuning entry point ────────────────────────────────────────
    def quantize_model(self, model, token_ids, device=None) -> None:
        """Tune the whole model jointly against the cached teacher distribution.

        Steps: (1) cache the FP teacher Top-K via a real full-model forward,
        (2) wrap every quantizable layer, (3) optimize all rounding/min-max
        params jointly with a full-model forward → KD loss → backward, and
        (4) apply the best params and unwrap.
        """
        import copy

        from auto_round.compressors.utils import IndexSampler, collect_best_params
        from auto_round.wrapper import unwrapper_block

        device = device or device_manager.device
        if token_ids is None or len(token_ids) == 0:
            raise ValueError("QAD requires calibration token_ids but none were provided.")

        # Make the FP model resident on the compute device and hook-free.
        self._strip_accelerate_hooks(model)
        model.to(device)

        # 1) Offline teacher cache from a real full-model forward (no probing).
        #    Skipped entirely for a pure hard-label QAT run (``--qad_kd_weight 0``).
        if self._qad_use_teacher:
            self._build_teacher_cache_from_model(model, token_ids, device)
        else:
            logger.info("QAD: KD weight is 0 -> hard-label QAT only; skipping the teacher cache.")

        # 2) Wrap every quantizable layer across the whole model.
        quantized_layers, unquantized_layers = self.wrapper_block(
            model,
            self.enable_minmax_tuning,
            self.enable_norm_bias_tuning,
            enable_torch_compile=self.compress_context.enable_torch_compile,
            device=device,
            enable_neuqi=getattr(self.config, "enable_neuqi", False),
            enable_shared_expert_tuning=self.enable_shared_expert_tuning,
        )
        logger.info("QAD: wrapped %d quantizable layers (%d kept FP).", len(quantized_layers), len(unquantized_layers))
        # The wrappers have now copied any warm-start params (e.g. from a prior
        # SignRound pass in ``--algs signround,qad``) into their tuning params.
        # Report how many layers were warm-started and drop the CPU stash.
        n_warm = 0
        for _n, m in model.named_modules():
            orig = getattr(m, "orig_layer", None)
            if orig is not None and getattr(orig, "_ar_warm_start", None):
                n_warm += 1
                del orig._ar_warm_start
        if n_warm:
            logger.info("QAD: warm-started %d layers from the previous block-wise pass.", n_warm)
        else:
            logger.info("QAD: no warm-start params found; tuning starts from RTN init.")

        # 3) Collect trainable params and build the optimizer / lr schedule.
        round_params, minmax_params = [], []
        for _n, m in model.named_modules():
            if hasattr(m, "orig_layer"):
                for key in m.params.keys():
                    (minmax_params if ("min" in key or "max" in key) else round_params).append(m.params[key])
        if not round_params and not minmax_params:
            logger.warning("QAD: no trainable quantization parameters found; nothing to tune.")
            unwrapper_block(model, {})
            return

        base_lr = self.lr if self.lr is not None else (1.0 / self.iters if self.iters > 0 else 1e-3)
        minmax_lr = self.minmax_lr if self.minmax_lr is not None else base_lr
        extra_kwargs = {} if self.momentum is None else {"momentum": self.momentum}
        param_groups = [{"params": round_params, "lr": torch.tensor(float(base_lr))}]
        if self.enable_minmax_tuning and minmax_params:
            param_groups.append({"params": minmax_params, "lr": torch.tensor(float(minmax_lr))})
        optimizer = self.optimizer(param_groups, lr=torch.tensor(float(base_lr)), weight_decay=0, **extra_kwargs)
        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(1, self.iters)
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)
        scaler = self._get_scaler()

        # 4) Joint tuning loop: full-model forward → KD loss → step.
        if self.gradient_checkpointing:
            # HF only runs the checkpointed path when the module is in training
            # mode, so switch to train(). LLM decoder dropout is 0 by default, so
            # the forward stays deterministic; the wrapper params still train via
            # STE. Recomputing activations in backward is the whole point here.
            self._enable_gradient_checkpointing(model)
            model.train()
            logger.info("QAD: gradient checkpointing enabled (activations recomputed in backward).")
        else:
            model.eval()  # deterministic forward; wrapper params still train via STE
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        nsamples = len(token_ids)
        batch_size = min(self.calibration_context.batch_size, nsamples)
        grad_accum = max(1, self.gradient_accumulate_steps)
        global_bs = min(nsamples, batch_size * grad_accum)
        sampler = IndexSampler(nsamples, global_bs)
        best_loss = float("inf")
        best_iter = -1
        best_params: dict = {}
        # The best-param snapshot covers every wrapped layer of the whole model
        # (``value`` alone is weight-sized), so always keep it on CPU instead of
        # duplicating it on the accelerator.
        best_cache_device = "cpu"
        from tqdm import tqdm

        # Whole-model tuning has no per-block loop, so progress is tracked over
        # the joint tuning iterations instead of over blocks.
        pbar = tqdm(range(self.iters), desc="QAD tuning")
        try:
            for it in pbar:
                global_indices = sampler.next_batch()
                if self._qad_use_teacher:
                    self._teacher_cache.prefetch(global_indices)
                optimizer.zero_grad()
                total_loss = 0.0
                for start in range(0, len(global_indices), batch_size):
                    idx = global_indices[start : start + batch_size]
                    # The teacher cache is the single source of truth: it carries the
                    # sample's own input_ids together with the teacher Top-K, so no
                    # separate token-id copy is indexed here. For a pure QAT run
                    # (no teacher) the labels come straight from ``token_ids``.
                    if self._qad_use_teacher:
                        batch_ids, t_index, t_prob = self._load_teacher_batch(idx, device)
                    else:
                        batch_ids, t_index, t_prob = self._load_ids_batch(idx, token_ids, device), None, None
                    logits = self._forward_logits(model, batch_ids, vocab_size)
                    loss = self._qad_loss(logits, t_index, t_prob, batch_ids)
                    self._scale_loss_and_backward(scaler, loss)
                    total_loss += loss.item()
                    # Drop the large per-microbatch logits/graph before the next
                    # forward so the cached allocator can reuse the blocks.
                    del logits, loss, batch_ids, t_index, t_prob
                self._step(scaler, optimizer, lr_schedule)
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_iter = it
                    if not self.not_use_best_mse:
                        best_params = collect_best_params(model, best_cache_device)
                # Switch: with ``--not_use_best_mse`` keep the LAST iteration's
                # params instead of the best-loss ones. Collect them explicitly --
                # an empty dict would make ``unwrapper`` fall back to
                # value=0/scale=1 (plain RTN), not the tuned state.
                if self.not_use_best_mse and it == self.iters - 1:
                    best_params = collect_best_params(model, best_cache_device)
                pbar.set_postfix(loss=f"{total_loss:.3e}", best=f"{best_loss:.3e}@{best_iter}")
                logger.debug("QAD iter %d loss %.5e", it, total_loss)
                # # Periodically release cached blocks to curb fragmentation during
                # # the long whole-model tuning loop (cheap relative to a full step).
                # if self.iters <= 1 or (it % 10 == 0):
                #     clear_memory()
        finally:
            # 5) Apply the best params and unwrap, then release the prefetch pool.
            pbar.close()
            if best_iter >= 0:
                logger.info(
                    "QAD: best loss %.4e at iter %d/%d%s.",
                    best_loss,
                    best_iter,
                    self.iters,
                    " (not used: --not_use_best_mse keeps the last iter)" if self.not_use_best_mse else "",
                )
            if self.gradient_checkpointing:
                self._disable_gradient_checkpointing(model)
                model.eval()
            unwrapper_block(model, best_params)
            if self._teacher_cache is not None:
                self._teacher_cache.shutdown()
            clear_memory()

    def _load_ids_batch(self, sample_indices, token_ids, device) -> torch.Tensor:
        """Stack raw calibration ``input_ids`` for a batch (pure-QAT path).

        Used when the teacher cache is disabled (``--qad_kd_weight 0``): the
        hard-label CE only needs the token ids (with their ``-100`` ignore
        sentinels preserved as loss labels), so they are read straight from
        ``token_ids`` instead of from the on-disk teacher files.
        """
        ids_list = [self._as_batch_ids(token_ids[int(i)], device).squeeze(0) for i in sample_indices]
        return torch.stack(ids_list, dim=0)

    def _load_teacher_batch(self, sample_indices, device):
        """Load and stack cached ``(input_ids, index, prob)`` for a batch.

        Returns ``(input_ids, index, prob)`` where ``input_ids`` is ``[B, seq]``,
        ``index`` is ``[B, seq, K]`` (Top-K mode) or ``None`` (full-distribution
        mode), and ``prob`` is ``[B, seq, K]`` / ``[B, seq, vocab]``.
        """
        ids_list, idx_list, prob_list = [], [], []
        full_mode = False
        for i in sample_indices:
            input_ids, index, prob = self._teacher_cache.load(int(i), device)
            if index is None:
                full_mode = True
            ids_list.append(input_ids)
            idx_list.append(index)
            prob_list.append(prob)
        batch_ids = torch.stack(ids_list, dim=0)
        teacher_prob = torch.stack(prob_list, dim=0)
        teacher_index = None if full_mode else torch.stack(idx_list, dim=0)
        return batch_ids, teacher_index, teacher_prob

    def _qad_kd_loss(self, student_logits, teacher_index, teacher_prob, labels):
        """Temperature-scaled forward KL against the FP teacher's distribution.

        Per next-token position, the loss is ``KL(p_T || p_S)`` with
        ``p_T = softmax(t/τ)`` from the cached teacher logits and
        ``p_S = softmax(s/τ)``. When
        *teacher_index* is given, the sum is over the teacher's Top-K set (teacher
        probs renormalized within it); when it is ``None`` the teacher is the full
        vocabulary distribution. Positions with label ``-100`` are ignored.

        The heavy lifting is delegated to :class:`_QADDistillLoss`, a custom
        autograd Function that -- following Unsloth's fused cross-entropy -- keeps
        only the per-row ``logsumexp`` in the graph and recomputes ``softmax`` in
        the backward (Triton on CUDA, chunked PyTorch otherwise), so no full-vocab
        ``log_softmax`` tensor is ever retained.

        Args:
            student_logits: ``[B, seq, vocab]``.
            teacher_index:  ``[B, seq, K]`` teacher Top-K token ids, or ``None``
                            for a full-distribution teacher.
            teacher_prob:   ``[B, seq, K]`` / ``[B, seq, vocab]`` teacher probs.
            labels:         ``[B, seq]`` token ids (``-100`` = ignore).

        Returns:
            Scalar loss tensor (mean over valid next-token positions).
        """
        return _QADDistillLoss.apply(
            student_logits,
            teacher_index,
            teacher_prob,
            labels,
            float(self.qad_temperature),
            1e-6,
            self.qad_loss_vocab_chunk,
        )

    def _qad_ce_loss(self, student_logits, labels):
        """Hard-label next-token cross-entropy over the student logits.

        This is the standard causal-LM objective: at position ``t`` the target is
        the ground-truth token ``labels[t+1]`` (``-100`` positions are ignored).
        It is expressed as a one-hot special case of :class:`_QADDistillLoss`
        (teacher mass 1.0 on the label, temperature 1.0), so it reuses the same
        memory-lean streaming/Triton path as the KD loss and shares the exact
        next-token shift and ignore mask -- no extra full-vocab log-softmax is
        materialized.

        Args:
            student_logits: ``[B, seq, vocab]``.
            labels:         ``[B, seq]`` token ids (``-100`` = ignore).

        Returns:
            Scalar cross-entropy loss (mean over valid next-token positions).
        """
        # Target for student position ``t`` is the token at ``t+1``; align a
        # one-hot teacher index to the same ``[:, :-1]`` slice _QADDistillLoss
        # consumes. Invalid (-100) targets are set to 0 for the gather and are
        # dropped anyway by the shared ``labels[:, 1:] != -100`` mask.
        next_labels = torch.full_like(labels, -100)
        next_labels[:, :-1] = labels[:, 1:]
        ce_index = next_labels.clamp_min(0).unsqueeze(-1)  # [B, seq, 1]
        ce_prob = torch.ones_like(ce_index, dtype=torch.float32)  # one-hot mass
        return _QADDistillLoss.apply(
            student_logits,
            ce_index,
            ce_prob,
            labels,
            1.0,  # hard-label CE uses no temperature smoothing
            1e-6,
            self.qad_loss_vocab_chunk,
        )

    def _qad_loss(self, student_logits, teacher_index, teacher_prob, labels):
        """Combined QAD objective: ``kd_weight * KD + ce_weight * CE``.

        * ``kd_weight > 0`` adds the teacher-distribution distillation term
          (:meth:`_qad_kd_loss`); *teacher_index*/*teacher_prob* come from the
          offline cache and may be ``None`` when distillation is disabled.
        * ``ce_weight > 0`` adds the hard-label cross-entropy (:meth:`_qad_ce_loss`),
          turning QAD into quantization-aware training (QAT).

        At least one weight is positive (validated by the config).
        """
        loss = None
        if self.qad_kd_weight > 0.0 and teacher_prob is not None:
            loss = self.qad_kd_weight * self._qad_kd_loss(student_logits, teacher_index, teacher_prob, labels)
        if self.qad_ce_weight > 0.0:
            ce = self.qad_ce_weight * self._qad_ce_loss(student_logits, labels)
            loss = ce if loss is None else loss + ce
        if loss is None:  # defensive: config guarantees at least one active term
            raise RuntimeError("QAD loss has no active term; set --qad_kd_weight or --qad_ce_weight > 0.")
        return loss






