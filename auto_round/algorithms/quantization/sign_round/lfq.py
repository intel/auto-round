# Copyright (c) 2024-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LFQ model helpers and losses, including memory-efficient soft cross-entropy.

The LFQ objective (Eq. 3 of the paper) minimizes the cross-entropy between the
FP model's token distribution ``σ(X·W_FP·W_Head)`` and the quantized model's
distribution ``σ(X·W_q·W_Head)``.  A naive implementation materializes two full
``[batch*seq, vocab]`` float32 tensors (softmax and log_softmax) plus the
autograd graph that produces them -- prohibitive for large-vocab models.

This module reproduces the essence of Unsloth's fused cross-entropy trick, but
for a *soft* target distribution instead of hard integer labels:

* logsumexp reformulation so ``log_softmax`` is never materialized::

      softCE_i = logsumexp(q_i) - Σ_j softmax(fp_i)_j · q_{i,j}        (Σ_j p=1)

* analytic gradient (no autograd graph through softmax)::

      dL/dq_{i,j} = softmax(q_i)_j - softmax(fp_i)_j

* row chunking: only ``chunk_rows × vocab`` float32 elements are live at a time;
  softmax is recomputed in the backward pass (compute-for-memory tradeoff).

Unlike Unsloth's Triton kernel this is pure PyTorch, so it runs on CPU / CUDA /
HPU without a Triton dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from auto_round.logger import logger

if TYPE_CHECKING:
    from auto_round.algorithms.block_runner import BlockForwardRunner
    from auto_round.context.model import ModelContext

__all__ = ["DEFAULT_SOFT_CE_CHUNK_ELEMS", "LFQMixin", "soft_cross_entropy"]

# Target number of (rows × vocab) float32 elements processed per chunk.
# 2**23 ≈ 8.4M elems ≈ 32 MB per temporary fp32 tensor.
DEFAULT_SOFT_CE_CHUNK_ELEMS = 1 << 23


def _rows_per_chunk(vocab_size: int, chunk_elems: int) -> int:
    return max(1, chunk_elems // max(1, vocab_size))


class _SoftCrossEntropy(torch.autograd.Function):
    """Chunked soft cross-entropy with an analytic, memory-light backward."""

    @staticmethod
    def forward(ctx, q_logits, fp_logits, valid_mask, chunk_elems):
        # q_logits / fp_logits: [N, V];  valid_mask: [N] (float, 1.0 keep / 0.0 drop)
        n_rows, vocab_size = q_logits.shape
        rows = _rows_per_chunk(vocab_size, chunk_elems)

        denom = valid_mask.sum().clamp_min(1.0)
        total = q_logits.new_zeros((), dtype=torch.float32)

        for start in range(0, n_rows, rows):
            end = min(start + rows, n_rows)
            m = valid_mask[start:end]
            # No host-side ``if m.sum() == 0`` skip here: it would force a
            # device-to-host sync on every chunk (dozens per forward for large
            # vocabs). Invalid rows are already zeroed by multiplying with ``m``,
            # and ``denom`` guards the all-invalid case, so the math stays correct.
            q_c = q_logits[start:end].float()
            lse_q = torch.logsumexp(q_c, dim=-1)
            p_fp = torch.softmax(fp_logits[start:end].float(), dim=-1)
            dot = (p_fp * q_c).sum(dim=-1)
            ce = (lse_q - dot) * m
            total = total + ce.sum()

        loss = total / denom

        ctx.save_for_backward(q_logits, fp_logits, valid_mask, denom)
        ctx.chunk_elems = chunk_elems
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        q_logits, fp_logits, valid_mask, denom = ctx.saved_tensors
        n_rows, vocab_size = q_logits.shape
        rows = _rows_per_chunk(vocab_size, ctx.chunk_elems)

        # scale = dLoss / denom, broadcast per valid row.
        scale = (grad_output / denom).to(torch.float32)
        dq = torch.empty_like(q_logits)

        for start in range(0, n_rows, rows):
            end = min(start + rows, n_rows)
            m = valid_mask[start:end].unsqueeze(-1)
            # dL/dq = softmax(q) - softmax(fp), recomputed to avoid storing it.
            grad = torch.softmax(q_logits[start:end].float(), dim=-1)
            grad -= torch.softmax(fp_logits[start:end].float(), dim=-1)
            grad *= m * scale
            dq[start:end] = grad.to(dq.dtype)

        return dq, None, None, None


def soft_cross_entropy(
    q_logits: torch.Tensor,
    fp_logits: torch.Tensor,
    valid_mask: torch.Tensor = None,
    chunk_elems: int = DEFAULT_SOFT_CE_CHUNK_ELEMS,
) -> torch.Tensor:
    """Mean soft cross-entropy ``-Σ_j σ(fp)_j log σ(q)_j`` over valid positions.

    Gradients flow only through ``q_logits``; ``fp_logits`` is treated as a
    detached soft target.

    Args:
        q_logits:   Quantized logits, shape ``[..., vocab]``.
        fp_logits:  FP logits (soft target), same shape as ``q_logits``.
        valid_mask: Optional boolean/0-1 mask over positions (all leading dims
                    except vocab). Positions with 0 are excluded from the loss.
        chunk_elems: Target ``rows × vocab`` float32 elements per chunk.

    Returns:
        Scalar loss tensor.
    """
    vocab_size = q_logits.shape[-1]
    q2d = q_logits.reshape(-1, vocab_size)
    fp2d = fp_logits.reshape(-1, vocab_size).detach()

    if valid_mask is None:
        mask = q2d.new_ones(q2d.shape[0], dtype=torch.float32)
    else:
        mask = valid_mask.reshape(-1).to(device=q2d.device, dtype=torch.float32)

    return _SoftCrossEntropy.apply(q2d, fp2d, mask, chunk_elems)


class LFQMixin:
    """Please refer to https://arxiv.org/html/2605.29756v1. LFQ capabilities shared by quantizers without a separate lifecycle.

    The host initializes the LFQ configuration attributes and provides ``model``,
    ``model_context`` and ``block_forward``. It also clears the cached
    ``_lm_head`` and ``_post_block_modules`` when finalizing a run.
    """

    # Supplied by the host; annotations do not shadow its context properties.
    model: Any
    model_context: ModelContext
    block_forward: BlockForwardRunner
    enable_lfq: bool
    enable_qad: bool
    lfq_loss_type: str
    lfq_top_k: int
    lfq_temperature: float
    lfq_num_blocks: int

    def _init_lm_components(self) -> None:
        """Lazily locate and cache ``_post_block_modules`` and ``_lm_head`` for LFQ loss.

        ``_post_block_modules`` is an ordered list of modules that must be applied
        between the last transformer-block output and the lm_head projection.
        Most architectures have a single final norm; OPT additionally has an
        optional ``project_out`` projection.

        Supported without manual configuration:
            LLaMA / Qwen / Gemma / Mistral / InternLM / Phi-3 — ``model.model.norm``
            OPT — ``model.model.decoder.{final_layer_norm, project_out}`` (both optional)
            GPT-2 / Falcon / Bloom — ``model.transformer.ln_f``
            GPT-NeoX / Pythia — ``model.gpt_neox.final_layer_norm``
            Phi / Phi-2 — ``model.model.final_layernorm``
            MPT — ``model.transformer.norm_f``
            ChatGLM — ``model.transformer.encoder.final_layernorm``
            RWKV — ``model.rwkv.ln_out``

        Raises ``AttributeError`` if no lm_head equivalent can be found.
        """
        if hasattr(self, "_lm_head"):
            return

        model = self.model

        # ── lm_head ──────────────────────────────────────────────────────────
        for name in ("lm_head", "embed_out", "output", "head"):
            if hasattr(model, name):
                self._lm_head = getattr(model, name)
                break
        else:
            raise AttributeError(
                f"Cannot locate lm_head in {type(model).__name__}. " "Checked: lm_head, embed_out, output, head."
            )

        # ── post-block processing (ordered list applied before lm_head) ───────
        # OPT: decoder has both an optional final_layer_norm *and* an optional
        # project_out that maps ffn_dim → word_embed_proj_dim.  Detect by probing
        # for the characteristic project_out attribute (may be None).
        try:
            decoder = model.model.decoder
            _ = decoder.project_out  # raises AttributeError if not an OPT decoder
            self._post_block_modules = [m for m in (decoder.final_layer_norm, decoder.project_out) if m is not None]
            return
        except AttributeError:
            pass

        # All other architectures: single optional final norm.
        norm_getters = [
            lambda: model.model.norm,  # LLaMA / Qwen / Gemma / Mistral / InternLM / Phi-3
            lambda: model.transformer.ln_f,  # GPT-2 / Falcon / Bloom
            lambda: model.gpt_neox.final_layer_norm,  # GPT-NeoX / Pythia
            lambda: model.model.final_layernorm,  # Phi / Phi-2
            lambda: model.transformer.norm_f,  # MPT
            lambda: model.transformer.encoder.final_layernorm,  # ChatGLM
            lambda: model.rwkv.ln_out,  # RWKV
        ]
        self._post_block_modules = []
        for getter in norm_getters:
            try:
                norm = getter()
                if norm is not None:
                    self._post_block_modules = [norm]
                    break
            except AttributeError:
                continue

    # Keywords that identify non-text (visual / audio / multimodal) blocks.
    # LFQ loss is only meaningful for pure language-model decoder blocks.
    _NON_TEXT_BLOCK_KEYWORDS = frozenset(
        {
            "vis",
            "vision",
            "visual",
            "image",
            "img",
            "audio",
            "video",
            "patch",
            "pixel",
            "clip",
            "vit",
            "perceiver",
            "resampler",
            "connector",
            "projector",
        }
    )

    def _is_text_decoder_block(self, block_name: str) -> bool:
        """Return ``True`` if *block_name* refers to a text-decoder block.

        Blocks whose names contain any of the non-text keywords (vision, audio,
        image, …) are considered multimodal and excluded from LFQ loss.
        """
        name_lower = block_name.lower()
        return not any(kw in name_lower for kw in self._NON_TEXT_BLOCK_KEYWORDS)

    def _lfq_is_kd_block(self, block_ctx) -> bool:
        """Return ``True`` if the KD (LFQ logit-lens) loss applies to this block.

        * QAD off (default): the last ``lfq_num_blocks`` text-decoder blocks.
        * QAD on: every fused group (chunk) is tuned against the final-output
          LM loss.

        Requires LFQ enabled, token IDs available (checked by the caller) and a
        pure text-decoder block.
        """
        if not self.enable_lfq or not self._is_text_decoder_block(block_ctx.block_name):
            return False
        # QAD: every fused group (chunk) is tuned against the final-output LM loss.
        if self.enable_qad:
            return True
        all_names = getattr(block_ctx, "all_block_names", None)
        group_names = getattr(block_ctx, "block_names", None) or []
        if all_names and group_names and group_names[-1] in all_names:
            text_names = [name for name in all_names if self._is_text_decoder_block(name)]
            endpoint = group_names[-1]
            return endpoint in text_names and text_names.index(endpoint) >= len(text_names) - self.lfq_num_blocks
        return block_ctx.block_index >= block_ctx.block_cnt - self.lfq_num_blocks

    def _get_lfq_downstream_blocks(self, block_ctx, device) -> list:
        """Return the (FP, not-yet-quantized) blocks that follow the current one.

        Only used in the default (non-QAD) LFQ mode with ``lfq_num_blocks`` > 1:
        a non-final KD block's output must go through the remaining decoder
        blocks before the final norm / lm_head, otherwise the loss would be a
        logit-lens on an intermediate hidden state. The returned blocks are
        moved to *device*; the caller is responsible for moving them back.

        Returns an empty list when the current block is the last one, when the
        block order is unknown, or when a downstream block is not materialized
        (e.g. offloaded / meta), in which case it falls back to the logit-lens.
        """
        if self.enable_qad:
            return []
        all_names = getattr(block_ctx, "all_block_names", None)
        cur_names = getattr(block_ctx, "block_names", None) or []
        if not all_names or not cur_names or cur_names[-1] not in all_names:
            if block_ctx.block_index < block_ctx.block_cnt - 1:
                logger.warning_once(
                    "LFQ: block order unavailable, non-final KD blocks fall back to the logit-lens loss "
                    "(block output fed directly to the final norm/lm_head)."
                )
            return []
        downstream_names = all_names[all_names.index(cur_names[-1]) + 1 :]
        if not downstream_names:
            return []

        from auto_round.utils import convert_module_to_hp_if_necessary, get_module

        model = getattr(block_ctx, "model", None) or self.model
        blocks = []
        for name in downstream_names:
            m = get_module(model, name)
            if m is None or any(p.device.type == "meta" or p.numel() == 0 for p in m.parameters()):
                logger.warning_once(
                    f"LFQ: downstream block {name} is not materialized (offloaded/meta); non-final KD blocks fall "
                    "back to the logit-lens loss. Disable low_cpu_mem_usage/disk streaming to propagate "
                    "through the remaining blocks."
                )
                return []
            blocks.append(m)
        for m in blocks:
            convert_module_to_hp_if_necessary(m, self.model_context.amp_dtype, device)
            m.to(device)
        logger.info(
            f"LFQ: propagating {block_ctx.block_name} output through {len(blocks)} downstream FP block(s) "
            "before lm_head"
        )
        return blocks

    def _forward_lfq_downstream(
        self,
        hidden_state: torch.Tensor,
        downstream_blocks: list,
        inputs,
        input_others: dict,
        indices,
        device,
    ) -> torch.Tensor:
        """Forward *hidden_state* through *downstream_blocks* (frozen FP blocks).

        Gradients flow through the activations back to the current block; the
        downstream block parameters themselves are frozen (``requires_grad=False``).
        """
        if not downstream_blocks:
            return hidden_state
        from auto_round.utils import to_device

        block_fwd = self.block_forward
        out_device = hidden_state.device
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        _, batch_others = block_fwd.select_batch(inputs, input_others, indices)
        batch_others = to_device(batch_others, device)
        h = hidden_state.to(device)
        for m in downstream_blocks:
            # block_forward mutates the kwargs dict (pops positional_inputs, injects the
            # hidden-state kwarg), so hand each block a fresh shallow copy.
            raw = block_fwd._forward_one_batch(m, h, dict(batch_others))
            h = block_fwd._normalize_output(raw, m)
        return h.to(out_device)

    def _lfq_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project a block hidden state to vocabulary logits.

        Applies every post-block module (final norm, optional projection, …) in
        order, then runs lm_head.

        Args:
            hidden_state: Block output, shape ``[batch, seq_len, hidden]``.

        Returns:
            Logits tensor of shape ``[batch, seq_len, vocab_size]``.
        """
        self._init_lm_components()
        device = hidden_state.device

        for module in self._post_block_modules:
            module.to(device)
            hidden_state = module(hidden_state)

        self._lm_head.to(device)
        return self._lm_head(hidden_state)

    def lfq_loss(
        self,
        hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        teacher_hidden_state: torch.Tensor | None = None,
        sample_indices: list | None = None,
    ) -> torch.Tensor:
        """Compute the LFQ loss from the last block's hidden states.

        Two loss types are supported, selected by ``self.lfq_loss_type``:

        * ``"ce"`` (default): next-token cross-entropy against ground-truth
          labels. Positions marked with ``-100`` in *input_ids* are excluded.
        * ``"dtk"``: Decoupled Tempered Top-K forward-KL knowledge distillation
          (``KL(p_T || p_S)``). The FP block output (*teacher_hidden_state*) is
          the teacher and the quantized block output (*hidden_state*) is the
          student. Requires *teacher_hidden_state*; falls back to ``"ce"`` when
          it is ``None``.
        * ``"rkl"``: same decoupled Top-K objective but using the reverse KL
          ``KL(p_S || p_T)`` (mode-seeking). Also requires *teacher_hidden_state*.
        * ``"tk"``: only aligns the Top-K-normalized conditional distributions
          (tempered conditional KL within the teacher Top-K, dropping the binary
          mass term). Also requires *teacher_hidden_state*.

        Args:
            hidden_state: Last (quantized/student) block output,
                shape ``[batch, seq_len, hidden]``.
            input_ids:    Token-ID labels with ``-100`` for ignored positions,
                          shape ``[batch, seq_len]``.
            teacher_hidden_state: FP (teacher) block output used by the ``"dtk"``,
                ``"rkl"`` and ``"tk"`` losses, same shape as *hidden_state*.
            sample_indices: Calibration-sample indices for this batch. Unused by
                the base implementation; QAD subclasses use it to load the
                offline teacher Top-K cache from disk.

        Returns:
            Scalar loss tensor.
        """
        device = hidden_state.device
        logits = self._lfq_logits(hidden_state)

        if self.lfq_loss_type in ("dtk", "rkl", "tk") and teacher_hidden_state is not None:
            with torch.no_grad():
                teacher_logits = self._lfq_logits(teacher_hidden_state.to(device))
            return self._decoupled_topk_kd_loss(
                logits,
                teacher_logits,
                input_ids.to(device),
                reverse=(self.lfq_loss_type == "rkl"),
                conditional_only=(self.lfq_loss_type == "tk"),
            )

        if hasattr(self.model, "loss_function"):
            loss = self.model.loss_function(
                logits=logits,
                labels=input_ids.to(device),
                vocab_size=self.model.config.vocab_size,
            )
        else:
            import torch.nn.functional as F

            # Standard causal-LM shift: predict token t+1 from hidden state t.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous().to(device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return loss

    def _decoupled_topk_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        reverse: bool = False,
        conditional_only: bool = False,
    ) -> torch.Tensor:
        """Decoupled Tempered Top-K knowledge-distillation loss (per-token mean).

        Decomposes the KL divergence between the teacher and student next-token
        distributions via the chain rule into:

        * ``L_B``: a binary (Bernoulli) term matching the total probability mass
          the teacher/student assign to the teacher's Top-K set, and
        * ``L_T``: the temperature-scaled conditional KL *within* the Top-K set,
          weighted by the expectation distribution's Top-K mass.

        This "decoupled, tempered Top-K" objective avoids support mismatch while
        preserving the benefits of temperature-scaled distillation.

        The Top-K set is always the teacher's. ``reverse`` selects the KL
        direction:

        * ``reverse=False`` (forward KL): the expectation is over the teacher,
          i.e. ``KL(p_T || p_S)`` — mode-covering, the standard distillation
          objective.
        * ``reverse=True`` (reverse KL): the expectation is over the student,
          i.e. ``KL(p_S || p_T)`` — mode-seeking.

        When ``conditional_only`` is ``True`` the binary mass term ``L_B`` and the
        Top-K mass weighting are dropped, so the loss only aligns the two
        distributions *renormalized within the Top-K set* (the tempered
        conditional KL alone).

        Gradients flow only through the student; the teacher is detached.

        Args:
            student_logits: Student logits, shape ``[batch, seq_len, vocab]``.
            teacher_logits: Teacher logits, shape ``[batch, seq_len, vocab]``.
            labels:         Token-ID labels with ``-100`` for ignored positions,
                            shape ``[batch, seq_len]``. Only used to mask out
                            padding / ignored next-token positions.
            reverse:        When ``True`` compute the reverse KL ``KL(p_S || p_T)``
                            instead of the forward KL ``KL(p_T || p_S)``.
            conditional_only: When ``True`` only align the Top-K-normalized
                            conditional distributions (drop ``L_B`` and the mass
                            weighting).

        Returns:
            Scalar loss tensor (mean over valid next-token positions).
        """
        import torch.nn.functional as F

        top_k = int(self.lfq_top_k)
        tau = float(self.lfq_temperature)
        eps = 1e-6

        # Causal-LM shift: position t predicts token t+1.
        s = student_logits[..., :-1, :].contiguous()
        t = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab = s.size(-1)
        s = s.reshape(-1, vocab).float()
        t = t.reshape(-1, vocab).float()
        mask = shift_labels.reshape(-1) != -100

        if mask.sum() == 0:
            # Keep the graph connected so backward() is a no-op but valid.
            return student_logits.float().sum() * 0.0

        s = s[mask]
        t = t[mask]
        top_k = min(top_k, vocab)

        log_pS = F.log_softmax(s, dim=-1)
        log_pT = F.log_softmax(t, dim=-1)

        # Teacher Top-K set per token (used for both forward and reverse KL).
        topk_idx = t.topk(top_k, dim=-1).indices  # [N, k]
        log_pS_topk = log_pS.gather(-1, topk_idx)  # [N, k]
        log_pT_topk = log_pT.gather(-1, topk_idx)  # [N, k]

        # Total mass on the Top-K set (in log space for stability).
        log_PT_T = torch.logsumexp(log_pT_topk, dim=-1)  # [N]
        log_PS_T = torch.logsumexp(log_pS_topk, dim=-1)  # [N]

        # Select the expectation (E) and the "other" (O) distribution per KL
        # direction. Forward KL(p_T || p_S): E=teacher, O=student.
        # Reverse KL(p_S || p_T): E=student, O=teacher.
        if reverse:
            log_PE_T, log_PO_T = log_PS_T, log_PT_T
            log_pE_topk, log_pO_topk = log_pS_topk, log_pT_topk
        else:
            log_PE_T, log_PO_T = log_PT_T, log_PS_T
            log_pE_topk, log_pO_topk = log_pT_topk, log_pS_topk

        PE_T = log_PE_T.exp().clamp(min=eps, max=1.0 - eps)

        # Conditional distributions within the Top-K: log p(x|T) = log p(x) - log P(T).
        log_cE = log_pE_topk - log_PE_T.unsqueeze(-1)  # [N, k]
        log_cO = log_pO_topk - log_PO_T.unsqueeze(-1)  # [N, k]

        # Apply temperature tau: p^(tau)(x) = p(x)^(1/tau) / Z, computed in log space.
        log_cE_tau = log_cE / tau
        log_cE_tau = log_cE_tau - torch.logsumexp(log_cE_tau, dim=-1, keepdim=True)
        log_cO_tau = log_cO / tau
        log_cO_tau = log_cO_tau - torch.logsumexp(log_cO_tau, dim=-1, keepdim=True)

        # Tempered KL: D^(tau)_KL = tau^2 * KL( pE^(tau) || pO^(tau) ).
        kl_tau = (log_cE_tau.exp() * (log_cE_tau - log_cO_tau)).sum(dim=-1)  # [N]

        if conditional_only:
            # Only align the Top-K-normalized conditional distributions; drop the
            # binary mass term L_B and the Top-K mass weighting.
            return ((tau * tau) * kl_tau).mean()

        # L_B: KL( Bern(P_E(T)) || Bern(P_O(T)) ).
        log_one_minus_PE = torch.log1p(-PE_T)
        log_one_minus_PO = torch.log1p(-log_PO_T.exp().clamp(min=eps, max=1.0 - eps))
        l_b = PE_T * (log_PE_T - log_PO_T) + (1.0 - PE_T) * (log_one_minus_PE - log_one_minus_PO)

        l_t = PE_T * (tau * tau) * kl_tau

        return (l_b + l_t).mean()
