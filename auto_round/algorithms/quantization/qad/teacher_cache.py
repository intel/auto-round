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
"""Offline teacher Top-K cache for QAD (quantization-aware distillation).

The full-precision teacher is forwarded through to the LM head once, and only
the Top-K ``(index, prob)`` pairs per token are written to disk together with
the sample's own ``input_ids``. During tuning the student loads these compact
tensors per batch, so neither the teacher model nor the full-vocab logits need
to stay resident -- keeping GPU/CPU memory low.

Storing the ``input_ids`` alongside the teacher makes each file self-describing:
the cache is the single source of truth for both the student forward and the
distillation target (no separate token-id copy), and it can be matched to the
original data by content. The large first-block hidden states are never stored.

On-disk layout (one file per calibration sample)::

    <workspace>/teacher_<sample_idx>.pt  -> {
        "input_ids": int32[seq],          # the tokens that produced the teacher
        "index": int32[seq, K] | None,    # teacher Top-K token ids (None = full vocab)
        "prob":  float32[seq, K] | float16[seq, vocab],
                                          # Top-K: softmax(topk(logits)/τ) re-normalized over K
                                          # full:  softmax(logits/τ)
    }
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from filelock import FileLock

from auto_round.logger import logger


class TeacherTopKCache:
    """Disk-backed store of per-token teacher Top-K ``(index, prob)`` tensors.

    When ``top_k == -1`` the full vocabulary distribution is stored instead (the
    ``index`` field is omitted and the student compares against every logit).

    A small thread pool provides optional read-ahead: :meth:`prefetch` submits
    background disk reads for a set of sample indices so that a later
    :meth:`load` returns an already-loaded tensor instead of blocking on IO.

    The cache is **self-invalidating**: a ``manifest.json`` records the signature
    of the run that produced the files (model fingerprint, vocab size, Top-K, …).
    When a new run opens the workspace with a different signature the stale
    teacher files are cleared, so a cache written for one model/config is never
    silently reused by another (which would, e.g., feed out-of-range token ids
    from a larger vocabulary into a smaller model's ``gather``).
    """

    _MANIFEST = "manifest.json"

    def __init__(
        self,
        workspace: str,
        top_k: int,
        num_workers: int = 4,
        signature: dict | None = None,
        temperature: float = 1.0,
    ) -> None:
        self.top_k = int(top_k)
        # Distillation temperature τ: the cached teacher target is softmax(t/τ),
        # matching the student's softmax(s/τ) in the KD loss.
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("`temperature` must be positive")
        self.store_full = self.top_k == -1
        # Different teachers never invalidate each other's files. Identical
        # signatures share a namespace for reuse, with one active owner at a
        # time so neither rebuilding nor prefetching can race another process.
        if signature is not None:
            namespace = json.dumps(
                {**signature, "top_k": self.top_k, "temperature": self.temperature}, sort_keys=True
            ).encode()
            workspace = os.path.join(workspace, hashlib.sha256(namespace).hexdigest())
        self.workspace = workspace
        os.makedirs(self.workspace, exist_ok=True)
        self._workspace_lock = FileLock(os.path.join(self.workspace, ".cache.lock"), timeout=0, thread_local=False)
        self._workspace_lock.acquire()
        self._num_workers = max(1, int(num_workers))
        self._executor: ThreadPoolExecutor | None = None
        self._pending: dict[int, Future] = {}
        try:
            self._reconcile_signature(signature)
        except BaseException:
            self._workspace_lock.release()
            raise

    # ── signature / invalidation ──────────────────────────────────────────────
    def _manifest_path(self) -> str:
        return os.path.join(self.workspace, self._MANIFEST)

    def _reconcile_signature(self, signature: dict | None) -> None:
        """Drop stale teacher files when the workspace signature changed.

        The Top-K width is always part of the effective signature (a cache built
        with a different ``top_k`` cannot be reused), so it is merged in here.
        """
        if signature is None:
            return
        # `target` versions the stored semantics (Top-K logits re-softmaxed) so
        # caches written by older code are rebuilt.
        sig = {**signature, "top_k": self.top_k, "temperature": self.temperature, "target": "topk_logit_softmax"}
        existing = None
        path = self._manifest_path()
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    existing = json.load(fh)
            except (OSError, json.JSONDecodeError):
                existing = None
        if existing != sig:
            if existing is not None:
                logger.warning(
                    "QAD: teacher cache at %s was built for a different run "
                    "(signature mismatch); clearing stale files.",
                    self.workspace,
                )
            self.clear()
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(sig, fh)
            except OSError:  # pragma: no cover - best effort
                pass

    def _path(self, sample_idx: int) -> str:
        return os.path.join(self.workspace, f"teacher_{int(sample_idx)}.pt")

    def exists(self, sample_idx: int) -> bool:
        return os.path.exists(self._path(sample_idx))

    def is_valid_for(self, sample_idx: int, input_ids: torch.Tensor) -> bool:
        """Return True only when the cached sample was produced by *input_ids*.

        Guards against reusing a file whose tokens differ from the current
        calibration sample (different dataset, order, or sequence length), and
        drops files that are unreadable or whose Top-K width no longer matches
        the configured ``top_k``.
        """
        if not self.exists(sample_idx):
            return False
        try:
            data = torch.load(self._path(sample_idx), map_location="cpu")
        except Exception:  # pragma: no cover - corrupt file -> rebuild
            return False
        cached_ids = data.get("input_ids")
        ids = input_ids.reshape(-1).to(torch.int32).cpu()
        if cached_ids is None or tuple(cached_ids.shape) != tuple(ids.shape) or not torch.equal(cached_ids, ids):
            return False
        index = data.get("index")
        if self.store_full:
            return index is None
        if index is None:
            return False
        # A rebuilt file with a different top_k width cannot be stacked with
        # peers, so force a rebuild when the stored width diverges.
        return index.size(-1) == self.top_k

    @torch.no_grad()
    def build_sample(self, sample_idx: int, input_ids: torch.Tensor, logits: torch.Tensor) -> None:
        """Extract and persist the teacher distribution (and its input) for one sample.

        Args:
            sample_idx: Calibration-sample index (the key).
            input_ids: The token ids that produced *logits*, shape ``[1, seq]`` or ``[seq]``.
            logits: Teacher logits of shape ``[1, seq, vocab]`` or ``[seq, vocab]``.
        """
        if logits.dim() == 3:
            logits = logits[0]
        ids = input_ids.reshape(-1).to(torch.int32).cpu()
        logits = logits.float()
        if self.store_full or self.top_k >= logits.size(-1):
            # Store the full distribution softmax(t/τ); indices are implicit (0..vocab-1).
            probs = torch.softmax(logits / self.temperature, dim=-1)
            torch.save(
                {"input_ids": ids, "index": None, "prob": probs.to(torch.float16).cpu()},
                self._path(sample_idx),
            )
            return
        # Top-K mode: take the teacher's Top-K *logits*, then re-softmax over just
        # those K entries -> p_T = softmax(topk(t)/τ), which sums to 1 over the K set.
        top_logits, indices = logits.topk(self.top_k, dim=-1)  # [seq, K]
        probs = torch.softmax(top_logits / self.temperature, dim=-1)
        torch.save(
            {
                "input_ids": ids,
                "index": indices.to(torch.int32).cpu(),
                # K is small, keep fp32 so the renormalized target is stored exactly.
                "prob": probs.cpu(),
            },
            self._path(sample_idx),
        )

    # Read-ahead prefetching
    def _pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._num_workers, thread_name_prefix="qad_teacher")
        return self._executor

    def _read_cpu(self, sample_idx: int):
        """Blocking disk read of one sample → ``(input_ids, index_or_None, prob)`` on CPU."""
        data = torch.load(self._path(sample_idx), map_location="cpu")
        return data["input_ids"], data["index"], data["prob"]

    def prefetch(self, sample_indices) -> None:
        """Submit background disk reads for *sample_indices* (non-blocking)."""
        for i in sample_indices:
            i = int(i)
            if i not in self._pending:
                self._pending[i] = self._pool().submit(self._read_cpu, i)

    def load(self, sample_idx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Load one sample's ``(input_ids, index, prob)`` onto *device*.

        Uses a pending prefetch future when available; otherwise reads from disk
        synchronously.

        Returns:
            ``(input_ids, index, prob)`` where ``input_ids`` is ``[seq]`` long,
            ``index`` is ``[seq, K]`` long (or ``None`` for a full-distribution
            cache) and ``prob`` is ``[seq, K]`` / ``[seq, vocab]`` float.
        """
        sample_idx = int(sample_idx)
        future = self._pending.pop(sample_idx, None)
        if future is not None:
            input_ids, index, prob = future.result()
        else:
            input_ids, index, prob = self._read_cpu(sample_idx)
        input_ids = input_ids.to(device=device, dtype=torch.long)
        if index is not None:
            index = index.to(device=device, dtype=torch.long)
        prob = prob.to(device=device, dtype=torch.float32)
        return input_ids, index, prob

    def shutdown(self) -> None:
        """Stop the prefetch thread pool and drop any pending futures."""
        self._pending.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._workspace_lock.release()

    def clear(self) -> None:
        """Remove all cached teacher files from the workspace."""
        if not os.path.isdir(self.workspace):
            return
        removed = 0
        for name in os.listdir(self.workspace):
            if name.startswith("teacher_") and name.endswith(".pt"):
                try:
                    os.remove(os.path.join(self.workspace, name))
                    removed += 1
                except OSError:
                    pass
        if removed:
            logger.debug("QAD: cleared %d cached teacher files from %s", removed, self.workspace)
