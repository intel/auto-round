# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Local addition (not upstream). Checkpoints the tuning loop's per-block
# progress to disk so a crash or kill mid-run doesn't require restarting from
# block 0 -- important for large models where each block's tuning can take
# minutes and a full run spans hours. See LOCAL_PATCHES.md.
#
# What this caches and why: AutoRound's block-sequential tuning chains two
# things forward from one block to the next -- ``input_ids`` (the current
# block's reference/FP output, used as the next block's FP reference input)
# and, when ``enable_quanted_input=True`` (SignRound's default), ``q_input``
# (the *quantized* block's output, used as the next block's quantized-input
# companion). Both are cached here, not just ``q_input``: an earlier version of
# this patch assumed ``input_ids`` could be cheaply regenerated from the
# per-block inputs already pre-cached by `cache_inter_data`/
# `try_cache_inter_data_gpucpu` before tuning starts (reasoning that it's a
# pure function of unmodified weights, so it shouldn't matter which code path
# computed it) -- but that pre-cache pass and the in-loop reference forward
# turned out not to be numerically identical (confirmed by a resume test
# producing a 20x-larger tuning loss on the first resumed block vs. an
# uninterrupted control run), so the actual live chain value has to be
# persisted and reloaded verbatim, the same way ``q_input`` already is.
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch

from auto_round.logger import logger

__all__ = ["ResumeState"]

_MANIFEST_NAME = "resume_manifest.json"
_Q_INPUT_NAME = "resume_q_input.pt"
_INPUT_IDS_NAME = "resume_input_ids.pt"


def _to_cpu_recursive(obj):
    """`q_input` (SignRound's `enable_quanted_input` chain value) may be a
    plain Tensor or a nested list/tuple/dict of tensors depending on the
    algorithm/batching path; detach + move to cpu recursively so it's both
    safely picklable and reloadable regardless of which GPU produced it."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_recursive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_recursive(v) for v in obj)
    return obj


def _atomic_write_json(path: Path, data: dict) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_resume_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


class ResumeState:
    """Tracks which blocks of a tuning run have already been quantized and
    written to disk, plus the one small tensor (``q_input``) needed to resume
    the sequential calibration chain correctly. Keyed by a signature over the
    run's identifying configuration, so a resume directory reused for a
    different model/scheme/dataset is detected and ignored rather than
    silently misapplied.
    """

    def __init__(self, resume_dir: str, signature: str, block_names: list[str]):
        self.dir = Path(resume_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.signature = signature
        self.block_names = list(block_names)
        self.manifest_path = self.dir / _MANIFEST_NAME
        self.q_input_path = self.dir / _Q_INPUT_NAME
        self.input_ids_path = self.dir / _INPUT_IDS_NAME
        self.completed_blocks: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            with open(self.manifest_path) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"ResumeState: failed to read {self.manifest_path}: {e}; starting fresh.")
            return
        if data.get("signature") != self.signature:
            logger.info(
                "ResumeState: existing resume manifest is for a different run "
                "(model/scheme/dataset/block list changed); ignoring it and starting fresh."
            )
            return
        completed = data.get("completed_blocks", [])
        # Only trust a prefix of block_names in order -- anything else means the
        # manifest was corrupted or hand-edited; safer to restart from scratch
        # than resume from a possibly-inconsistent point.
        if completed != self.block_names[: len(completed)]:
            logger.warning(
                "ResumeState: completed_blocks in manifest is not a prefix of the "
                "current block order; ignoring it and starting fresh."
            )
            return
        self.completed_blocks = completed
        logger.info(f"ResumeState: resuming after {len(completed)}/{len(self.block_names)} already-quantized blocks")

    @property
    def resume_index(self) -> int:
        """Index of the first block not yet completed."""
        return len(self.completed_blocks)

    def load_q_input(self):
        return self._load_tensor(self.q_input_path)

    def load_input_ids(self):
        return self._load_tensor(self.input_ids_path)

    def _load_tensor(self, path: Path):
        if not self.completed_blocks or not path.exists():
            return None
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            logger.warning(f"ResumeState: failed to load {path.name} ({e}); resuming without it.")
            return None

    def mark_block_done(self, block_name: str, q_input, input_ids) -> None:
        expected = self.block_names[len(self.completed_blocks)]
        assert block_name == expected, (
            f"ResumeState.mark_block_done called out of order: expected {expected!r}, got {block_name!r}"
        )
        if q_input is not None:
            torch.save(_to_cpu_recursive(q_input), self.q_input_path)
        elif self.q_input_path.exists():
            self.q_input_path.unlink()
        # input_ids is required (unlike q_input, which is legitimately None
        # when enable_quanted_input=False) -- the FP reference chain always
        # exists.
        torch.save(_to_cpu_recursive(input_ids), self.input_ids_path)
        self.completed_blocks.append(block_name)
        _atomic_write_json(
            self.manifest_path,
            {"signature": self.signature, "completed_blocks": self.completed_blocks},
        )

    def clear(self) -> None:
        """Remove the resume manifest/cache -- call after a full run completes
        successfully, so a later unrelated run doesn't mistake stale state for
        an in-progress one (only possible if reusing the exact same
        model/scheme/dataset/block list, but still worth cleaning up)."""
        for p in (self.manifest_path, self.q_input_path, self.input_ids_path):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass


def compute_run_signature(
    model_dir: Optional[str],
    scheme_desc: str,
    dataset_desc: str,
    nsamples: int,
    seqlen: int,
    block_names: list[str],
) -> str:
    """Hash the run's identifying configuration. Any change here (different
    model, scheme, dataset, calibration size, or block set) must produce a
    different signature so `ResumeState` refuses to reuse a stale manifest."""
    h = hashlib.sha256()
    for part in (model_dir or "", scheme_desc, dataset_desc, str(nsamples), str(seqlen), "|".join(block_names)):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def layer_config_fingerprint(layer_config) -> str:
    """Deterministic string over the resolved per-layer quantization config.

    ``str(self.scheme)`` (and the literal ``"rtn_with_imatrix"`` used by the
    imatrix-RTN path) does not capture the per-layer bit allocation that
    AutoScheme resolves from ``avg_bits`` -- two runs with different avg_bits
    targets would otherwise produce identical run signatures and silently
    resume each other's state (observed: a quality sweep's avg_bits=4.8
    candidate resuming the 4.2 candidate's 40/40-complete manifest, saving an
    output with no layer tensors at all). Fold this into ``scheme_desc`` when
    calling :func:`compute_run_signature`.

    Only scalar config values (bits, group_size, sym, data_type, ...) are
    included; anything non-scalar is ignored so the fingerprint stays cheap
    and deterministic.
    """
    if not layer_config:
        return "<no-layer-config>"
    parts = []
    for name in sorted(layer_config):
        cfg = layer_config[name]
        if isinstance(cfg, dict):
            desc = ",".join(
                f"{k}={v}"
                for k, v in sorted(cfg.items(), key=lambda kv: str(kv[0]))
                if isinstance(v, (int, float, bool, str, type(None)))
            )
        else:
            desc = str(cfg)
        parts.append(f"{name}:{desc}")
    return ";".join(parts)
