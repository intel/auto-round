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

"""Primitives for the mocked-prefix continuation of the post-block region.

The lm_head input-capture lane replaces the decoder layers of a loaded model
with pass-through stubs, injects the cached last-block output at the last
layer slot, and records the exact rows the model's own post-block code feeds
the language head. Keeping these primitives side-effect free (no orchestrator
state) makes the smoke gate and the capture pass testable in isolation.
"""

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from auto_round.utils.model import get_module

__all__ = [
    "CaptureHead",
    "PassthroughStub",
    "TailInjector",
    "install_block_stubs_",
    "restore_blocks_",
    "tail_smoke_check",
]


class PassthroughStub(nn.Module):
    """Decoder-layer stand-in that forwards the hidden states untouched.

    ``arity`` selects the return convention the model's layer loop expects:
    ``1`` (default) a single-element tuple (``out[0]``/``out[-1]`` both resolve
    to the tensor, which also satisfies cache-collecting loops), ``2`` a
    ``(hidden, residual)`` pair for residual-stream families (e.g. zaya), or
    ``0`` a bare tensor.
    """

    def __init__(self, arity: int = 1):
        super().__init__()
        self.arity = arity

    def forward(self, hidden_states, *args, **kwargs):
        if self.arity == 0:
            return hidden_states
        return tuple(hidden_states for _ in range(self.arity))


class TailInjector(nn.Module):
    """Last-layer-slot stand-in that returns a cached last-block output.

    The payload is swapped per calibration sample before each mocked pass;
    the forward ignores its inputs entirely and returns the cached rows with
    their original dtype and device preserved.
    """

    def __init__(self, tail: torch.Tensor, arity: int = 1):
        super().__init__()
        self.arity = arity
        self.tail = tail

    def forward(self, *args, **kwargs):
        if self.arity == 0:
            return self.tail
        return tuple(self.tail for _ in range(self.arity))


class CaptureHead(nn.Module):
    """Language-head stand-in that records its input rows.

    Records the full-sequence input tensor (detached, on the host) for each
    call and returns a tiny dummy ``[batch, 1, out_features]`` zero tensor so
    the head GEMM and the full-vocab logits stay out of the mocked pass;
    post-head wrapper ops (float casts, softcap) run harmlessly on the dummy.
    Collection hooks never fire here: they are attached to the original head
    module, not to this stand-in.
    """

    def __init__(self, out_features: int, original: nn.Module = None):
        super().__init__()
        self.out_features = out_features
        # kept as a plain attribute (not a submodule) so it never fires
        self._capture_original = original
        self.records: List[torch.Tensor] = []

    def __getattr__(self, name):
        # models may probe head attributes during forward (e.g. mamba reads
        # ``self.lm_head.weight.dtype`` for its pre-head cast): delegate to the
        # real head after the normal nn.Module lookup fails. The original is
        # stored via nn.Module's setattr (it lands in ``_modules``).
        try:
            return super().__getattr__(name)
        except AttributeError:
            original = self._modules.get("_capture_original") if "_modules" in self.__dict__ else None
            if original is not None:
                return getattr(original, name)
            raise

    def forward(self, hidden_states):
        self.records.append(hidden_states.detach().to("cpu"))
        return torch.zeros(
            hidden_states.shape[0],
            1,
            self.out_features,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )


def install_block_stubs_(
    model: nn.Module,
    block_names: List[str],
    arity: int = 1,
) -> Dict[str, object]:
    """Replace each named block module with a :class:`PassthroughStub`.

    Returns the restore bookkeeping for :func:`restore_blocks_`. The original
    modules are kept alive by the returned mapping, so streaming/meta state
    underneath them survives the temporary swap.
    """
    restore_info: Dict[str, object] = {"slots": [], "arity": arity}
    try:
        for name in block_names:
            parent_path, _, attr = name.rpartition(".")
            parent = get_module(model, parent_path) if parent_path else model
            if parent is None or not hasattr(parent, attr):
                raise ValueError(f"cannot install a stub for unknown block '{name}'")
            original = getattr(parent, attr)
            setattr(parent, attr, PassthroughStub(arity=arity))
            restore_info["slots"].append((parent, attr, original))
    except Exception:
        # roll back partial installs so a failed call leaves the model untouched
        # (otherwise a later retry could record a stub as the "original")
        restore_blocks_(model, restore_info)
        raise
    return restore_info


def restore_blocks_(model: nn.Module, restore_info: Union[Dict[str, object], None]) -> None:
    """Put the original block modules back after a mocked pass.

    Restores by ``(parent, attr)`` identity rather than by name so intermediate
    slot changes (the last slot holding a :class:`TailInjector` during the
    capture pass) are irrelevant.
    """
    if not restore_info:
        return
    for parent, attr, original in restore_info["slots"]:
        setattr(parent, attr, original)


def _smoke_logits_(output) -> Union[torch.Tensor, None]:
    """Best-effort logits extraction from a model forward output."""
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def tail_smoke_check(
    model: nn.Module,
    lm_head_name: str,
    block_names: List[str],
    seq_len: int = 2,
) -> Tuple[bool, Union[int, None]]:
    """Init-time smoke gate for the mocked-continuation lane.

    Temporarily replaces the named blocks with pass-through stubs and runs one
    tiny forward with the REAL language head. The model's own pre-loop and
    post-block code executes on stub outputs, which validates everything the
    lane depends on: text-only callability, the stub return arity the layer
    loop accepts, and the post-block chain. Arity 1 (single-element tuple) is tried first, then 2
    (residual-stream pairs, e.g. zaya), then the bare tensor.

    Returns ``(ok, arity)``; ``(False, None)`` means the conservative
    capture walk should feed the head instead. The original block modules are
    always restored, and the training mode is preserved.
    """
    head = get_module(model, lm_head_name) if lm_head_name else None
    vocab = getattr(head, "out_features", None) if head is not None else None
    if vocab is None:
        return (False, None)
    device = None
    try:
        embeddings = model.get_input_embeddings()
        device = embeddings.weight.device if embeddings is not None else None
    except Exception:
        device = None
    if device is None or device.type == "meta":
        try:
            device = next(p.device for p in model.parameters() if p.device.type != "meta")
        except StopIteration:
            return (False, None)
    input_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    was_training = model.training
    model.eval()
    try:
        for arity in (1, 2, 0):
            restore_info = None
            try:
                restore_info = install_block_stubs_(model, block_names, arity=arity)
                with torch.no_grad():
                    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = _smoke_logits_(output)
                if logits is not None and logits.dim() == 3 and logits.shape[0] == 1 and logits.shape[-1] == vocab:
                    return (True, arity)
            except Exception:  # any family-specific failure: try the next arity
                continue
            finally:
                restore_blocks_(model, restore_info)
    finally:
        model.train(was_training)
    return (False, None)
