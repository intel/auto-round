# Copyright (c) 2024 Intel Corporation
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

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Iterator

import torch

from auto_round.utils import logger


@dataclass(frozen=True)
class RestoredTensor:
    checkpoint_name: str
    tensor_fn: Callable[[], torch.Tensor]
    hf_names: tuple[str, ...]
    transform_kind: str


class HFCheckpointRestorer:
    """Expose a Transformers model state dict as checkpoint-style tensors.

    Transformers may rename or structurally convert checkpoint tensors while loading
    a model. GGUF conversion expects the original checkpoint namespace, while
    AutoRound layer config is keyed by live HF module names. This restorer keeps
    both names available.
    """

    _SUPPORTED_OPS = {"Chunk", "Concatenate", "MergeModulelist", "SplitModulelist"}

    def __init__(self, model, completed_hf_names=None):
        self.model = model
        self.completed_hf_names = completed_hf_names or set()

    def iter_tensors(self) -> Iterator[RestoredTensor]:
        state_dict = self.model.state_dict()
        weight_conversions = self._get_weight_conversions()
        if not weight_conversions:
            for name, tensor in state_dict.items():
                if name in self.completed_hf_names:
                    continue
                yield RestoredTensor(name, lambda tensor=tensor: tensor, (name,), "passthrough")
            return

        inverted_transforms = self._get_inverted_transforms(weight_conversions)
        inverted_converters = [transform for transform in inverted_transforms if self._is_weight_converter(transform)]
        inverted_renamings = [
            transform for transform in inverted_transforms if not self._is_weight_converter(transform)
        ]
        pattern_to_converter = {
            pattern: converter for converter in inverted_converters for pattern in converter.source_patterns
        }

        conversion_mapping = {}
        converter_lineage = defaultdict(list)
        converter_fallback_tensors = defaultdict(list)
        converter_order = {}
        ordered_tensors = []

        for source_order, (original_key, tensor) in enumerate(state_dict.items()):
            converter_key, matched_pattern = self._rename_source_key(original_key, [], inverted_converters)
            checkpoint_key, _ = self._rename_source_key(converter_key, inverted_renamings, [])

            if matched_pattern is None:
                if original_key in self.completed_hf_names:
                    continue
                transform_kind = "rename" if checkpoint_key != original_key else "passthrough"
                ordered_tensors.append(
                    (
                        source_order,
                        RestoredTensor(
                            checkpoint_key,
                            lambda tensor=tensor: tensor,
                            (original_key,),
                            transform_kind,
                        ),
                    )
                )
                continue

            mapping = conversion_mapping.setdefault(converter_key, deepcopy(pattern_to_converter[matched_pattern]))
            mapping.add_tensor(checkpoint_key, original_key, matched_pattern, tensor)
            converter_lineage[converter_key].append(original_key)
            converter_order.setdefault(converter_key, source_order)
            converter_fallback_tensors[converter_key].append(
                RestoredTensor(checkpoint_key, lambda tensor=tensor: tensor, (original_key,), "passthrough")
            )

        for layer_name, mapping in sorted(conversion_mapping.items()):
            if all(name in self.completed_hf_names for name in converter_lineage[layer_name]):
                continue
            try:
                realized = mapping.convert(layer_name, model=self.model, config=getattr(self.model, "config", None))
            except Exception as exc:
                logger.debug(
                    "gguf: falling back to original tensors for failed Transformers weight conversion %s: %s",
                    layer_name,
                    exc,
                )
                ordered_tensors.extend(
                    (converter_order[layer_name], tensor) for tensor in converter_fallback_tensors[layer_name]
                )
                continue
            normalized_realized = []
            for target_name, tensor in sorted(realized.items()):
                if self._is_weight_converter(mapping):
                    target_name, _ = self._rename_source_key(target_name, inverted_renamings, [])
                tensor = tensor[0] if isinstance(tensor, list) else tensor
                if tensor is None or tensor.numel() == 0:
                    logger.debug(
                        "gguf: falling back to original tensors for empty Transformers weight conversion %s",
                        layer_name,
                    )
                    ordered_tensors.extend(
                        (converter_order[layer_name], fallback) for fallback in converter_fallback_tensors[layer_name]
                    )
                    break
                normalized_realized.append((target_name, tensor))
            else:
                lineage = tuple(dict.fromkeys(converter_lineage[layer_name]))
                for target_name, tensor in normalized_realized:
                    ordered_tensors.append(
                        (
                            converter_order[layer_name],
                            RestoredTensor(
                                target_name,
                                lambda tensor=tensor: tensor,
                                lineage,
                                "converter",
                            ),
                        )
                    )

        for _, restored in sorted(ordered_tensors, key=lambda item: item[0]):
            yield restored

    def _get_weight_conversions(self):
        weight_conversions = getattr(self.model, "_weight_conversions", None)
        if weight_conversions is not None:
            return weight_conversions

        try:
            from transformers.conversion_mapping import get_model_conversion_mapping
            from transformers.core_model_loading import PrefixChange
        except Exception:
            return None

        try:
            conversions = get_model_conversion_mapping(self.model, add_legacy=False)
        except Exception as exc:
            logger.warning("gguf: unable to get Transformers weight conversion mapping: %s", exc)
            return None

        # Match Transformers' own fallback in revert_weight_conversion: if the
        # model was not loaded through from_pretrained, do not invent prefix adds.
        conversions = [conversion for conversion in conversions if not isinstance(conversion, PrefixChange)]
        return conversions or None

    def _get_inverted_transforms(self, weight_conversions):
        inverted = []
        for transform in weight_conversions[::-1]:
            try:
                reversed_transform = transform.reverse_transform()
            except Exception as exc:
                logger.warning("gguf: skipping non-reversible Transformers weight transform %s: %s", transform, exc)
                continue
            if self._is_weight_converter(reversed_transform) and not self._is_supported_converter(reversed_transform):
                logger.warning("gguf: skipping unsupported Transformers weight converter %s", reversed_transform)
                continue
            inverted.append(reversed_transform)
        return inverted

    def _is_supported_converter(self, converter):
        return all(type(op).__name__ in self._SUPPORTED_OPS for op in getattr(converter, "operations", ()))

    @staticmethod
    def _is_weight_converter(transform):
        try:
            from transformers.core_model_loading import WeightConverter
        except Exception:
            return False
        return isinstance(transform, WeightConverter)

    @staticmethod
    def _rename_source_key(source_key, weight_renamings, weight_converters):
        from transformers.core_model_loading import rename_source_key

        return rename_source_key(source_key, weight_renamings, weight_converters)
