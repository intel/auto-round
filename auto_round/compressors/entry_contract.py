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

from types import MappingProxyType
from typing import Any, Mapping

from auto_round.logger import logger

ENTRY_KWARG_OWNERS: Mapping[str, str] = MappingProxyType(
    {
        "model_free": "route",
        "disable_model_free": "route",
        "disable_opt_rtn": "route",
        "scale_dtype": "compressor",
        "ignore_layers": "compressor",
        "quant_lm_head": "compressor",
        "to_quant_block_names": "compressor",
        "format": "base",
        "dataset": "base",
        "batch_size": "base",
        "model_dtype": "base",
        "trust_remote_code": "base",
        "amp": "base",
        "nblocks": "base",
        "disable_deterministic_algorithms": "base",
        "enable_deterministic_algorithms": "base",
        "static_kv_dtype": "base",
        "static_attention_dtype": "base",
        "processor": "mllm",
        "image_processor": "mllm",
        "template": "mllm",
        "extra_data_dir": "mllm",
        "quant_nontext_module": "mllm",
        "guidance_scale": "diffusion",
        "num_inference_steps": "diffusion",
        "generator_seed": "diffusion",
    }
)
ENTRY_ALLOWED_KWARGS = frozenset(ENTRY_KWARG_OWNERS)


def filter_supported_entry_kwargs(kwargs: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    """Return entry-owned keyword arguments and warn about ignored values."""
    supported = {key: value for key, value in kwargs.items() if key in ENTRY_KWARG_OWNERS}
    unknown = sorted(set(kwargs) - ENTRY_KWARG_OWNERS.keys())
    if unknown:
        logger.warning_once(
            "%s received unsupported kwargs %s. They will be ignored.",
            context,
            ", ".join(unknown),
        )
    return supported


def split_entry_kwargs(kwargs: Mapping[str, Any], *, context: str = "AutoRound entry") -> dict[str, dict[str, Any]]:
    """Partition supported entry kwargs by their owning constructor boundary."""
    buckets: dict[str, dict[str, Any]] = {
        "route": {},
        "compressor": {},
        "base": {},
        "mllm": {},
        "diffusion": {},
    }
    for key, value in filter_supported_entry_kwargs(kwargs, context=context).items():
        buckets[ENTRY_KWARG_OWNERS[key]][key] = value
    return buckets
