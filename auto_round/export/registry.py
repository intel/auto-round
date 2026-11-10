# Copyright (c) 2025 Intel Corporation
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

"""One registry for format implementations and public names."""

FORMAT_REGISTRY = {}
NON_PUBLIC_NAMES = set()


def register_format(*names: str, public: bool = True):
    """Register a format class and its aliases without duplicating name lists."""
    if not names:
        raise ValueError("At least one format name is required")

    def register(format_class):
        for name in names:
            FORMAT_REGISTRY[name] = format_class
            if public:
                NON_PUBLIC_NAMES.discard(name)
            else:
                NON_PUBLIC_NAMES.add(name)
        return format_class

    return register


def load_builtin_formats():
    """Load implementations lazily, after utils and schemes can be initialized."""
    from importlib import import_module

    for module in (
        "gptq.format",
        "awq.format",
        "mlx.format",
        "compressed_tensors.format",
        "autoround.format",
        "fp8",
        "fake",
        "gguf.format",
        "svdquant.format",
        "svdquant.omni_format",
    ):
        import_module(f"auto_round.export.formats.{module}")


def public_format_names():
    """Return registered public names together with GGUF's configured variants."""
    load_builtin_formats()
    from auto_round.export.formats.gguf.config import GGUF_CONFIG

    return tuple(sorted((FORMAT_REGISTRY.keys() - NON_PUBLIC_NAMES) | GGUF_CONFIG.keys()))
