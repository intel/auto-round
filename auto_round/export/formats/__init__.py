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

"""Format-owned implementations; configuration submodules can be imported independently."""


def __getattr__(name):
    if name in {"OutputFormat", "BackendDataType", "resolve_formats"}:
        import auto_round.export as export

        return getattr(export, name)
    from auto_round.export.registry import FORMAT_REGISTRY, load_builtin_formats

    load_builtin_formats()
    for format_class in FORMAT_REGISTRY.values():
        if format_class.__name__ == name:
            return format_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
