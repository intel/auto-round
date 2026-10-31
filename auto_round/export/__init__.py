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

"""Unified output-format API, loaded lazily to keep configuration imports light."""

__all__ = ["BackendDataType", "OutputFormat", "register_format", "resolve_formats"]


def __getattr__(name):
    if name in {"OutputFormat", "BackendDataType"}:
        from auto_round.export import base

        return getattr(base, name)
    if name == "register_format":
        from auto_round.export.registry import register_format

        return register_format
    if name == "resolve_formats":
        from auto_round.export.resolver import resolve_formats

        return resolve_formats
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
