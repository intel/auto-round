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
"""Tests for the shared harness helpers in test/helpers.py."""

from test.helpers import noop_torch_compile


def test_noop_torch_compile_supports_bare_decorator():
    @noop_torch_compile
    def fn(x):
        return x + 1

    assert fn(1) == 2


def test_noop_torch_compile_supports_decorator_factory():
    """Regression for #2679: third-party modules import with ``@torch.compile(dynamic=True)``."""

    @noop_torch_compile(dynamic=True)
    def fn(x):
        return x + 1

    assert fn(1) == 2


def test_noop_torch_compile_supports_direct_call():
    def fn(x):
        return x

    assert noop_torch_compile(fn) is fn
    assert noop_torch_compile(fn, dynamic=True) is fn
    assert noop_torch_compile(model=fn) is fn
