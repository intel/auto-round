# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""MLX is exposed through the same interface as every output format."""

from auto_round.export.base import OutputFormat
from auto_round.export.formats.mlx.format import MLXFormat


def test_mlx_implements_unified_interface():
    assert issubclass(MLXFormat, OutputFormat)
    assert callable(MLXFormat.pack_layer)
    assert callable(MLXFormat.save_quantized)
    assert not MLXFormat.__abstractmethods__
