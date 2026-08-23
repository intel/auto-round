# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.export.export_to_mlx.__init__``."""

from auto_round.export.export_to_mlx import __all__, pack_layer, save_quantized_as_mlx


class TestMlxInitExports:
    """Verify __all__ and lazy import surface."""

    def test_pack_layer_in_all(self):
        assert "pack_layer" in __all__

    def test_save_quantized_as_mlx_in_all(self):
        assert "save_quantized_as_mlx" in __all__

    def test_pack_layer_callable(self):
        assert callable(pack_layer)

    def test_save_quantized_as_mlx_callable(self):
        assert callable(save_quantized_as_mlx)
