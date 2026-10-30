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

"""Contract for format-owned export implementations."""

import ast
import importlib
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("name", ["gptq", "awq", "autoround", "compressed_tensors", "gguf", "mlx"])
def test_format_owns_implementation(name):
    module = importlib.import_module(f"auto_round.export.formats.{name}.format")
    assert module.__file__


def test_old_exporter_trees_are_removed():
    import auto_round.export

    root = Path(auto_round.export.__file__).parent
    assert not list(root.glob("export_to_*"))
    assert not (root / "formats" / "backends").exists()


def test_custom_format_implements_only_unified_methods(tmp_path):
    from auto_round.compressors.config_resolution import resolve_scheme_value
    from auto_round.export import OutputFormat, resolve_formats

    name = "test_unified_interface"

    @OutputFormat.register(name)
    class MarkerFormat(OutputFormat):
        def pack_layer(self, layer_name, model, device=None, **kwargs):
            return None

        def save_quantized(self, output_dir, model=None, **kwargs):
            Path(output_dir, "format.txt").write_text(self.output_format)
            return model

    try:
        (fmt,) = resolve_formats(resolve_scheme_value("W4A16", {}), format=name).formats
        model = torch.nn.Linear(4, 4)
        assert fmt.save_quantized(tmp_path, model=model) is model
        assert (tmp_path / "format.txt").read_text() == name
    finally:
        OutputFormat._format_list.pop(name, None)


def test_source_does_not_import_removed_exporters():
    import auto_round

    root = Path(auto_round.__file__).parent.parent
    for package in ("auto_round", "auto_round_extension"):
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                assert not any("auto_round.export.export_to_" in module for module in modules), path


def test_support_matrix_loads_builtins_in_fresh_process():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from auto_round.export import OutputFormat; "
                "matrix = OutputFormat.get_support_matrix(); "
                "assert 'auto_gptq' in matrix; assert 'gguf' in matrix; assert 'mlx' in matrix"
            ),
        ],
        check=True,
        timeout=60,
    )
