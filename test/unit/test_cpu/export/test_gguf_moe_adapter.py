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

import gguf
import numpy as np
import pytest
import torch
from torch import nn

from auto_round.export.export_to_gguf import convert
from auto_round.export.export_to_gguf.hf_checkpoint_restorer import RestoredTensor
from auto_round.export.export_to_gguf.moe_adapter import (
    GGUFMoEOutput,
    pack_moe_output,
    resolve_moe_output,
    validate_moe_imatrices,
    validate_moe_source_qtypes,
)
from auto_round.modeling.fused_moe.fusion_spec import MoETensorSource


class _FakeConverter:
    _TENSOR_NAMES = {
        gguf.MODEL_TENSOR.FFN_GATE_EXP: "blk.0.ffn_gate_exps.weight",
        gguf.MODEL_TENSOR.FFN_UP_EXP: "blk.0.ffn_up_exps.weight",
        gguf.MODEL_TENSOR.FFN_DOWN_EXP: "blk.0.ffn_down_exps.weight",
    }

    @classmethod
    def match_model_tensor_name(cls, name, tensor_type, bid):
        assert bid == 0
        return name == cls._TENSOR_NAMES[tensor_type]


def _restored(*sources):
    return RestoredTensor(
        checkpoint_name="model.layers.0.mlp.experts.gate_up_proj",
        tensor_fn=lambda: torch.empty(0),
        hf_names=tuple(name for source in sources for name in source.hf_names),
        transform_kind="moe_refusion",
        moe_sources=tuple(sources),
    )


GATE_NAMES = (
    "model.layers.0.experts.0.gate_proj.weight",
    "model.layers.0.experts.1.gate_proj.weight",
)
UP_NAMES = (
    "model.layers.0.experts.0.up_proj.weight",
    "model.layers.0.experts.1.up_proj.weight",
)


@pytest.mark.parametrize(
    ("new_name", "projection", "hf_names"),
    [
        ("blk.0.ffn_gate_exps.weight", "gate_proj", GATE_NAMES),
        ("blk.0.ffn_up_exps.weight", "up_proj", UP_NAMES),
    ],
)
def test_resolve_moe_output_selects_only_matching_source(new_name, projection, hf_names):
    restored = _restored(
        MoETensorSource("gate_proj", GATE_NAMES),
        MoETensorSource("up_proj", UP_NAMES),
    )

    output = resolve_moe_output(_FakeConverter, restored, new_name, bid=0)

    assert output == GGUFMoEOutput(projection=projection, hf_names=hf_names)


def test_resolve_moe_output_rejects_ambiguous_projection_source():
    restored = _restored(
        MoETensorSource("gate_proj", GATE_NAMES),
        MoETensorSource("w1", UP_NAMES),
    )

    with pytest.raises(ValueError, match="cannot map GGUF MoE output"):
        resolve_moe_output(_FakeConverter, restored, "blk.0.ffn_gate_exps.weight", bid=0)


class _ExpertModule(nn.Module):
    def __init__(self, scale, imatrix):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 3))
        self.scale = scale
        if imatrix is not None:
            self.imatrix = imatrix


def _expert_modules():
    return {
        GATE_NAMES[0]: _ExpertModule(torch.tensor([1.0]), torch.tensor([1.0, 2.0, 3.0])),
        GATE_NAMES[1]: _ExpertModule(torch.tensor([2.0]), torch.tensor([4.0, 5.0, 6.0])),
    }


def test_pack_moe_output_quantizes_each_expert_with_its_source_metadata():
    modules = _expert_modules()
    data = torch.stack([torch.full((2, 3), 1.0), torch.full((2, 3), 2.0)])
    calls = []

    def quantize_fn(expert_tensor, hf_name):
        module = modules[hf_name]
        calls.append((expert_tensor, hf_name, module, module.scale, module.imatrix))
        return np.full((2, 2), int(expert_tensor[0, 0].item()), dtype=np.uint8), gguf.GGMLQuantizationType.Q4_0

    packed, qtype = pack_moe_output(
        data,
        gguf.GGMLQuantizationType.Q4_0,
        GGUFMoEOutput("gate_proj", GATE_NAMES),
        quantize_fn,
        "block 0",
    )

    reference = np.stack([np.full((2, 2), 1, dtype=np.uint8), np.full((2, 2), 2, dtype=np.uint8)], axis=0)
    assert np.array_equal(packed, reference)
    assert qtype == gguf.GGMLQuantizationType.Q4_0
    assert [call[1] for call in calls] == list(GATE_NAMES)
    assert calls[0][2] is modules[GATE_NAMES[0]]
    assert calls[1][2] is modules[GATE_NAMES[1]]


def test_pack_moe_output_rejects_expert_count_mismatch():
    with pytest.raises(ValueError, match="block 0.*projection gate_proj.*expert 1"):
        pack_moe_output(
            torch.zeros(1, 2, 3),
            gguf.GGMLQuantizationType.Q4_0,
            GGUFMoEOutput("gate_proj", GATE_NAMES),
            lambda *_: pytest.fail("callback must not run"),
            "block 0",
        )


def test_pack_spec_moe_output_routes_each_expert_to_its_source(monkeypatch):
    restored = _restored(MoETensorSource("gate_proj", GATE_NAMES))
    moe_output = resolve_moe_output(_FakeConverter, restored, "blk.0.ffn_gate_exps.weight", bid=0)
    data = torch.stack([torch.full((2, 3), 1.0), torch.full((2, 3), 2.0)])
    calls = []

    def quant_data(cls, arr, data_qtype, name, modify_name, new_name, bid, device=None, use_layer_attrs=True):
        calls.append((arr, name, use_layer_attrs))
        return np.full((2, 2), len(calls), dtype=np.uint8), data_qtype

    monkeypatch.setattr(convert, "_quant_data", quant_data)

    packed, qtype = convert._pack_spec_moe_output(
        _FakeConverter,
        data,
        gguf.GGMLQuantizationType.Q4_0,
        moe_output,
        "model.layers.0.mlp.experts.gate_up_proj",
        "blk.0.ffn_gate_exps.weight",
        0,
        device="cpu",
        context="checkpoint -> blk.0.ffn_gate_exps.weight",
    )

    assert [call[1] for call in calls] == list(GATE_NAMES)
    assert all(call[2] is True for call in calls)
    assert np.array_equal(packed[:, 0, 0], np.array([1, 2], dtype=np.uint8))
    assert qtype == gguf.GGMLQuantizationType.Q4_0


def test_prepare_tensors_cleans_selected_moe_sources_and_marks_full_lineage(monkeypatch):
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 32))
            self.scale = torch.ones(2, 1)
            self.zp = torch.ones(2, 1)
            self.w_d_scale = torch.ones(2, 1)
            self.w_d_wmin = torch.ones(2, 1)
            self.w_wmin = torch.ones(2, 1)
            self.imatrix = torch.ones(32)

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].experts = nn.ModuleList([nn.Module(), nn.Module()])
    for expert in model.model.layers[0].experts:
        expert.gate_proj = Expert()
        expert.up_proj = Expert()

    restored = RestoredTensor(
        checkpoint_name="model.layers.0.mlp.experts.gate_up_proj",
        tensor_fn=lambda: torch.ones(2, 4, 32),
        hf_names=GATE_NAMES + UP_NAMES,
        transform_kind="moe_refusion",
        moe_sources=(
            MoETensorSource("gate_proj", GATE_NAMES),
            MoETensorSource("up_proj", UP_NAMES),
        ),
    )

    class Writer:
        def __init__(self):
            self.tensors = []

        def add_tensor(self, name, data, raw_dtype):
            self.tensors.append((name, data, raw_dtype))

    class Converter(_FakeConverter):
        device = "cpu"
        _gguf_dtype_selector = object()
        _gguf_name_resolution_diagnostics = []
        tensor_map = type("TensorMap", (), {"mapping": {}})()
        model_arch = gguf.MODEL_ARCH.LLAMA
        current_packing_block = "model.layers.0"
        quant_nontext_module = True
        low_cpu_mem_usage = True
        ftype = gguf.LlamaFileType.MOSTLY_Q4_0

        def __init__(self):
            self.model = model
            self.layer_config = {name.removesuffix(".weight"): {"bits": 4, "sym": True} for name in restored.hf_names}
            self.gguf_writer = Writer()
            self._gguf_written_checkpoint_names = set()
            self._gguf_written_hf_names = set()

        def generate_extra_tensors(self):
            return ()

        def get_tensors(self):
            return (restored,)

        def filter_tensors(self, item):
            return item

        def modify_tensors(self, data, modify_name, bid):
            yield "blk.0.ffn_gate_exps.weight", data[:, :2]
            yield "blk.0.ffn_up_exps.weight", data[:, 2:]

        def tensor_force_quant(self, checkpoint_name, new_name, bid, n_dims):
            return gguf.GGMLQuantizationType.Q4_0

    routed_sources = []

    def pack_spec_moe_output(cls, data, qtype, output, *args, **kwargs):
        routed_sources.append(output.hf_names)
        return np.zeros(data.shape, dtype=np.float32), qtype

    monkeypatch.setattr(convert, "_pack_spec_moe_output", pack_spec_moe_output)
    converter = Converter()

    convert.prepare_tensors(converter)

    assert routed_sources == [GATE_NAMES, UP_NAMES]
    assert converter._gguf_written_checkpoint_names == {restored.checkpoint_name}
    assert converter._gguf_written_hf_names == set(restored.hf_names)
    for source_name in restored.hf_names:
        module = convert.get_module(model, source_name.removesuffix(".weight"))
        assert module.weight.numel() == 0
        assert module.scale is None
        assert module.imatrix is None


def test_validate_moe_source_qtypes_returns_uniform_explicit_qtype():
    qtype = validate_moe_source_qtypes(
        GATE_NAMES,
        gguf.GGMLQuantizationType.Q4_0,
        lambda _: gguf.GGMLQuantizationType.Q5_0,
        "block 0",
    )

    assert qtype == gguf.GGMLQuantizationType.Q5_0


@pytest.mark.parametrize(
    "explicit",
    [
        [gguf.GGMLQuantizationType.Q4_0, None],
        [gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q5_0],
    ],
)
def test_validate_moe_source_qtypes_rejects_missing_or_inconsistent_values(explicit):
    by_name = dict(zip(GATE_NAMES, explicit))

    with pytest.raises(ValueError, match="MoE source qtypes are missing or inconsistent"):
        validate_moe_source_qtypes(
            GATE_NAMES,
            gguf.GGMLQuantizationType.Q4_0,
            by_name.get,
            "block 0",
        )


def test_validate_moe_imatrices_rejects_partial_missing():
    modules = _expert_modules()
    del modules[GATE_NAMES[1]].imatrix

    with pytest.raises(ValueError, match="block 0.*projection gate_proj.*expert 1"):
        validate_moe_imatrices(GATE_NAMES, modules.__getitem__, "block 0, projection gate_proj")
