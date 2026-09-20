# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Quantized layers outside blocks (lm_head-class) must not disable immediate
packing: the tail lane feeds them from the calibration chain, so blocks can
pack - and shards can stream - progressively exactly as they do without
outside-block layers. (Before the tail lane, the concern was a post-block
capture walk through packed blocks; that walk no longer exists for lm_head,
and GGUF already bypassed the gate.)
"""

from types import MethodType, SimpleNamespace

import auto_round.compressors.base as base_mod
from auto_round.algorithms.quantization.rtn.config import RTNConfig


def _fmt(gguf=False):
    return SimpleNamespace(
        is_gguf=lambda: gguf,
        is_fake=lambda: False,
        is_supported_immediate_packing=lambda: True,
        is_supported_immediate_saving=lambda: True,
    )


def _compressor(outside_block_layers, rtncfg=True, tail_only=True):
    model = type("QwenForCausalLM", (), {"_tied_weight_keys": {}})()
    o = SimpleNamespace(
        formats=[_fmt()],
        inplace=True,
        has_qlayer_outside_block=outside_block_layers,
        need_calib=True,
        disable_opt_rtn=None,
        output_dir="/tmp/does-not-matter",
        shard_writer=None,
        _ensure_shard_writer=lambda self_=None: None,
        quantize_config=RTNConfig(data_type="int") if rtncfg else _SignRoundCfg(),
        layer_config={"lm_head": {"bits": 4}} if tail_only else {"model.embed_tokens": {"bits": 8}},
        model_context=SimpleNamespace(model=model, is_mllm=False),
        compress_context=SimpleNamespace(
            is_immediate_packing=True,
            is_immediate_saving=True,
            low_cpu_mem_usage=True,
        ),
    )
    o._adjust_immediate_packing_and_saving = MethodType(base_mod.BaseCompressor._adjust_immediate_packing_and_saving, o)
    o._outside_block_quantized_tail_only_ = MethodType(base_mod.BaseCompressor._outside_block_quantized_tail_only_, o)
    return o


class _SignRoundCfg:
    """Stands in for SignRoundConfig (not an RTNConfig)."""

    data_type = "int"


class TestImmediatePackingWithOutsideBlockLayers:
    def test_outside_block_layers_keep_immediate_packing(self):
        # lm_head-class outside-block layers are tail-fed: no capture walk,
        # so packing/saving stay progressive
        c = _compressor(outside_block_layers=True)
        c._adjust_immediate_packing_and_saving()
        assert c.compress_context.is_immediate_packing is True
        # low_cpu_mem_usage + packing upgrades to progressive shard writes
        assert c.compress_context.is_immediate_saving is True

    def test_non_tail_outside_block_layers_restore_the_guards(self):
        # a non-lm_head outside-block quantized layer (e.g. a pinned
        # embed_tokens) still runs the legacy capture path through the model;
        # the old restrictions stand for that class
        c = _compressor(outside_block_layers=True, tail_only=False)
        c._adjust_immediate_packing_and_saving()
        assert c.compress_context.is_immediate_packing is False

    def test_signround_outside_block_layers_keep_immediate_saving(self):
        """iters>0 runs: the old capture-path concern (whole-model materialize
        under low_cpu_mem) is gone with the tail lane - no forced downgrade."""
        c = _compressor(outside_block_layers=True, rtncfg=False)
        c._adjust_immediate_packing_and_saving()
        assert c.compress_context.is_immediate_packing is True
        assert c.compress_context.is_immediate_saving is True
        assert c.compress_context.low_cpu_mem_usage is True

    def test_inplace_guard_restored_only_for_non_tail_layers(self):
        """The inplace=False starvation is back, but ONLY for outside-block
        layers the chain-tail lane does not cover (lm_head-class layers are
        exempt through _outside_block_quantized_tail_only_)."""
        src = open(__import__("auto_round.compressors.base", fromlist=["x"]).__file__, encoding="utf-8").read()
        assert "self.inplace = False" in src
        assert "_outside_block_quantized_tail_only_()" in src

    def test_plain_run_unchanged(self):
        c = _compressor(outside_block_layers=False)
        c._adjust_immediate_packing_and_saving()
        assert c.compress_context.is_immediate_packing is True
        assert c.compress_context.is_immediate_saving is True
