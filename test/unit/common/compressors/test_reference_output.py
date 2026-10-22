# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests: compress_block skips the collect_reference forward pass when
reference_output is pre-seeded (intel/auto-round#2288)."""

from unittest.mock import MagicMock

import torch

from auto_round import RTNConfig
from auto_round.algorithms.composer import AlgorithmComposer, BlockContext
from auto_round.algorithms.transforms.base import BasePreprocessor


def _make_ctx():
    return BlockContext(
        model=None,
        block_names=["model.layers.0"],
        block_name="model.layers.0",
        block_index=0,
    )


def _patch_composer(composer, fp_inputs):
    """Replace heavy internals with lightweight no-ops."""
    mock_forward = MagicMock(return_value=fp_inputs)
    mock_forward.last_output_dict = None
    mock_forward.enable_torch_compile = False
    composer.block_forward = mock_forward
    composer.block_quantizer.quantize_block = MagicMock()
    composer.block_quantizer.enable_quanted_input = False
    composer._get_fp_act_hooks = MagicMock(return_value=[])
    composer._get_q_act_hooks = MagicMock(return_value=[])
    composer.preprocessors = []
    return mock_forward


class TestCompressBlockReferenceOutputSkip:
    """Verify that a pre-seeded reference_output bypasses the Step-3 forward pass."""

    def setup_method(self):
        self.composer = AlgorithmComposer([RTNConfig()])
        self.ctx = _make_ctx()
        self.block = torch.nn.Linear(4, 4)
        self.fp_inputs = [torch.zeros(1, 4)]
        self.input_others = {}

    def test_forward_skipped_when_reference_output_provided(self):
        """block_forward must NOT be called in Step 3 when reference_output is given."""
        pre_computed = [torch.ones(1, 4)]
        mock_forward = _patch_composer(self.composer, self.fp_inputs)

        self.composer.compress_block(
            self.block,
            self.fp_inputs,
            self.input_others,
            block_ctx=self.ctx,
            reference_output=pre_computed,
        )

        mock_forward.assert_not_called()

    def test_forward_called_when_reference_output_is_none(self):
        """block_forward MUST be called in Step 3 when reference_output=None (default)."""
        mock_forward = _patch_composer(self.composer, self.fp_inputs)

        self.composer.compress_block(
            self.block,
            self.fp_inputs,
            self.input_others,
            block_ctx=self.ctx,
            reference_output=None,
        )

        mock_forward.assert_called_once()

    def test_seeded_reference_output_propagates_to_return_value(self):
        """The returned reference_next_input must be the pre-seeded tensors, not a
        freshly computed one, so the caller's cache and the quantizer see the same object."""
        pre_computed = [torch.ones(1, 4) * 7.0]
        _patch_composer(self.composer, self.fp_inputs)

        _, ref_next = self.composer.compress_block(
            self.block,
            self.fp_inputs,
            self.input_others,
            block_ctx=self.ctx,
            reference_output=pre_computed,
        )

        assert ref_next is pre_computed


class _WeightChangingPreprocessor(BasePreprocessor):
    """Minimal transform used to exercise reference invalidation."""

    def register_fp_input_forward_hooks(self, block):
        return []

    def pre_quantize_block(self, ctx):
        ctx.model.weight.data.mul_(2)

    def invalidates_reference_output(self, ctx):
        return True


def test_weight_changing_preprocessor_recomputes_seeded_reference():
    """A pre-transform reference must not survive a weight-changing preprocessor."""
    from auto_round.algorithms.block_runner import BlockForwardRunner

    composer = AlgorithmComposer([RTNConfig()])
    block = torch.nn.Linear(1, 1, bias=False)
    block.weight.data.fill_(1)
    fp_inputs = [torch.tensor([[3.0]])]
    stale_reference = [torch.tensor([[3.0]])]
    ctx = BlockContext(
        model=block,
        block_names=["block"],
        block_name="block",
        block_index=0,
    )
    composer.block_forward = BlockForwardRunner(batch_size=1, amp=False, enable_torch_compile=False)
    composer.block_quantizer.quantize_block = MagicMock()
    composer.block_quantizer.enable_quanted_input = False
    composer._get_fp_act_hooks = MagicMock(return_value=[])
    composer._get_q_act_hooks = MagicMock(return_value=[])
    composer.preprocessors = [_WeightChangingPreprocessor()]

    _, reference_next_input = composer.compress_block(
        block,
        fp_inputs,
        {},
        block_ctx=ctx,
        reference_output=stale_reference,
    )

    expected = torch.tensor([[6.0]])
    assert torch.equal(reference_next_input[0], expected)
    quantizer_reference = composer.block_quantizer.quantize_block.call_args.args[3]
    assert torch.equal(quantizer_reference[0], expected)


def test_reference_propagation_ignores_stale_last_output_dict():
    """A skipped reference forward cannot reuse metadata from preprocessor calibration."""
    composer = AlgorithmComposer([RTNConfig()])
    fp_inputs = [torch.zeros(1, 4)]
    pre_computed = [torch.ones(1, 4)]
    mock_forward = _patch_composer(composer, fp_inputs)
    mock_forward.last_output_dict = {"hidden_states": [torch.full((1, 4), 9.0)]}

    _, reference_next_input = composer.compress_block(
        torch.nn.Linear(4, 4),
        fp_inputs,
        {},
        block_ctx=_make_ctx(),
        reference_output=pre_computed,
    )

    assert reference_next_input is pre_computed
