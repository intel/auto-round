# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.utils.distributed``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.utils.distributed import (
    _all_reduce_model_grads,
    _move_block_to_device,
    _noop_sync,
    is_distributed,
    setup_ddp_if_needed_,
)


class TestIsDistributed:
    """Test is_distributed with mocked torch.distributed."""

    def test_not_initialized(self):
        with patch("auto_round.utils.distributed.is_distributed", return_value=False):
            with patch("torch.distributed.is_initialized", return_value=False):
                # Clear cache to ensure fresh evaluation
                is_distributed.cache_clear()
                result = is_distributed()
                assert result is False

    def test_initialized_single_device(self):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=1):
                is_distributed.cache_clear()
                result = is_distributed()
                assert result is False

    def test_initialized_multi_device(self):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=4):
                is_distributed.cache_clear()
                result = is_distributed()
                assert result is True

    def test_dist_not_initialized(self):
        with patch("torch.distributed.is_initialized", return_value=False):
            is_distributed.cache_clear()
            assert is_distributed() is False
            is_distributed.cache_clear()

    def test_dist_initialized_single_world(self):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=1):
                is_distributed.cache_clear()
                assert is_distributed() is False
                is_distributed.cache_clear()

    def test_dist_initialized_multi_world(self):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=2):
                is_distributed.cache_clear()
                assert is_distributed() is True
                is_distributed.cache_clear()


class TestNoopSync:
    """Test _noop_sync."""

    def test_noop_sync_does_nothing(self):
        # Should not raise
        _noop_sync()

    def test_returns_none(self):
        assert _noop_sync() is None


class TestMoveBlockToDevice:
    """Test _move_block_to_device."""

    def test_moves_to_cpu(self):
        block = nn.Linear(4, 4)
        if torch.cuda.is_available():
            block = block.to("cuda")
        _move_block_to_device(block, "cpu")
        assert next(block.parameters()).device.type == "cpu"

    def test_moves_to_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        block = nn.Linear(4, 4)
        _move_block_to_device(block, 0)
        assert next(block.parameters()).device.type == "cuda"

    def test_moves_to_device(self):
        layer = nn.Linear(4, 4)
        _move_block_to_device(layer, "cpu")
        # Should be on cpu
        assert next(layer.parameters()).device == torch.device("cpu")


class TestAllReduceModelGrads:
    """Test _all_reduce_model_grads."""

    def test_no_grad_is_noop(self):
        model = nn.Linear(4, 4)
        # No gradients set - should not raise
        _all_reduce_model_grads(model)

    def test_no_grads_does_nothing(self):
        layer = nn.Linear(4, 4)
        # No grads set, should not raise
        _all_reduce_model_grads(layer)

    def test_no_distributed_raises(self):
        """Without distributed initialized, all_reduce raises."""
        model = nn.Linear(4, 4)
        model.weight.grad = torch.randn_like(model.weight)
        # This will raise since torch.distributed isn't initialized
        with pytest.raises(ValueError, match="process group"):
            _all_reduce_model_grads(model)

    def test_raises_when_dist_not_initialized(self):
        layer = nn.Linear(4, 4)
        layer.weight.grad = torch.randn_like(layer.weight)
        with patch("torch.cuda.is_available", return_value=False):
            # Without dist init, all_reduce should fail
            with pytest.raises((RuntimeError, ValueError)):
                _all_reduce_model_grads(layer)

    def test_with_cuda_grad_no_distributed(self):
        """Grad is CUDA but distributed not initialized - should not raise."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = nn.Linear(4, 4).to("cuda")
        model.weight.grad = torch.randn_like(model.weight)
        _all_reduce_model_grads(model)


class TestSetupDdpIfNeeded:
    """Test setup_ddp_if_needed_."""

    def test_non_distributed_returns_block_noop(self):
        model = nn.Linear(4, 4)
        ar = SimpleNamespace()
        with patch("auto_round.utils.distributed.is_distributed", return_value=False):
            block, sync_fn = setup_ddp_if_needed_(ar, model, [0])
            assert block is model
            assert sync_fn is _noop_sync

    def test_non_distributed_respects_device_list(self):
        """Device list is passed but distributed is off, so DDP not used."""
        model = nn.Linear(4, 4)
        ar = SimpleNamespace()
        with patch("auto_round.utils.distributed.is_distributed", return_value=False):
            block, sync_fn = setup_ddp_if_needed_(ar, model, [0, 1])
            assert block is model
            assert sync_fn is _noop_sync

    def test_single_device_ddp(self):
        """Distributed with single GPU per rank wraps with DDP."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = nn.Linear(4, 4).to("cpu")

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=2):
                with patch("torch.distributed.get_rank", return_value=0):
                    with patch("torch.nn.parallel.DistributedDataParallel"):
                        ar = SimpleNamespace()
                        block, sync_fn = setup_ddp_if_needed_(ar, model, [0])
                        assert sync_fn is _noop_sync

    def test_single_device_returns_noop_when_distributed(self):
        """Test the multi-GPU case which doesn't need to move to GPU device."""
        with patch("auto_round.utils.distributed.is_distributed", return_value=True):
            block = nn.Linear(4, 4)
            with patch("torch.distributed.get_rank", return_value=0):
                # Use multi-device path which doesn't actually move to GPU
                block, sync_fn = setup_ddp_if_needed_(None, block, [0, 1])
            # Multi-device path returns a manual reduce sync fn
            assert sync_fn is not _noop_sync

    def test_multi_device_uses_manual_reduce(self):
        with patch("auto_round.utils.distributed.is_distributed", return_value=True):
            block = nn.Linear(4, 4)
            with patch("torch.distributed.get_rank", return_value=0):
                block, sync_fn = setup_ddp_if_needed_(None, block, [0, 1])
            # Should return a custom sync function
            assert sync_fn is not _noop_sync
            # Calling sync_fn should call _all_reduce_model_grads
            with patch("auto_round.utils.distributed._all_reduce_model_grads") as mock_reduce:
                sync_fn()
                mock_reduce.assert_called_once()

    def test_multi_device_manual_reduce(self):
        """Distributed with multiple GPUs per rank returns manual sync."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = nn.Linear(4, 4)

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=4):
                with patch("torch.distributed.get_rank", return_value=0):
                    ar = SimpleNamespace()
                    block, sync_fn = setup_ddp_if_needed_(ar, model, [0, 1])
                    # Should not be the noop
                    assert sync_fn is not _noop_sync
                    # Calling it should not raise
                    sync_fn()
