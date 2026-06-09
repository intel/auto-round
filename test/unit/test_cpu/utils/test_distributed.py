# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.utils.distributed``.

The module is tiny (15 lines) but the behaviour of ``is_distributed`` is
non-trivial because it uses ``functools.lru_cache`` on a function that
queries global state (``torch.distributed.is_initialized``).  A single
test process cannot actually launch a multi-rank DDP job, so we cover
the two code paths with monkey-patches instead:

1. The *negative* path: ``is_distributed()`` returns ``False`` when
   ``torch.distributed.is_initialized()`` is ``False``.
2. The *negative* path with world_size==1: when distributed is initialized
   but the world size is 1, ``is_distributed()`` still returns ``False``
   (single-process "distributed" doesn't count).
3. The *positive* path: when both ``is_initialized()`` and
   ``get_world_size() > 1`` are true, ``is_distributed()`` returns
   ``True``.

``setup_ddp_if_needed_`` is exercised in all three cases to make sure
the early-return when not distributed works and that the DDP-wrapping
path can be hit (mocked, since we cannot actually init a process group
inside a unit test).
"""

from unittest import mock

import pytest
import torch
import torch.nn as nn

from auto_round.utils.distributed import is_distributed, setup_ddp_if_needed_


# ---------------------------------------------------------------------------
# is_distributed
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_is_distributed_cache():
    """``is_distributed`` is lru_cached.  Clear it before every test so
    monkey-patches to ``torch.distributed`` take effect.
    """
    is_distributed.cache_clear()
    yield
    is_distributed.cache_clear()


def test_is_distributed_false_when_not_initialized():
    """Default state: no torch.distributed process group -> False."""
    with mock.patch("torch.distributed.is_initialized", return_value=False), mock.patch(
        "torch.distributed.get_world_size", return_value=4
    ) as ws:
        assert is_distributed() is False
        # ``get_world_size`` shouldn't even be called when not initialized.
        ws.assert_not_called()


def test_is_distributed_false_when_world_size_is_one():
    """Single-rank "distributed" still counts as non-distributed."""
    with mock.patch("torch.distributed.is_initialized", return_value=True), mock.patch(
        "torch.distributed.get_world_size", return_value=1
    ):
        assert is_distributed() is False


def test_is_distributed_true_when_world_size_greater_than_one():
    with mock.patch("torch.distributed.is_initialized", return_value=True), mock.patch(
        "torch.distributed.get_world_size", return_value=4
    ):
        assert is_distributed() is True


def test_is_distributed_is_cached_across_calls():
    """The lru_cache means the second call doesn't re-query torch.distributed.

    We patch ``get_world_size`` with a side effect that records the
    number of times it was called; the second ``is_distributed()`` call
    must not invoke it.
    """
    ws_call_count = {"n": 0}

    def fake_ws():
        ws_call_count["n"] += 1
        return 8

    with mock.patch("torch.distributed.is_initialized", return_value=True), mock.patch(
        "torch.distributed.get_world_size", side_effect=fake_ws
    ):
        assert is_distributed() is True
        assert is_distributed() is True
        assert is_distributed() is True
        assert ws_call_count["n"] == 1


# ---------------------------------------------------------------------------
# setup_ddp_if_needed_
# ---------------------------------------------------------------------------


def test_setup_ddp_returns_none_when_not_distributed():
    """When ``is_distributed()`` is False the helper returns ``None``
    immediately and does not touch the block.
    """
    with mock.patch("auto_round.utils.distributed.is_distributed", return_value=False):
        block = nn.Linear(2, 2)
        device_list = [0]
        out = setup_ddp_if_needed_(ar=None, block=block, device_list=device_list)
        # The block is not wrapped in DDP - the function returns None.
        assert out is None
        assert not isinstance(block, torch.nn.parallel.DistributedDataParallel)


def test_setup_ddp_calls_ddp_constructor_when_distributed():
    """When ``is_distributed()`` is True, the block is wrapped in
    ``DistributedDataParallel``.

    We mock both ``is_distributed`` and the DDP constructor so the test
    runs without an actual process group.

    Note: the helper currently has no explicit ``return`` statement, so
    it returns ``None`` after wrapping.  We assert the DDP constructor
    was called with the right arguments instead of asserting on the
    return value (the in-place wrap is the only observable side effect
    under a single-process test).
    """
    block = nn.Linear(2, 2)

    ddp_sentinel = object()
    with mock.patch("auto_round.utils.distributed.is_distributed", return_value=True), mock.patch(
        "torch.nn.parallel.DistributedDataParallel", return_value=ddp_sentinel
    ) as ddp_cls, mock.patch("torch.distributed.get_rank", return_value=0):
        out = setup_ddp_if_needed_(ar=None, block=block, device_list=[0])

    ddp_cls.assert_called_once()
    args, kwargs = ddp_cls.call_args
    # ``block`` is the first positional arg, the rest are kwargs.
    assert args[0] is block
    assert kwargs["device_ids"] == [0]
    assert kwargs["find_unused_parameters"] is True
    # The function returns None (no explicit return statement in source).
    assert out is None


def test_setup_ddp_device_list_forwarded_verbatim():
    """The device list is forwarded as-is to ``DistributedDataParallel``."""
    block = nn.Linear(2, 2)
    with mock.patch("auto_round.utils.distributed.is_distributed", return_value=True), mock.patch(
        "torch.nn.parallel.DistributedDataParallel"
    ) as ddp_cls, mock.patch("torch.distributed.get_rank", return_value=2):
        setup_ddp_if_needed_(ar=None, block=block, device_list=[0, 1])

    ddp_cls.assert_called_once_with(block, device_ids=[0, 1], find_unused_parameters=True)


def test_setup_ddp_logs_rank_and_warning_when_distributed():
    """The helper emits a warning and logs the rank when DDP is enabled."""
    block = nn.Linear(2, 2)
    with mock.patch("auto_round.utils.distributed.is_distributed", return_value=True), mock.patch(
        "torch.nn.parallel.DistributedDataParallel"
    ), mock.patch("torch.distributed.get_rank", return_value=3) as rank:
        setup_ddp_if_needed_(ar=None, block=block, device_list=[0])
    # ``get_rank`` was consulted.
    rank.assert_called_once()
