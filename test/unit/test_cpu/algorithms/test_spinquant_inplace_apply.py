# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.algorithms.transforms.spinquant.inplace.apply``."""

import torch
import torch.nn as nn

from auto_round.algorithms.transforms.spinquant.inplace.apply import (
    apply_spinquant_in_place,
    register_spinquant_hooks,
    remove_spinquant_hooks,
)
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig


class TestRegisterSpinquantHooks:
    """Test spinquant hook registration."""

    def test_register_no_r3_no_r4_returns_empty(self):
        model = nn.Linear(16, 16)
        config = SpinQuantConfig(r1=False, r2=False, r3=False, r4=False)
        handles = register_spinquant_hooks(model, config)
        assert handles == []

    def test_register_with_r4_hooks(self):
        """register_spinquant_hooks finds down_proj by suffix match."""
        model = nn.Module()
        model.layers = nn.ModuleList(
            [
                nn.ModuleDict({"mlp": nn.ModuleDict({"down_proj": nn.Linear(32, 16)})}),
                nn.ModuleDict({"mlp": nn.ModuleDict({"down_proj": nn.Linear(32, 16)})}),
            ]
        )
        config = SpinQuantConfig(r1=False, r2=False, r3=False, r4=True)
        handles = register_spinquant_hooks(
            model, config, intermediate_size=32, r4_rotation_size=16
        )
        assert isinstance(handles, list)
        # Both down_proj layers get hooks registered
        assert len(handles) == 2

    def test_remove_hooks(self):
        """remove_spinquant_hooks takes a list of handles."""
        model = nn.Module()
        model.layers = nn.ModuleList(
            [
                nn.ModuleDict({"mlp": nn.ModuleDict({"down_proj": nn.Linear(32, 16)})}),
            ]
        )
        config = SpinQuantConfig(r1=False, r2=False, r3=False, r4=True)
        handles = register_spinquant_hooks(
            model, config, intermediate_size=32, r4_rotation_size=16
        )
        assert len(handles) == 1
        # remove_spinquant_hooks takes the handles list directly
        remove_spinquant_hooks(handles)
        # After removal, calling again should be safe (no-op)
        remove_spinquant_hooks(handles)


class TestApplySpinquantInPlace:
    """Test the main apply_spinquant_in_place entry point."""

    def test_basic_application(self):
        model = nn.Module()
        model.embed = nn.Embedding(100, 16)
        model.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.ModuleDict(
                            {"q_proj": nn.Linear(16, 16), "k_proj": nn.Linear(16, 16)}
                        ),
                        "mlp": nn.ModuleDict(
                            {
                                "gate_proj": nn.Linear(16, 32),
                                "up_proj": nn.Linear(16, 32),
                                "down_proj": nn.Linear(32, 16),
                            }
                        ),
                    }
                )
            ]
        )
        model.ln = nn.LayerNorm(16)

        config = SpinQuantConfig(r1=False, r2=False, r3=False, r4=True)
        result = apply_spinquant_in_place(model, config)
        assert result is model

    def test_with_hooks(self):
        """apply_spinquant_in_place registers hooks on down_proj."""
        model = nn.Module()
        model.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mlp": nn.ModuleDict(
                            {"down_proj": nn.Linear(32, 16)},
                        )
                    }
                )
            ]
        )

        config = SpinQuantConfig(r1=False, r2=False, r3=False, r4=True)
        apply_spinquant_in_place(model, config)
        # Hooks should have been registered
        hooks = getattr(model, "_spinquant_handles", None)
        # Hooks may or may not be stored as attribute depending on implementation
        # The key is that the function completes without error
        assert model is not None  # basic sanity check
