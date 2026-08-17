# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.__main__``."""

import sys
from unittest.mock import patch

import pytest

import auto_round.__main__ as main_module


class TestMainModuleImports:
    """Test that __main__ module re-exports CLI entry points."""

    def test_run_exported(self):
        assert hasattr(main_module, "run")
        assert callable(main_module.run)

    def test_run_rtn_exported(self):
        assert hasattr(main_module, "run_rtn")
        assert callable(main_module.run_rtn)

    def test_run_best_exported(self):
        assert hasattr(main_module, "run_best")
        assert callable(main_module.run_best)

    def test_run_light_exported(self):
        assert hasattr(main_module, "run_light")
        assert callable(main_module.run_light)

    def test_run_eval_exported(self):
        assert hasattr(main_module, "run_eval")
        assert callable(main_module.run_eval)

    def test_run_mllm_exported(self):
        assert hasattr(main_module, "run_mllm")
        assert callable(main_module.run_mllm)

    def test_run_opt_rtn_exported(self):
        assert hasattr(main_module, "run_opt_rtn")
        assert callable(main_module.run_opt_rtn)


class TestMainEntryPoint:
    """Test __main__ entry point execution."""

    def test_main_block_calls_run(self):
        with patch("auto_round.cli.main.run") as mock_run:
            # Simulate running as __main__
            runpy_path = "auto_round.__main__"
            import runpy

            # Clear the module cache so it re-runs the __main__ block
            if runpy_path in sys.modules:
                del sys.modules[runpy_path]

            runpy.run_module(runpy_path, run_name="__main__")
            mock_run.assert_called_once()
