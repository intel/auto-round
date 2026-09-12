# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "quantize_wan_a14b_svdquant.py"
spec = importlib.util.spec_from_file_location("quantize_wan_a14b_script", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_smoke_can_calibrate_without_downloading_a_dataset(tmp_path):
    args = runner.parse_args(["--output", str(tmp_path / "new")])
    assert args.calib_steps >= 2
    assert len(runner.calibration_data(args)) == args.nsamples


@pytest.mark.parametrize("extra", [["--height", "257"], ["--num-frames", "8"], ["--calib-steps", "1"]])
def test_rejects_invalid_wan_calibration_dimensions(extra):
    with pytest.raises(SystemExit):
        runner.parse_args(["--output", "unused", *extra])


def test_quality_requires_representative_prompts():
    with pytest.raises(SystemExit):
        runner.parse_args(["--output", "unused", "--profile", "quality"])


def test_prompt_file_enforces_sample_count(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first scene\n\nsecond scene\n")
    args = runner.parse_args(["--output", "unused", "--prompts-file", str(prompts), "--nsamples", "3"])
    with pytest.raises(ValueError, match="Need 3 prompts"):
        runner.calibration_data(args)
    args.nsamples = 2
    assert runner.calibration_data(args) == [([0], ["first scene"]), ([1], ["second scene"])]
