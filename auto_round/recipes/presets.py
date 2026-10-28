# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Built-in recommendations; numerical scheme definitions live in schemes.py."""

from auto_round.recipes.recipe import AlgorithmSpec, Recipe, calibration_defaults


def builtin_recipes() -> tuple[Recipe, ...]:
    calibration = calibration_defaults()
    return (
        Recipe(
            "ar",
            (AlgorithmSpec("quantizer", "auto_round", {"iters": 200}),),
            dict(calibration),
            description="Balanced quantization cost and accuracy.",
            aliases=("default",),
        ),
        Recipe(
            "ar-light",
            (AlgorithmSpec("quantizer", "auto_round", {"iters": 50, "lr": 5e-3}),),
            dict(calibration),
            description="Lower-cost iterative quantization.",
            aliases=("light",),
        ),
        Recipe(
            "ar-best",
            (AlgorithmSpec("quantizer", "auto_round", {"iters": 1000}),),
            {**calibration, "nsamples": 512},
            description="Accuracy-oriented iterative quantization.",
            aliases=("best",),
        ),
        Recipe(
            "rtn",
            (AlgorithmSpec("quantizer", "rtn", {"disable_opt_rtn": True}),),
            {**calibration, "nsamples": 1},
            description="Plain round-to-nearest quantization.",
        ),
        Recipe(
            "opt-rtn",
            (AlgorithmSpec("quantizer", "rtn", {"disable_opt_rtn": False}),),
            dict(calibration),
            description="Optimized round-to-nearest quantization.",
            aliases=("opt_rtn",),
        ),
        Recipe(
            "awq",
            (AlgorithmSpec("smooth", "awq"), AlgorithmSpec("quantizer", "rtn", {"disable_opt_rtn": True})),
            dict(calibration),
            description="AWQ smoothing followed by plain RTN.",
        ),
    )
