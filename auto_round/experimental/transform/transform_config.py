# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, field_validator

__all__ = ["TransformConfig"]


class TransformConfig(BaseModel):
    """
    Configuration of transforms to be applied to a model. This config is to be
    serialized within a model's `config.json` file
    """

    # required, currently only supports mxfp4
    quant_scheme: str = Field(..., description="Quantization scheme. Currently supports 'MXFP4/MXFP8'.")

    transform_block_size: int = Field(default=32)

    transform_type: str = Field(default="hadamard")

    location: str = Field(default="weight", exclude=True)

    # apply transform inside modules for nvfp4, autoround tuning etc.
    need_calibration: bool = Field(default=False, exclude=True)

    # for random hadamard transform
    random_seed: bool = Field(default=False, exclude=True)

    @field_validator("quant_scheme")
    @classmethod
    def validate_quant_scheme(cls, v: str) -> str:
        if v not in ["MXFP4", "MXFP8"]:
            raise ValueError(f"Unsupported quant_scheme: {v}. Currently 'mxfp4/mxfp8' are supported.")
        return v

    @field_validator("transform_type")
    @classmethod
    def validate_transform_type(cls, v: str) -> str:
        allowed = {"hadamard", "random_hadamard"}
        if v not in allowed:
            raise ValueError(f"Unsupported transform_type: {v}. Supported values: {sorted(allowed)}")
        return v
