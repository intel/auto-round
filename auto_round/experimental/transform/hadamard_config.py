# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, field_validator

__all__ = ["HadamardConfig"]


class HadamardConfig(BaseModel):
    """
    Configuration of transforms to be applied to a model. This config is to be
    serialized within a model's `config.json` file
    """

    block_size: int = Field(default=32)

    hadamard_type: str = Field(default="hadamard")

    # for random hadamard transform
    random_seed: bool = Field(default=False, exclude=True)


    @field_validator("hadamard_type")
    @classmethod
    def validate_hadamard_type(cls, v: str) -> str:
        allowed = {"hadamard", "random_hadamard"}
        if v not in allowed:
            raise ValueError(f"Unsupported hadamard_type: {v}. Supported values: {sorted(allowed)}")
        return v
