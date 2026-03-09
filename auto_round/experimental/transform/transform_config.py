# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

__all__ = ["TransformConfig"]


class TransformConfig(BaseModel):
    """
    Configuration of transforms to be applied to a model. This config is to be
    serialized within a model's `config.json` file
    """

    transform_block_size: int = Field(default=32)

    transform_type: str = Field(default="hadamard")

    location: str = Field(default="weight", exclude=True)

    requires_grad: bool = Field(default=False, exclude=True)
