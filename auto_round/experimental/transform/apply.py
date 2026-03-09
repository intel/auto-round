# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import torch
import tqdm

from ..qmodules.mx import MXQuantLinearBase
from .transform_config import TransformConfig
from .transforms import build_transform

__all__ = ["apply_transform"]


def apply_transform(model: torch.nn.Module, config: TransformConfig, use_tqdm=True, desc=None):
    """
    Apply a transform config to a model. Add weight transforms and
    activation transforms are attached as submodules and trigger via pytorch hooks

    :param model: model to apply config to
    :param config: transform config to apply
    """

    modules_config = [
        (name, module, config)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) or isinstance(module, MXQuantLinearBase)
    ]

    desc = f"Applying {config.transform_type} transforms" if desc is None else desc
    for name, module, config in tqdm.tqdm(modules_config, desc=desc, disable=(not use_tqdm)):
        if "lm_head" in name:
            continue
        _apply_to_module(model, module, config)

    # attach config to model for compression/serialization
    setattr(model, "transform_config", config)

    return model


def _apply_to_module(
    model: torch.nn.Module,
    module: torch.nn.Module,
    config: TransformConfig,
):
    """
    Create transforms and apply them to the module

    :param model: model which module belongs to
    :param module: target module to apply transforms to
    """

    # create transform as submodule
    transform_name = "forward_hadamard_matrix"
    transform = build_transform(**config.dict())
    module.register_module(transform_name, transform)

    if config.location == "input":
        from .triton.mxfp4 import mxfp4_forward_kernel_wrapper

        def input_hook(_, args):
            input = args[0]
            # transform(input)
            orig_shape = input.shape
            x_flat = input.contiguous().flatten(end_dim=-2)
            qdq_input, _ = mxfp4_forward_kernel_wrapper(
                x_flat, transform.get_transform_matrix(input.device, input.dtype)
            )
            return qdq_input.reshape(orig_shape)

        # for fused transform + quantization kernel
        module.pre_dequantized_input = True

        module.register_forward_pre_hook(input_hook, prepend=True)

    elif config.location == "weight":
        # eagerly apply transformation to weight
        # fuse transform into weight
        assert hasattr(module, "weight")
        with torch.no_grad():
            getattr(module, "weight").copy_(transform(module.weight.to("cuda")).to(module.weight.device))
        # transform is no longer needed (unfusing is not supported)
        delattr(module, transform_name)

    else:
        # TODO: apply transform to output/q/k
        raise NotImplementedError()
