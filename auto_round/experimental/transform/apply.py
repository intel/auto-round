# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import torch
import tqdm

from auto_round.experimental.qmodules.mx import MXQuantLinearBase
from auto_round.experimental.transform.transform_config import TransformConfig
from auto_round.experimental.transform.transforms import build_transform

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
    transform_name = "transform_matrix"

    if config.location == "input":
        from auto_round.experimental.transform.triton.utils import is_triton_kernel_available

        # activation needs transpose
        inp_transform = build_transform(
            **config.dict(),
            inverse=True,
            device="cpu",
            precision=module.dtype,
        )

        if config.transform_type != "random_hadamard":
            transform_weight = inp_transform.weight
        else:
            transform_weight = None

        if is_triton_kernel_available():
            from auto_round.experimental.transform.triton.mxfp4 import mxfp4_forward_kernel_wrapper

            def input_hook(self, args):
                input = args[0]
                # transform(input)
                orig_shape = input.shape
                x_flat = input.contiguous().flatten(end_dim=-2)
                qdq_input, _ = mxfp4_forward_kernel_wrapper(
                    x_flat,
                    (
                        transform_weight if transform_weight is not None else self.transform_matrix.T
                    ),  # this matrix from w_transform, needs transpose
                )
                return qdq_input.reshape(orig_shape)

            # for fused transform + quantization kernel
            module.pre_dequantized_input = True
            module.register_forward_pre_hook(input_hook, prepend=True)
        else:

            from auto_round.experimental.transform.utils.matrix import _multihead_matmul

            def input_hook(self, args):
                input = args[0]

                ori_shape = input.shape

                if transform_weight is not None:
                    input = input.view(-1, transform_weight.shape[0])
                    return _multihead_matmul(input, transform_weight.to(input.device)).view(ori_shape)
                else:
                    input = input.view(-1, self.transform_matrix.shape[0])
                    return _multihead_matmul(input, self.transform_matrix.T).view(ori_shape)

            # for fused transform + quantization kernel
            module.pre_dequantized_input = False
            module.register_forward_pre_hook(input_hook, prepend=True)

    elif config.location == "weight":
        # eagerly apply transformation to weight
        # fuse transform into weight
        assert hasattr(module, "weight")

        w_transform = build_transform(
            **config.dict(),
            device=module.weight.device,
            precision=module.weight.dtype,
        )

        # need save random hadamard matrix needed when inference
        if config.transform_type == "random_hadamard":
            module.register_module(transform_name, w_transform)
            # for saving transform weight
            from auto_round.experimental.transform.patch_modules import patch_quantlinear

            patch_quantlinear()

        if config.need_calibration:
            # for training, the weight changes with every forward pass
            # for autoround tuning: patch wrapper linear qdq_weight func
            from auto_round.experimental.transform.patch_modules import (
                patch_wrapperlinear_to_apply_transform,
                patch_wrapperwalayer_forward_to_apply_transform,
            )

            inp_transform = build_transform(
                **config.dict(),
                location="input",
                inverse=True,
                device=module.weight.device,
                precision=module.weight.dtype,
            )

            patch_wrapperlinear_to_apply_transform(w_transform, inp_transform)
            patch_wrapperwalayer_forward_to_apply_transform(inp_transform)

        else:
            # transform is no longer needed (unfusing is not supported)
            # delattr(module, transform_name)
            # fuse transform into weight
            with torch.no_grad():
                getattr(module, "weight").copy_(w_transform(module.weight).to(module.weight.device))

    else:
        # TODO: apply transform to output/q/k
        raise NotImplementedError()
