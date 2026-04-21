# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import torch
import tqdm

from auto_round.experimental.qmodules.base import QModuleBase
from auto_round.experimental.transform.hadamard_config import HadamardConfig
from auto_round.experimental.transform.hadamards import build_hadamard_transform
from auto_round.experimental.utils import is_triton_kernel_available, normalize_hadamard_config

__all__ = ["apply_hadamard_transform"]


def apply_hadamard_transform(
    model: torch.nn.Module,
    config: str | dict | HadamardConfig | None,
    location: str = "weight",
    use_tqdm=True,
    desc=None,
    data_type="mx_fp",
):
    """
    Apply a transform configuration to a model.

    Weight and activation transforms are attached as submodules and are
    triggered via PyTorch hooks.

    :param model: Model to which the transform configuration will be applied.
    :param config: Transform configuration to apply. Supported values are:
        * ``str``: A named/preset transform configuration. In this case,
          resolved to a concrete quantization/transform configuration.
        * ``dict``: A raw configuration mapping that will be normalized
          (via :func:`normalize_hadamard_config`) and then passed to
          :class:`TransformConfig`.
        * :class:`TransformConfig`: An existing configuration instance.
          This will be used to construct the final configuration after
          normalization.
        * ``None``: Uses the default behavior of
          :func:`_normalize_hadamard_config` (for example, inferring a
          configuration from ``data_type`` or other project defaults), if
          supported.
    :param data_type: quantization data type.
    :param use_tqdm: If ``True``, wrap the per-module application in a
        tqdm progress bar.
    :param desc: Optional description string to show in the tqdm progress
        bar. If ``None``, a description will be derived from
        ``config.transform_type``.
    """

    config = normalize_hadamard_config(config, data_type)
    if not isinstance(config, HadamardConfig):
        config = HadamardConfig(**config)

    modules_config = [
        (name, module, config)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) or isinstance(module, QModuleBase)
    ]

    desc = f"Applying {config.hadamard_type} transforms" if desc is None else desc
    for name, module, config in tqdm.tqdm(modules_config, desc=desc, disable=(not use_tqdm)):
        if "lm_head" in name:
            continue
        _apply_to_module(model, module, config, location, data_type)

    # attach config to model for compression/serialization
    setattr(model, "hadamard_config", config)

    return model


def _apply_to_module(
    model: torch.nn.Module,
    module: torch.nn.Module,
    config: HadamardConfig,
    location: str = "weight",
    data_type: str = "mx_fp",
):
    """
    Create transforms and apply them to the module

    :param model: model which module belongs to
    :param module: target module to apply transforms to
    """

    # create transform as submodule
    hadamard_name = config.hadamard_type

    if location == "input":

        # activation needs transpose
        input_hadamard_transform = build_hadamard_transform(
            **config.model_dump(),
            location="input",
            inverse=True,
            device="cpu",
            precision=module.dtype,  # for online activation, the transform dtype maybe bfloat16/float16.
        )

        if config.hadamard_type != "random_hadamard":
            hadamard_weight = input_hadamard_transform.weight
        else:
            hadamard_weight = None

        if is_triton_kernel_available(data_type):
            from auto_round.experimental.transform.triton.mxfp4 import mxfp4_forward_kernel_wrapper

            def input_hook(self, args):
                input = args[0]
                # transform(input)
                orig_shape = input.shape
                orig_dtype = input.dtype
                x_flat = input.contiguous().flatten(end_dim=-2)
                qdq_input, _ = mxfp4_forward_kernel_wrapper(
                    x_flat,
                    (
                        hadamard_weight.to(orig_dtype)
                        if hadamard_weight is not None
                        else self.hadamard_matrix.T.to(orig_dtype)
                    ),  # this matrix from w_transform, needs transpose
                )
                return qdq_input.reshape(orig_shape).to(orig_dtype)

            # for fused transform + quantization kernel
            module.pre_dequantized_input = True
            module.register_forward_pre_hook(input_hook, prepend=True)
        else:

            from auto_round.experimental.transform.utils.matrix import _multihead_matmul

            def input_hook(self, args):
                input = args[0]

                ori_shape = input.shape
                orig_dtype = input.dtype

                if hadamard_weight is not None:
                    input = input.view(-1, hadamard_weight.shape[0])
                    return (
                        (_multihead_matmul(input, hadamard_weight.to(input.device).to(orig_dtype)))
                        .view(ori_shape)
                        .to(orig_dtype)
                    )
                else:
                    input = input.view(-1, self.hadamard_matrix.shape[0])
                    return (
                        (_multihead_matmul(input, self.hadamard_matrix.T.to(orig_dtype))).view(ori_shape).to(orig_dtype)
                    )

            # for fused transform + quantization kernel
            module.pre_dequantized_input = False
            module.register_forward_pre_hook(input_hook, prepend=True)

    elif location == "weight":
        # eagerly apply transformation to weight
        # fuse transform into weight
        assert hasattr(module, "weight")

        weight_hadamard_transform = build_hadamard_transform(
            **config.model_dump(),
            location="weight",
            device=module.weight.device,
        )

        # need save random hadamard matrix needed when inference
        if config.hadamard_type == "random_hadamard":
            # for saving transform weight
            from auto_round.experimental.transform.patch_modules import patch_quantlinear

            patch_quantlinear(weight_hadamard_transform)

        # for autoround tuning: weight not tuning
        # for rtn: weight transformed before saving
        from auto_round.experimental.transform.patch_modules import (
            patch_wrapperlinear_to_apply_transform,
            patch_wrapperwalayer_forward_to_apply_transform,
        )

        input_hadamard_transform = build_hadamard_transform(
            **config.model_dump(),
            location="input",
            inverse=True,
            device=module.weight.device,
            precision=module.weight.dtype,  # for online activation, the transform dtype maybe bfloat16/float16.
        )

        patch_wrapperlinear_to_apply_transform(weight_hadamard_transform, input_hadamard_transform)
        patch_wrapperwalayer_forward_to_apply_transform(input_hadamard_transform)

    else:
        # TODO: apply transform to output/q/k
        raise NotImplementedError()
