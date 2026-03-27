# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import torch
import tqdm

from auto_round.experimental.qmodules.mx import MXQuantLinearBase
from auto_round.experimental.transform.hadamard_config import HadamardConfig
from auto_round.experimental.transform.hadamards import build_hadamard_transform
from auto_round.experimental.utils import is_triton_kernel_available, normalize_hadamard_config

__all__ = ["apply_hadamard_transform"]


def apply_hadamard_transform(
    model: torch.nn.Module,
    config: str | dict | HadamardConfig | None,
    need_calibration: bool = False,
    location: str = "weight",
    use_tqdm=True,
    desc=None,
):
    """
    Apply a transform configuration to a model.

    Weight and activation transforms are attached as submodules and are
    triggered via PyTorch hooks.

    :param model: Model to which the transform configuration will be applied.
    :param config: Transform configuration to apply. Supported values are:
        * ``str``: A named/preset transform configuration. In this case,
          ``scheme`` is typically required so that the preset can be
          resolved to a concrete quantization/transform configuration.
        * ``dict``: A raw configuration mapping that will be normalized
          (via :func:`normalize_hadamard_config`) and then passed to
          :class:`TransformConfig`.
        * :class:`TransformConfig`: An existing configuration instance.
          This will be used to construct the final configuration after
          normalization.
        * ``None``: Uses the default behavior of
          :func:`_normalize_hadamard_config` (for example, inferring a
          configuration from ``scheme`` or other project defaults), if
          supported.
    :param scheme: Optional quantization/transform scheme identifier used
        when ``config`` is a ``str`` (and, if supported, when it is
        ``None``) to determine which concrete configuration to build.
        Ignored when ``config`` is already a ``dict`` or
        :class:`TransformConfig`.
    :param use_tqdm: If ``True``, wrap the per-module application in a
        tqdm progress bar.
    :param desc: Optional description string to show in the tqdm progress
        bar. If ``None``, a description will be derived from
        ``config.transform_type``.
    """

    config = normalize_hadamard_config(config)
    if not isinstance(config, HadamardConfig):
        config = HadamardConfig(**config)

    modules_config = [
        (name, module, config)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) or isinstance(module, MXQuantLinearBase)
    ]

    desc = f"Applying {config.hadamard_type} transforms" if desc is None else desc
    for name, module, config in tqdm.tqdm(modules_config, desc=desc, disable=(not use_tqdm)):
        if "lm_head" in name:
            continue
        _apply_to_module(model, module, config, need_calibration, location)

    # attach config to model for compression/serialization
    setattr(model, "hadamard_config", config)

    return model


def _apply_to_module(
    model: torch.nn.Module,
    module: torch.nn.Module,
    config: HadamardConfig,
    need_calibration: bool = False,
    location: str = "weight",
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
            **config.dict(),
            location="input",
            inverse=True,
            device="cpu",
            precision=module.dtype,
        )

        if config.hadamard_type != "random_hadamard":
            hadamard_weight = input_hadamard_transform.weight
        else:
            hadamard_weight = None

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
                        hadamard_weight if hadamard_weight is not None else self.hadamard_matrix.T
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

                if hadamard_weight is not None:
                    input = input.view(-1, hadamard_weight.shape[0])
                    return _multihead_matmul(input, hadamard_weight.to(input.device)).view(ori_shape)
                else:
                    input = input.view(-1, self.hadamard_matrix.shape[0])
                    return _multihead_matmul(input, self.hadamard_matrix.T).view(ori_shape)

            # for fused transform + quantization kernel
            module.pre_dequantized_input = False
            module.register_forward_pre_hook(input_hook, prepend=True)

    elif location == "weight":
        # eagerly apply transformation to weight
        # fuse transform into weight
        assert hasattr(module, "weight")

        weight_hadamard_transform = build_hadamard_transform(
            **config.dict(),
            location="weight",
            device=module.weight.device,
            precision=module.weight.dtype,
        )

        # need save random hadamard matrix needed when inference
        if config.hadamard_type == "random_hadamard":
            module.register_module(config.hadamard_type, weight_hadamard_transform)
            # for saving transform weight
            from auto_round.experimental.transform.patch_modules import patch_quantlinear

            patch_quantlinear(config.hadamard_type)

        if need_calibration:
            # for training, the weight changes with every forward pass
            # for autoround tuning: patch wrapper linear qdq_weight func
            from auto_round.experimental.transform.patch_modules import (
                patch_wrapperlinear_to_apply_transform,
                patch_wrapperwalayer_forward_to_apply_transform,
            )

            input_hadamard_transform = build_hadamard_transform(
                **config.dict(),
                location="input",
                inverse=True,
                device=module.weight.device,
                precision=module.weight.dtype,
            )

            patch_wrapperlinear_to_apply_transform(weight_hadamard_transform, input_hadamard_transform)
            patch_wrapperwalayer_forward_to_apply_transform(input_hadamard_transform)

        else:
            # transform is no longer needed (unfusing is not supported)
            # delattr(module, transform_name)
            # fuse transform into weight
            with torch.no_grad():
                getattr(module, "weight").copy_(weight_hadamard_transform(module.weight).to(module.weight.device))

    else:
        # TODO: apply transform to output/q/k
        raise NotImplementedError()
