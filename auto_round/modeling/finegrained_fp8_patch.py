# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/finegrained_fp8.py
from transformers.core_model_loading import ConversionOps
from transformers.quantizers.quantizers_utils import should_convert_module
from transformers.utils import is_kernels_available, is_torch_accelerator_available, is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn
    # import triton
    # import triton.language as tl
    from torch.nn import functional as F


logger = logging.get_logger(__name__)



_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max



class FP8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=torch.float8_e4m3fn,
        block_size: tuple[int, int] | None = None,
        activation_scheme="dynamic",
    ):
        super().__init__(in_features, out_features)

        # If block size is None, it means that we are doing per-tensor quantization
        self.block_size = block_size
        self.activation_scheme = activation_scheme

        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))

        if self.block_size is None:
            self.weight_scale_inv = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            scale_out_features = (out_features + self.block_size[0] - 1) // self.block_size[0]
            scale_in_features = (in_features + self.block_size[1] - 1) // self.block_size[1]
            self.weight_scale_inv = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )

        if self.activation_scheme == "static":
            self.activation_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    # def forward(self, input: torch.Tensor) -> torch.Tensor:
    #     if self.weight.element_size() > 1:
    #         return F.linear(input, self.weight, self.bias)
    #     else:
    #         if isinstance(self.weight, torch.distributed.tensor.DTensor):
    #             weight = self.weight._local_tensor.contiguous()
    #             scale_inv = self.weight_scale_inv._local_tensor.contiguous()
    #         else:
    #             weight = self.weight.contiguous()
    #             scale_inv = self.weight_scale_inv.contiguous()
    #         # Context manager used to switch among the available accelerators
    #         device_type = torch.accelerator.current_accelerator().type if is_torch_accelerator_available() else "cuda"
    #         torch_accelerator_module = getattr(torch, device_type, torch.cuda)
    #         with torch_accelerator_module.device(input.device):
    #             if self.activation_scheme == "dynamic":
    #                 qinput, scale = act_quant(input, self.block_size[1])
    #             elif self.activation_scheme == "static":
    #                 scale = self.activation_scale.to(torch.float32)
    #                 qinput = (input / scale).clamp(min=_FP8_MIN, max=_FP8_MAX).to(torch.float8_e4m3fn)

    #             else:
    #                 raise NotImplementedError("Not supported")

    #             output = w8a8_block_fp8_matmul(
    #                 qinput,
    #                 weight,
    #                 scale,
    #                 scale_inv,
    #                 self.block_size,
    #                 output_dtype=input.dtype,
    #             )

    #         # Blocks the CPU until all accelerator operations on the specified device are complete. It is used to ensure that the results of the
    #         # preceding operations are ready before proceeding
    #         torch_accelerator_module.synchronize()
    #         if self.bias is not None:
    #             output = output + self.bias

    #         return output.to(dtype=input.dtype)


def _ceil_div(a, b):
    return (a + b - 1) // b



def replace_with_fp8_linear(
    model, modules_to_not_convert: list[str] | None = None, quantization_config=None, pre_quantized=False
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `FP8Linear` modules.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`list[`str`]`, *optional*, defaults to `None`):
            Names of the modules to not convert. In practice we keep the `lm_head` in full precision for numerical stability reasons.
        quantization_config (`FbgemmFp8Config`):
            The quantization config object that contains the quantization parameters.
        pre_quantized (`book`, defaults to `False`):
            Whether the model is pre-quantized or not
    """

    if quantization_config.dequantize:
        return model

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        # we need this to correctly materialize the weights during quantization
        module_kwargs = {} if pre_quantized else {"dtype": None}
        new_module = None
        with torch.device("meta"):
            if isinstance(module, nn.Linear):
                new_module = FP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    activation_scheme=quantization_config.activation_scheme,
                    block_size=quantization_config.weight_block_size,
                    **module_kwargs,
                )
            if new_module is not None:
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using fp8 but no linear modules were found in your model."
            " Please double check your model architecture."
        )
    return model


class Fp8Quantize(ConversionOps):
    """
    A quantization operation that creates two tensors, weight and scale out of a weight.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        # Unpack single key/value (value may be wrapped in a list)
        target_keys, value = tuple(input_dict.items())[0]
        value = value[0]

        # Resolve block size (support dict-like or attr-like quant_config)
        block_size = None
        if self.hf_quantizer.quantization_config is not None:
            if isinstance(self.hf_quantizer.quantization_config, dict):
                block_size = self.hf_quantizer.quantization_config.get("weight_block_size")
            else:
                block_size = getattr(self.hf_quantizer.quantization_config, "weight_block_size", None)
        if block_size is None:
            block_size = (value.shape[-2], value.shape[-1])

        block_m, block_n = block_size
        rows, cols = value.shape[-2], value.shape[-1]

        # Enforce exact tiling like your original
        if rows % block_m != 0 or cols % block_n != 0:
            raise ValueError(
                f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_m}, {block_n}). for {target_keys}"
            )

        # Leading dims can be empty (2D) or include num_experts/... (3D+)
        leading_shape = value.shape[:-2]
        rows_tiles = rows // block_m
        cols_tiles = cols // block_n

        original_shape = value.shape
        value_fp32 = value.to(torch.float32)

        # Reshape to (..., rows_tiles, block_m, cols_tiles, block_n)
        reshaped = value_fp32.reshape(*leading_shape, rows_tiles, block_m, cols_tiles, block_n)

        # Per-tile max-abs over the block dims
        # dims: block_m is at -3, block_n is at -1 after the reshape
        max_abs = reshaped.abs().amax(dim=(-3, -1))
        safe_max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))

        # Tile scale (we store inverse scale like your Linear: weight_scale_inv)
        scales = _FP8_MAX / safe_max_abs
        scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))  # keep zeros stable

        # Broadcast scales back over the block dims and quantize
        # max_abs/scales shape: (..., rows_tiles, cols_tiles)
        scales_broadcast = scales.unsqueeze(-1).unsqueeze(-3)  # -> (..., rows_tiles, 1, cols_tiles, 1)
        scaled = reshaped * scales_broadcast

        quantized = torch.clamp(scaled, min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)

        quantized = quantized.reshape(original_shape)

        inv_scales = (1.0 / scales).to(torch.float32)  # shape: (*leading, rows_tiles, cols_tiles)
        if target_keys.endswith("weight"):
            scale_key = target_keys.rsplit(".", 1)[0] + ".weight_scale_inv"
        else:
            scale_key = target_keys + "_scale_inv"

        # Return both quantized weights and per-tile inverse scales (keeps leading dims, e.g., num_experts)
        return {
            target_keys: quantized,
            scale_key: inv_scales,
        }


class Fp8Dequantize(ConversionOps):
    """Inverse operation of :class:`Fp8Quantize`. Takes a pair (weight, scale) and reconstructs the fp32 tensor."""

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if len(input_dict) < 2:
            # case where we only got weights, need to check for "weight$"
            return {full_layer_name: input_dict["weight$"]}

        quantized = input_dict["weight$"][0]
        scales = input_dict["weight_scale_inv"][0]

        rows, cols = quantized.shape[-2:]
        block_size = self.hf_quantizer.quantization_config.weight_block_size
        if block_size is None:
            block_size = (quantized.shape[-2], quantized.shape[-1])

        block_m, block_n = block_size

        if rows % block_m != 0 or cols % block_n != 0:
            raise ValueError(
                f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_m}, {block_n})."
            )
        quantized = quantized.to(scales.dtype)
        reshaped = quantized.reshape(-1, rows // block_m, block_m, cols // block_n, block_n)
        expanded_scales = scales.reshape(-1, rows // block_m, cols // block_n)
        expanded_scales = expanded_scales.unsqueeze(-1).unsqueeze(2)
        dequantized = reshaped * expanded_scales

        return {
            full_layer_name: dequantized.reshape(quantized.shape),
        }


