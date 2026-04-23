# Copyright (c) 2025 Intel Corporation
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

"""
MLX format QuantLinear for inference via MLX kernels on Apple Silicon.

Uses mx.quantized_matmul for hardware-accelerated quantized inference.
Falls back to PyTorch dequantization if MLX is not available.

Tensor naming follows MLX convention:
  - weight: uint32 packed integers [out_features, packed_in_features]
  - scales: float16 [out_features, num_groups]
  - biases: float16 [out_features, num_groups]  (MLX calls zero-point-derived term "biases")
  - bias:   optional linear bias
"""

import math

import torch
import torch.nn as nn

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def _torch_to_mlx(t):
    """Convert a PyTorch tensor to MLX array."""
    # Accept either a numpy array or a torch.Tensor.
    try:
        import numpy as _np
    except Exception:
        _np = None

    # If it's a torch tensor, try to use DLPack for zero-copy when supported by MLX.
    if isinstance(t, torch.Tensor):
        # Ensure tensor is contiguous
        tt = t.contiguous()
        try:
            from torch.utils.dlpack import to_dlpack

            if hasattr(mx, "from_dlpack"):
                return mx.from_dlpack(to_dlpack(tt))
        except Exception:
            # fall back to numpy
            pass

        return mx.array(tt.cpu().numpy())

    # Otherwise assume it's a numpy-like object
    return mx.array(t)


def _mlx_to_torch(a, dtype=None):
    """Convert an MLX array to PyTorch tensor."""
    import numpy as np

    t = torch.from_numpy(np.array(a))
    if dtype is not None:
        t = t.to(dtype)
    return t


class QuantLinearMLX(nn.Module):
    """Quantized linear layer using MLX kernels for fast inference on Apple Silicon.

    Falls back to pure PyTorch dequantization when MLX is not available.
    """

    QUANT_TYPE = "mlx"

    def __init__(self, bits, group_size, infeatures, outfeatures, bias, **kwargs):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2, 3, 4, 8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1

        # MLX packing: [out_features, in_features * bits / 32]
        packed_dim = infeatures * bits // 32
        num_groups = math.ceil(infeatures / self.group_size)

        self.register_buffer(
            "weight",
            torch.zeros((outfeatures, packed_dim), dtype=torch.uint32),
        )
        self.register_buffer(
            "scales",
            torch.zeros((outfeatures, num_groups), dtype=torch.float16),
        )
        self.register_buffer(
            "biases",
            torch.zeros((outfeatures, num_groups), dtype=torch.float16),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(outfeatures, dtype=torch.float16))
        else:
            self.bias = None

        # Cache MLX arrays after first forward
        self._mlx_weight = None
        self._mlx_scales = None
        self._mlx_biases = None

    def _ensure_mlx_cache(self):
        """Convert PyTorch buffers to MLX arrays, cached for reuse."""
        if self._mlx_weight is None:
            # Avoid unnecessary device transfers when buffers are already on CPU.
            weight_for_mx = self.weight if self.weight.device.type == "cpu" else self.weight.cpu()
            scales_for_mx = self.scales if self.scales.device.type == "cpu" else self.scales.cpu()
            biases_for_mx = self.biases if self.biases.device.type == "cpu" else self.biases.cpu()

            self._mlx_weight = _torch_to_mlx(weight_for_mx)
            self._mlx_scales = _torch_to_mlx(scales_for_mx.float())
            self._mlx_biases = _torch_to_mlx(biases_for_mx.float())

    def _forward_mlx(self, x):
        """Forward pass using MLX quantized_matmul kernel."""
        self._ensure_mlx_cache()

        input_dtype = x.dtype
        # Avoid an unnecessary copy when input is already on CPU.
        if x.device.type == "cpu":
            x_cpu = x.detach()
        else:
            # For MPS or other devices, move to CPU for MLX bridge.
            # Use .to('cpu') which may be optimized on Apple Silicon unified memory.
            x_cpu = x.detach().to("cpu")

        # Prepare contiguous float32 tensor for MLX consumption; _torch_to_mlx
        # will attempt a zero-copy DLPack transfer if supported by MLX.
        x_for_mx = x_cpu.float().contiguous()
        x_mlx = _torch_to_mlx(x_for_mx)

        # mx.quantized_matmul: x @ w.T with quantized w
        out_mlx = mx.quantized_matmul(
            x_mlx,
            self._mlx_weight,
            scales=self._mlx_scales,
            biases=self._mlx_biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        mx.eval(out_mlx)

        out = _mlx_to_torch(out_mlx, dtype=input_dtype)

        if self.bias is not None:
            out = out + self.bias.to(input_dtype)

        return out.to(x.device)

    def _forward_torch(self, x):
        """Fallback forward pass using pure PyTorch dequantization."""
        input_dtype = x.dtype
        input_device = x.device
        # Move to CPU for bitwise unpack ops (MPS may not support all int ops)
        intweight = self._unpack_weight_torch(device=torch.device("cpu")).float()
        scales = self.scales.cpu().float().repeat_interleave(self.group_size, dim=1)[:, : self.infeatures]
        biases = self.biases.cpu().float().repeat_interleave(self.group_size, dim=1)[:, : self.infeatures]
        weight = scales * intweight + biases
        out = torch.nn.functional.linear(
            x.cpu().to(weight.dtype), weight, self.bias.cpu() if self.bias is not None else None
        )
        return out.to(input_dtype).to(input_device)

    def _unpack_weight_torch(self, device=None):
        """Unpack uint32 weight tensor to integer values (PyTorch fallback)."""
        weight = self.weight.view(torch.int32)
        if device is not None:
            weight = weight.to(device)
        out_features, packed_dim = weight.shape

        if 32 % self.bits == 0:
            elems_per_int = 32 // self.bits
            shifts = torch.arange(elems_per_int, device=weight.device, dtype=torch.int32) * self.bits
            mask = (1 << self.bits) - 1
            unpacked = (weight.unsqueeze(-1) >> shifts) & mask
            unpacked = unpacked.reshape(out_features, -1)
        else:
            num_groups_32 = packed_dim // self.bits
            weight_reshaped = weight.reshape(out_features, num_groups_32, self.bits).to(torch.int64)
            unpacked = torch.zeros(out_features, num_groups_32, 32, dtype=torch.int64, device=weight.device)
            mask = (1 << self.bits) - 1
            for i in range(32):
                val = 0
                for b in range(self.bits):
                    abs_bit = i * self.bits + b
                    word_idx = abs_bit // 32
                    bit_pos = abs_bit % 32
                    val |= ((weight_reshaped[:, :, word_idx] >> bit_pos) & 1) << b
                unpacked[:, :, i] = val & mask
            unpacked = unpacked.reshape(out_features, -1).to(torch.int32)

        return unpacked[:, : self.infeatures]

    def forward(self, x):
        if MLX_AVAILABLE:
            return self._forward_mlx(x)
        return self._forward_torch(x)

    @classmethod
    def from_gptq(cls, module: nn.Module, sym: bool = False) -> "QuantLinearMLX":
        """Convert a GPTQ-format QuantLinear module into a QuantLinearMLX module.

        Repacks qweight/qzeros/scales (GPTQ format) into weight/scales/biases (MLX format)
        so that ``mx.quantized_matmul`` can be used for hardware-accelerated inference on
        Apple Silicon.

        Args:
            module: The source GPTQ-style quantized linear module (must expose
                ``qweight``, ``scales``, ``bits``, ``group_size`` and optionally ``qzeros``).
            sym: Whether symmetric quantization was used. For symmetric quantization the
                zero-point is fixed to ``2**(bits-1)`` which avoids the GPTQ ``qzeros`` ±1
                offset ambiguity.

        Returns:
            A fully initialized ``QuantLinearMLX`` module ready for inference.
        """
        # Use CPU for conversion to avoid MPS compatibility issues with bitwise ops
        convert_device = torch.device("cpu")

        bits = module.bits
        group_size = module.group_size
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError(f"Unsupported bits={bits} for GPTQ->MLX conversion.")

        qweight = module.qweight.to(convert_device)  # [in//32*bits, out] int32
        scales_gptq = module.scales.to(convert_device)  # [groups, out] float16

        out_features = scales_gptq.shape[1]
        in_features = qweight.shape[0] * 32 // bits

        # Unpack GPTQ qweight -> int weight [in_features, out_features]
        if bits in [2, 4, 8]:
            elems_per_int = 32 // bits
            wf = torch.arange(0, 32, bits, dtype=torch.int32, device=convert_device)
            weight_int = torch.bitwise_right_shift(
                qweight.unsqueeze(1).expand(-1, elems_per_int, -1),
                wf.unsqueeze(-1),
            ).to(torch.int16 if bits == 8 else torch.int8)
            weight_int = torch.bitwise_and(weight_int, (1 << bits) - 1)
            weight_int = weight_int.reshape(-1, out_features)
        else:  # bits == 3
            from auto_round_extension.torch.qlinear_torch import get_wf_3bits_tensor

            wf = get_wf_3bits_tensor(convert_device)
            weight_int = qweight.reshape(qweight.shape[0] // 3, 3, 1, qweight.shape[1]).expand(-1, -1, 12, -1)
            weight_int = (weight_int >> wf.unsqueeze(-1)) & 0x7
            weight_int[:, 0, 10] = (weight_int[:, 0, 10] & 0x3) | ((weight_int[:, 1, 0] << 2) & 0x4)
            weight_int[:, 1, 11] = (weight_int[:, 1, 11] & 0x1) | ((weight_int[:, 2, 0] << 1) & 0x6)
            weight_int = weight_int & 0x7
            weight_int = torch.cat(
                [weight_int[:, 0, :11], weight_int[:, 1, 1:12], weight_int[:, 2, 1:11]], dim=1
            )
            weight_int = weight_int.reshape(-1, out_features)

        # Determine zeros
        if sym:
            zero_val = 2 ** (bits - 1)
            zeros = torch.full(scales_gptq.shape, zero_val, dtype=torch.int32, device=convert_device)
        else:
            if not hasattr(module, "qzeros") or module.qzeros is None:
                zero_val = 2 ** (bits - 1)
                zeros = torch.full(scales_gptq.shape, zero_val, dtype=torch.int32, device=convert_device)
            else:
                qzeros = module.qzeros.to(convert_device)
                if bits in [2, 4, 8]:
                    elems_per_int = 32 // bits
                    wf = torch.arange(0, 32, bits, dtype=torch.int32, device=convert_device)
                    zeros = torch.bitwise_right_shift(
                        qzeros.unsqueeze(2).expand(-1, -1, elems_per_int),
                        wf.unsqueeze(0),
                    ).to(torch.int16 if bits == 8 else torch.int8)
                    zeros = torch.bitwise_and(zeros, (1 << bits) - 1)
                    zeros = zeros.reshape(scales_gptq.shape)
                else:  # bits == 3
                    from auto_round_extension.torch.qlinear_torch import get_wf_3bits_tensor

                    wf = get_wf_3bits_tensor(convert_device)
                    zeros = qzeros.reshape(qzeros.shape[0], qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
                    zeros = zeros >> wf.unsqueeze(0)
                    zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
                    zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
                    zeros = zeros & 0x7
                    zeros = torch.cat(
                        [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]], dim=2
                    )
                    zeros = zeros.reshape(scales_gptq.shape)

        # GPTQ: w = scale * (w_int - zero)
        # MLX:  w = mlx_scale * w_int + mlx_bias
        #   => mlx_scale = scale, mlx_bias = -scale * zero
        from auto_round.export.export_to_mlx.export import _pack_weight_mlx

        weight_int_t = weight_int.t().contiguous().to(torch.int32)
        scales_mlx = scales_gptq.t().contiguous().to(torch.float16)
        biases_mlx = (-scales_gptq.float().t().contiguous() * zeros.float().t().contiguous()).to(torch.float16)
        packed_weight = _pack_weight_mlx(weight_int_t, bits)

        has_bias = getattr(module, "bias", None) is not None
        mlx_layer = cls(
            bits=bits,
            group_size=group_size,
            infeatures=in_features,
            outfeatures=out_features,
            bias=has_bias,
        )
        mlx_layer.weight.copy_(packed_weight)
        mlx_layer.scales.copy_(scales_mlx)
        mlx_layer.biases.copy_(biases_mlx)
        if has_bias:
            mlx_layer.bias.copy_(module.bias.to(torch.float16))
        return mlx_layer

