#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Torch.nn.Module Class Definition."""
# Note: Do not import this file unless you have already imported torch,
# since the model classes inherit torch.nn.Module.
import math

import numpy as np
import torch
from packaging.version import Version
from torch.autograd import Function
from torch.nn import functional as F

from auto_round.utils import can_pack_with_numba, logger

NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
FP4_BNB = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -0.0625, 0, 0.0625, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_E2M1 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.0625, 0, 0.0625, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# the order is the same as float list, bit value range is [-7, 7]
# 1111 = -1, 1110 = -2, 1101= -3, ...

NF4_BIT = [7, 1, 2, 3, 4, 5, 6, 0, -8, -7, -6, -5, -4, -3, -2, -1]
FP4_BNB_BIT = [-5, -6, -3, -4, -1, -2, -7, 0, 1, 6, 7, 4, 5, 2, 3]
FP4_E2M1_BIT = [-1, -2, -3, -4, -5, -6, -7, 0, 1, 2, 3, 4, 5, 6, 7]

FLOAT_MAPPING = {
    "nf4": NF4,
    "fp4": FP4_BNB,
    "fp4_e2m1_bnb": FP4_BNB,
    "fp4_e2m1": FP4_E2M1,
}
INT_MAPPING = {
    "nf4": NF4_BIT,
    "fp4": FP4_BNB_BIT,
    "fp4_e2m1_bnb": FP4_BNB_BIT,
    "fp4_e2m1": FP4_E2M1_BIT,
}


def get_torch_version():
    try:
        torch_version = torch.__version__.split("+")[0]
    except ValueError as e:  # pragma: no cover
        assert False, "Got an unknown version of torch: {}".format(e)
    version = Version(torch_version)
    return version


PT_VERSION = get_torch_version().release


class WeightOnlyLinear(torch.nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        bits,
        groupsize,
        dtype="int",
        zp=False,
        bias=False,
        scale_dtype=torch.float16,
        compression_dtype=torch.int32,
        compression_dim=1,
        device="cpu",
        use_optimum_format=True,
        use_legacy_pack=False,
    ):
        super().__init__()
        self.use_optimum_format = use_optimum_format
        self.dtype = dtype
        if "int" not in self.dtype:  # for nf4, fp4

            float_list = FLOAT_MAPPING[self.dtype]
            int_list = INT_MAPPING[self.dtype]
            self.int2float_mapping = {}
            for k, v in zip(int_list, float_list):
                self.int2float_mapping[k] = v
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else in_features
        self.compression_dim = compression_dim
        assert compression_dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ], "Only support torch.int8|16|32|64 as compressed dtype."
        dtype_bits_mapping = {torch.int8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64}
        self.compress_bits = dtype_bits_mapping[compression_dtype]
        self.n_pack = self.compress_bits // self.bits
        # K is input channel, N is output channel
        assert compression_dim in [0, 1], (
            "Only support 0 or 1 as compression dimension, " + "0 is output channel, 1 is input channel."
        )
        if self.use_optimum_format:
            self.float_type = scale_dtype
            self.compression_dtype = torch.int32
            self.register_buffer(
                "scales",
                torch.zeros(
                    (math.ceil(in_features / self.groupsize), out_features),
                    dtype=self.float_type,
                ).to(device),
            )
            self.register_buffer(
                "qweight",
                torch.zeros(
                    (math.ceil(in_features / self.n_pack), out_features),
                    dtype=self.compression_dtype,
                ).to(device),
            )
            self.register_buffer(
                "qzeros",
                torch.zeros(
                    (math.ceil(self.in_features / self.groupsize), math.ceil(self.out_features / self.n_pack)),
                    dtype=self.compression_dtype,
                ).to(device),
            )
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=self.float_type).to(device))
        else:
            self.compression_dtype = compression_dtype
            self.float_type = scale_dtype
            self.register_buffer(
                "scales",
                torch.zeros(
                    (out_features, math.ceil(in_features / self.groupsize)),
                    dtype=self.float_type,
                ).to(device),
            )
            if compression_dim == 1:
                self.register_buffer(
                    "qweight",
                    torch.zeros(
                        (out_features, math.ceil(in_features / self.n_pack)),
                        dtype=self.compression_dtype,
                    ).to(device),
                )
                if zp:
                    self.register_buffer(
                        "qzeros",
                        torch.zeros(
                            (self.out_features, math.ceil(self.in_features / self.groupsize / self.n_pack)),
                            dtype=self.compression_dtype,
                        ).to(device),
                    )
            else:
                self.register_buffer(
                    "qweight",
                    torch.zeros(
                        (math.ceil(out_features / self.n_pack), in_features),
                        dtype=self.compression_dtype,
                    ).to(device),
                )
                if zp:
                    self.register_buffer(
                        "qzeros",
                        torch.zeros(
                            (math.ceil(self.out_features / self.n_pack), math.ceil(self.in_features / self.groupsize)),
                            dtype=self.compression_dtype,
                        ).to(device),
                    )
            if bias:
                self.register_buffer("bias", torch.zeros(self.out_features, dtype=self.float_type).to(device))
            else:
                self.bias = None
        self.pack_func = self.pack_tensor_native_impl if use_legacy_pack else self.pack_tensor

    def pack_tensor_native_impl(self, raw_tensor):
        mask = torch.tensor(2**self.bits - 1, dtype=self.compression_dtype).to(self.device)
        target_len = math.ceil(raw_tensor.shape[1] / self.n_pack)
        packed_tensor = torch.zeros(raw_tensor.shape[0], target_len, dtype=self.compression_dtype).to(raw_tensor.device)
        for j in range(packed_tensor.shape[1]):
            start = self.n_pack * j
            end = self.n_pack * (j + 1)
            tmp = raw_tensor[:, start:end].type(self.compression_dtype)
            for e in range(tmp.shape[1]):
                tmp[:, e] &= mask
                tmp[:, e] = tmp[:, e] << (self.bits * e)
                packed_tensor[:, j] |= tmp[:, e]
        return packed_tensor

    def pack(self, int_weight, scale, zp, bias):
        if self.use_optimum_format:
            self.scales = self.scales.t_().contiguous()
            self.qweight = self.qweight.t_().contiguous()
            self.qzeros = self.qzeros.t_().contiguous()

        int_weight = int_weight.to(self.device)
        if self.use_optimum_format and zp is None:
            # to avoid overflow
            int_weight = int_weight.type(torch.int32)
            shift_bias = 2 ** (self.bits - 1)
            int_weight += shift_bias
            zp = torch.zeros_like(scale, dtype=torch.uint8) + shift_bias
        if bias is not None:
            assert hasattr(self, "bias"), "bias is not set when initializing."
            self.bias = bias.type(self.float_type).to(self.device)
        assert (
            scale.shape == self.scales.shape
        ), f"Scale shape is mismatched, expect {self.scales.shape}, got {scale.shape}"
        self.scales = scale.type(self.float_type).to(self.device)
        if not self.use_optimum_format and self.compression_dim == 0:
            int_weight = int_weight.t_().contiguous()
            self.qweight = self.qweight.t_().contiguous()
        origin_shape = int_weight.shape
        target_shape = self.qweight.shape
        assert origin_shape[0] == target_shape[0], "output channels mismatch, please check."

        # pack weight
        self.qweight.copy_(self.pack_func(int_weight))
        if not self.use_optimum_format and self.compression_dim == 0:
            self.qweight = self.qweight.t_().contiguous()

        if zp is not None:
            zp = zp.to(self.device) if isinstance(zp, torch.Tensor) else zp
            if self.use_optimum_format:
                zp -= 1
            if self.use_optimum_format or self.compression_dim == 0:
                zp = zp.t_().contiguous()
                self.qzeros = self.qzeros.t_().contiguous()
            assert hasattr(self, "qzeros"), "zp is not set when initializing."
            self.qzeros.copy_(self.pack_func(zp))
            if self.use_optimum_format or self.compression_dim == 0:
                self.qzeros = self.qzeros.t_().contiguous()
        if self.use_optimum_format:
            self.scales = self.scales.t_().contiguous()
            self.qweight = self.qweight.t_().contiguous()
            self.qzeros = self.qzeros.t_().contiguous()

    def recover(self):
        logger.debug(f"Recovering {self} weight")
        scales = self.scales.t_().contiguous() if self.use_optimum_format else self.scales
        qweight = self.qweight.t_().contiguous() if self.use_optimum_format else self.qweight

        device = scales.device
        fp32_weight = torch.zeros(self.out_features, self.in_features, dtype=self.float_type).to(device)
        mask = torch.tensor(2**self.bits - 1, dtype=self.compression_dtype).to(device)
        if hasattr(self, "qzeros"):
            weight_dtype = torch.uint8
        else:
            weight_dtype = torch.int8
        # unpack weight
        weight = torch.zeros(self.out_features, self.in_features, dtype=weight_dtype).to(device)
        if not self.use_optimum_format and self.compression_dim == 0:
            weight = weight.t_().contiguous()
            qweight = qweight.t_().contiguous()
        origin_shape = weight.shape
        target_shape = qweight.shape
        for j in range(target_shape[1]):
            for e in range(self.n_pack):
                index = j * self.n_pack + e
                if index >= origin_shape[1]:
                    continue
                tmp = qweight[:, j]
                tmp = tmp << (self.compress_bits - self.bits * (e + 1))
                tmp = tmp >> self.compress_bits - self.bits
                if weight_dtype == torch.uint8:
                    tmp &= mask  # remove sign bit
                weight[:, index] = tmp.type(weight_dtype)
        if not self.use_optimum_format and self.compression_dim == 0:
            weight = weight.t_().contiguous()
        if "int" not in self.dtype:
            new_weight = torch.zeros(self.out_features, self.in_features).to(device)
            for k, v in self.int2float_mapping.items():
                new_weight += torch.where(weight == k, v, 0)
            weight = new_weight
        # unpack zero_point
        if hasattr(self, "qzeros"):
            zp_dtype = self.compression_dtype  # to avoid overflow when weight-zp
            zp = torch.zeros(scales.shape, dtype=zp_dtype).to(device)
            qzeros = self.qzeros.t_().contiguous() if self.use_optimum_format else self.qzeros
            if self.use_optimum_format or self.compression_dim == 0:
                zp = zp.t_().contiguous()
                qzeros = qzeros.t_().contiguous()
            origin_shape = zp.shape
            target_shape = qzeros.shape
            for j in range(target_shape[1]):
                for e in range(self.n_pack):
                    index = j * self.n_pack + e
                    if index >= origin_shape[1]:
                        continue
                    tmp = qzeros[:, j]
                    tmp = tmp << (self.compress_bits - self.bits * (e + 1))
                    tmp = tmp >> self.compress_bits - self.bits
                    tmp &= mask
                    zp[:, index] = tmp.type(zp_dtype)
            if self.use_optimum_format or self.compression_dim == 0:
                zp = zp.t_().contiguous()
            if self.use_optimum_format:
                # zp -= 1 may cause zp == -1, after recover it becomes 2**self.bits - 1
                zp += 1
                zp = torch.where(zp > (2**self.bits - 1), 0, zp)
            # recover fp32 weight with int_weight, scale, and zero_point
            for idx in range(self.in_features):
                g_idx = idx // self.groupsize
                fp32_weight[:, idx] = (weight[:, idx] - zp[:, g_idx]) * scales[:, g_idx]
        else:
            # recover fp32 weight with int_weight, scale
            for idx in range(self.in_features):
                g_idx = idx // self.groupsize
                fp32_weight[:, idx] = weight[:, idx] * scales[:, g_idx]
        return fp32_weight

    def forward(self, input):
        if not hasattr(self, "weight"):
            weight = self.recover()
            device = self.scales.device
            if weight.dtype == torch.float16 and device.type == "cpu":
                weight = weight.float()
                self.bias = self.bias.float() if self.bias is not None else None
        if True:  # keep reusing self.weight due to recover is too slow.
            if not hasattr(self, "weight"):
                self.weight = weight
            input = input.type(self.weight.dtype)
            logger.debug(f"Calculating {self}")
            return F.linear(input, self.weight, self.bias)  # pylint: disable=E1102
        else:
            input = input.type(weight.dtype)
            return F.linear(input, weight, self.bias)  # pylint: disable=E1102

    def extra_repr(self) -> str:
        tmp_str = "in_features={}, out_features={}, bits={}, group_size={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.bits,
            self.groupsize,
            self.bias is not None,
        )
        if self.use_optimum_format:
            tmp_str += ", use_optimum_format=True"
        return tmp_str

    def pack_tensor_with_torch(self, raw_tensor):
        """Pack the tensor with torch.

        Args:
            raw_tensor (tensor): raw tensor.

        Returns:
            tensor: packed tensor.
        """
        target_len = math.ceil(raw_tensor.shape[1] / self.n_pack)
        packed_tensor = torch.zeros(raw_tensor.shape[0], target_len, dtype=self.compression_dtype).to(raw_tensor.device)
        mask = torch.tensor(2**self.bits - 1, dtype=self.compression_dtype).to(raw_tensor.device)
        for j in range(packed_tensor.shape[1]):
            start = self.n_pack * j
            end = self.n_pack * (j + 1)
            tmp = raw_tensor[:, start:end].type(self.compression_dtype)
            tmp &= mask
            for e in range(tmp.shape[1]):
                tmp[:, e] = tmp[:, e] << (self.bits * e)
                packed_tensor[:, j] |= tmp[:, e]
        return packed_tensor

    def unpack_tensor_with_torch(self, packed_tensor):
        """Unpack the tensor with torch.

        Args:
            packed_tensor (tensor): packed tensor.

        Returns:
            tensor: unpacked tensor.
        """
        target_dtype = torch.int16
        target_len = packed_tensor.shape[1] * self.n_pack
        unpacked_tensor = torch.zeros(packed_tensor.shape[0], target_len, dtype=target_dtype).to(packed_tensor.device)
        mask = torch.tensor(2**self.bits - 1, dtype=self.compression_dtype).to(packed_tensor.device)
        for j in range(packed_tensor.shape[1]):
            for e in range(self.n_pack):
                index = j * self.n_pack + e
                tmp = packed_tensor[:, j]
                tmp = tmp << (self.compress_bits - self.bits * (e + 1))
                tmp = tmp >> self.compress_bits - self.bits
                if hasattr(self, "qzeros"):
                    tmp &= mask  # remove sign bit
                unpacked_tensor[:, index].copy_(tmp.type(target_dtype))
        return unpacked_tensor

    def pack_array_with_numba(
        self,
        raw_array: np.ndarray,
        n_pack: int,
        bits: int,
        compress_bits: int,
        compression_dtype=np.int32,
    ) -> np.ndarray:
        """Packs the input array by combining elements into a specified bit-width format using NumPy.

        Args:
            raw_array (np.ndarray): The array to be packed. Shape: [out_features, in_features] or [1, in_features].
            n_pack (int): The number of elements to be packed together.
            bits (int): The number of bits for each element.
            compress_bits (int): The number of bits for each element of the compressed array, supported 2, 4, 8.
            compression_dtype (np.dtype, optional): The data type of the compressed array. Defaults to np.int32.

        Returns:
            np.ndarray: The packed array.
        """
        # Try to pack with numba to accelerate the packing process.
        # If numba is not availabll or the packing method is not supported,
        # fallback to the torch implementation.
        if not can_pack_with_numba():
            return self.pack_tensor_with_torch(torch.from_numpy(raw_array)).cpu().numpy()

        from auto_round.export.export_to_itrex.bit_packer import bit_packers

        pack_func_name = (bits, compress_bits)
        if pack_func_name not in bit_packers:
            logger.warning(
                (
                    f"Unsupported packing with bits: {bits}, compress_bits: {compress_bits} using numba, "
                    "fallback to torch implementation."
                )
            )
            return self.pack_tensor_with_torch(torch.from_numpy(raw_array)).cpu().numpy()
        out_features, in_features = raw_array.shape
        new_in_features = (in_features + n_pack - 1) // n_pack
        packed_array = np.zeros((out_features, new_in_features), dtype=compression_dtype)
        raw_array = raw_array.astype(compression_dtype)
        pack_method = bit_packers[pack_func_name]
        return pack_method(raw_array, packed_array, n_pack, new_in_features)

    def pack_tensor_with_numpy_impl(self, raw_tensor):
        """The implement of packing tensor with numpy."""
        raw_array = raw_tensor.cpu().numpy()
        target_len = np.ceil(raw_array.shape[1] / self.n_pack).astype(int)
        target_dtype = torch.tensor(0, dtype=self.compression_dtype).numpy().dtype
        packed_array = np.zeros((raw_array.shape[0], target_len), dtype=target_dtype)
        mask = np.uint8(2**self.bits - 1)
        for j in range(packed_array.shape[1]):
            start = self.n_pack * j
            end = self.n_pack * (j + 1)
            tmp = raw_array[:, start:end].astype(target_dtype)
            tmp &= mask
            for e in range(tmp.shape[1]):
                tmp[:, e] = np.left_shift(tmp[:, e], self.bits * e)
                packed_array[:, j] |= tmp[:, e]
        packed_tensor = torch.from_numpy(packed_array).to(device=raw_tensor.device)
        return packed_tensor

    def pack_tensor_with_numpy(self, raw_tensor):
        """Pack the tensor with numpy."""
        if self.bits == 8 and self.compression_dtype == torch.int8:
            return raw_tensor
        if self.bits not in [2, 4, 8]:
            return self.pack_tensor_with_numpy_impl(raw_tensor)
        compression_dtype = torch.tensor(0, dtype=self.compression_dtype).numpy().dtype
        packed_array = self.pack_array_with_numba(
            raw_tensor.cpu().numpy(),
            self.n_pack,
            self.bits,
            self.compress_bits,
            compression_dtype,
        )
        return torch.from_numpy(packed_array).to(device=raw_tensor.device)

    def unpack_tensor_with_numpy(self, packed_tensor):
        """Unpack the packed tensor with numpy."""
        packed_array = packed_tensor.cpu().numpy()
        target_dtype = np.int16
        if self.bits == 8 and self.compression_dtype == torch.int8 and hasattr(self, "qzeros"):
            # special case for unpacking uint8 date from int8 compression_dtype
            target_dtype = np.uint8
        target_len = packed_array.shape[1] * self.n_pack
        unpacked_array = np.zeros((packed_array.shape[0], target_len), dtype=target_dtype)
        mask = np.uint8(2**self.bits - 1)
        for j in range(packed_array.shape[1]):
            for e in range(self.n_pack):
                index = j * self.n_pack + e
                tmp = packed_array[:, j]
                tmp = np.left_shift(tmp, self.compress_bits - self.bits * (e + 1))
                tmp = np.right_shift(tmp, self.compress_bits - self.bits)
                if hasattr(self, "qzeros"):
                    tmp &= mask
                unpacked_array[:, index] = tmp.astype(target_dtype)
        unpacked_tensor = torch.from_numpy(unpacked_array).to(device=packed_tensor.device)
        return unpacked_tensor

    def pack_tensor(self, raw_tensor):
        """Pack tensor."""
        if "cuda" in raw_tensor.device.type:
            return self.pack_tensor_with_torch(raw_tensor)
        else:
            return self.pack_tensor_with_numpy(raw_tensor)

    def unpack_tensor(self, packed_tensor):
        """Unpack tensor."""
        if "cuda" in packed_tensor.device.type:
            return self.unpack_tensor_with_torch(packed_tensor)
        else:
            return self.unpack_tensor_with_numpy(packed_tensor)
