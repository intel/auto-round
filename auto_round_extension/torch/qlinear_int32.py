import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers

from auto_round_extension.torch.torch_utils.mixin import TritonModuleMixin

logger = getLogger(__name__)

try:
    from auto_round_extension.triton.triton_utils.dequant import QuantLinearFunction, quant_matmul_248
except ImportError as e:
    triton_import_exception = e

    def error_raiser_triton(*args, **kwargs):
        raise ValueError(
            f"Trying to use the triton backend, but could not import triton dependencies with the following error: {triton_import_exception}"
        )

    class FakeTriton:
        def __getattr__(self, name):
            raise ImportError(
                f"Trying to use the triton backend, but could not import triton dependencies with the following error: {triton_import_exception}"
            )

    quant_matmul_248 = error_raiser_triton
    QuantLinearFunction = FakeTriton
    QuantLinearInferenceOnlyFunction = FakeTriton


class QuantLinear(nn.Module, TritonModuleMixin):
    """
    Triton v2 quantized linear layer.

    Calls dequant kernel (see triton_utils/dequant) to dequantize the weights then uses
    torch.matmul to compute the output whereas original `triton` quantized linear layer fused
    dequant and matmul into single kernel.add()
    """

    QUANT_TYPE = "tritonv2"

    def __init__(
        self, bits, group_size, infeatures, outfeatures, bias, trainable=False, **kwargs
    ):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        if infeatures % 32 != 0 or outfeatures % 32 != 0:
            raise NotImplementedError(
                "in_feature and out_feature must be divisible by 32."
            )
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1

        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(infeatures / self.group_size),
                    outfeatures // 32 * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(infeatures / self.group_size), outfeatures),
                dtype=torch.float16,
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros((outfeatures), dtype=torch.float16)
            )
        else:
            self.bias = None

        self.trainable = trainable
        
        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)
        elif self.bits == 3:
            self.wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32,
            ).reshape(1, 3, 12)
            

    def post_init(self):
        pass

    def pack(self, linear, scales, zeros, g_idx=None):
        scales_t = scales.t().contiguous()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
        self.scales = scales_t.clone().half()
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.xpu.is_available():
            device = "xpu:0"

        W = linear.weight.data.to(device).clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        repeat_scales = scales.to(device).repeat_interleave(self.group_size, 1)
        if isinstance(zeros, torch.Tensor):
            repeat_zeros = zeros.to(device).repeat_interleave(self.group_size, 1)
            intweight = torch.round(W.to(device) / repeat_scales[:, :W.shape[1]] + repeat_zeros[:, :W.shape[1]]).to(
                torch.int32)
        else:
            repeat_zeros = zeros
            intweight = torch.round(W.to(device) / repeat_scales[:, :W.shape[1]] + repeat_zeros).to(
                torch.int32)

        del repeat_scales
        
        if self.bits in [2, 4, 8]:
            intweight = intweight.reshape(-1, intweight.shape[1] // 32 * self.bits, 32 // self.bits)
            order_map = torch.arange(0, 32 // self.bits, device=device) * self.bits
            intweight = intweight.to(torch.int32)
            intweight = intweight << order_map
            intweight = torch.sum(intweight, dim=-1)

            intweight = intweight.t().contiguous().to(torch.int32)
            self.qweight = intweight.to("cpu")
        elif self.bits == 3:
            intweight = intweight.t().contiguous()
            intweight = intweight.cpu().numpy().astype(np.uint32)
            i = 0
            row = 0
            qweight = torch.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=torch.int32)
            while row < qweight.shape[0]:
                packed_weight = torch.tensor(intweight[i : i + 10]).to(dtype=torch.int32).t()
                shifts = torch.arange(0, 10) * self.bits
                shifted = (packed_weight << shifts)
                qweight[row] |= shifted.sum(dim=-1)
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                packed_weight = torch.tensor(intweight[i : i + 10]).to(dtype=torch.int32).t()
                shifts = torch.arange(0, 10) * self.bits + 1  
                shifted = packed_weight << shifts 
                qweight[row] |= shifted.sum(dim=-1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                packed_weight = torch.tensor(intweight[i : i + 10]).to(dtype=torch.int32).t()
                shifts = torch.arange(0, 10) * self.bits + 2
                shifted = packed_weight << shifts 
                qweight[row] |= shifted.sum(dim=-1)
                i += 10
                row += 1
                
            self.qweight = qweight.cpu()
            
        if isinstance(zeros, torch.Tensor):
            zeros = zeros.t().contiguous()
            zeros = zeros.numpy().astype(np.uint32)
            qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=torch.int32)
            i = 0
            col = 0
            while col < qzeros.shape[1]:
                if self.bits in [2, 4, 8]:
                    packed_zeros = torch.tensor(zeros[:, i: i + (32 // self.bits)]).to(dtype=torch.int32)
                    shifts = torch.arange(0, (32 // self.bits)) * self.bits
                    shifted = packed_zeros << shifts
                    qzeros[:, col] |= shifted.sum(dim=-1)
                    i += 32 // self.bits
                    col += 1
                elif self.bits == 3:
                    packed_zeros = torch.tensor(zeros[:, i : i + 10]).to(dtype=torch.int32)
                    shifts = torch.arange(0, 10) * self.bits
                    shifted = packed_zeros << shifts                  
                    qzeros[:, col] = shifted.sum(dim=-1)
                    i += 10
                    qzeros[:, col] |= zeros[:, i] << 30
                    col += 1
                    qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                    i += 1
                    packed_zeros = torch.tensor(zeros[:, i : i + 10]).to(dtype=torch.int32)
                    shifts = torch.arange(0, 10) * self.bits + 1  
                    shifted = packed_zeros << shifts 
                    qzeros[:, col] |= shifted.sum(dim=-1)
                    i += 10
                    qzeros[:, col] |= zeros[:, i] << 31
                    col += 1
                    qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                    i += 1
                    packed_zeros = torch.tensor(zeros[:, i : i + 10]).to(dtype=torch.int32) 
                    shifts = torch.arange(0, 10) * self.bits + 2
                    shifted = packed_zeros << shifts 
                    qzeros[:, col] |= shifted.sum(dim=-1)
                    i += 10
                    col += 1

            self.qzeros = qzeros.cpu()
        else:
            shape = scales_t.shape
            value = 0
            # need optimum for bits == 3
            for j in range(0, (32 // self.bits)):
                value |= zeros << (self.bits * j)
            qzeros = np.ones((shape[0], shape[1] // 32 * self.bits), dtype=np.uint32) * value
            qzeros = qzeros.astype(np.int32)
            self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype
        if not hasattr(self, "g_idx"):
                self.g_idx = torch.tensor(
                    [i // self.group_size for i in range(self.infeatures)], dtype=torch.int32
                ).to(self.qweight.device)
        if self.bits in [2, 4, 8]:
            quant_linear_fn = QuantLinearFunction
            out = quant_linear_fn.apply(
                x,
                self.qweight,
                self.scales,
                self.qzeros,
                self.g_idx,
                self.bits,
                self.maxq,
            )
            out = out.half().reshape(out_shape)
            out = out + self.bias if self.bias is not None else out
            return out.to(x.dtype)
        elif self.bits == 3:
            if self.wf.device != self.qzeros.device:
                self.wf = self.wf.to(self.qzeros.device)
            zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(
                -1, -1, -1, 12
            )
            zeros = zeros >> self.wf.unsqueeze(0)
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
            zeros = zeros & 0x7
            zeros = torch.cat(
                [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
                dim=2,
            )

            zeros = zeros.reshape(self.scales.shape)

            weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(
                -1, -1, 12, -1
            )
            weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
            weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
            num_itr = self.g_idx.shape[0] // x.shape[-1]
            if num_itr == 1: # for dummy g_idx
                weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])
            else:
                num_dim = self.g_idx.shape[0] // num_itr
                weights = []
                for i in range(num_itr):
                    scale_i = self.scales[:, i * num_dim : (i + 1) * num_dim]
                    weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
                    zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
                    g_idx_i = self.g_idx[i * num_dim : (i + 1) * num_dim]
                    weights.append(scale_i[g_idx_i.long()] * (weight_i - zeros_i[g_idx_i.long()]))
                weights = torch.cat(weights, dim=1)
            out = torch.matmul(x, weights)
            out = out.to(x_dtype)
            out = out.reshape(out_shape)
            out = out + self.bias if self.bias is not None else out
            return out
            

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        """
        Pre-tunes the quantized kernel
        """
        from tqdm import tqdm

        kn_values = {}

        for _, m in model.named_modules():
            if not isinstance(m, cls):
                continue

            k = m.infeatures
            n = m.outfeatures

            if (k, n) not in kn_values:
                kn_values[(k, n)] = (
                    m.qweight,
                    m.scales,
                    m.qzeros,
                    m.g_idx,
                    m.bits,
                    m.maxq,
                )

        logger.info(f"Found {len(kn_values)} unique KN Linear values.")
        logger.info("Warming up autotune cache ...")
        with torch.no_grad():
            for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
                m = 2**m
                for (k, n), (
                    qweight,
                    scales,
                    qzeros,
                    g_idx,
                    bits,
                    maxq,
                ) in kn_values.items():
                    a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                    quant_matmul_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
        del kn_values


__all__ = ["QuantLinear"]
