import torch
import math
import torch.nn as nn
from loguru import logger
from auto_gptq.nn_modules.qlinear.qlinear_exllamav2 import ext_make_q_matrix, ext_gemm_half_q_half

def float8_e4m3fn_ste(x: torch.Tensor):
    """Straight-Through Estimator (STE) for float8.

    Applies a quantization and dequantization step with float8 precision while maintaining
    gradient flow using a straight-through estimator.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Quantized and dequantized tensor using float8 format.
    """
    fp8 = (x.to(torch.float8_e4m3fn).to(x.dtype) - x).detach() + x

    return fp8

def input_qdq(tensor, act_scale):
    orig_dtype = tensor.dtype
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    fp8_res = (tensor / act_scale)
    fp8_res = torch.clip(fp8_res, -fp8_max, fp8_max)
    # fp8_res = float8_e4m3fn_ste(fp8_res)
    qdq_res = fp8_res * act_scale
    qdq_res = qdq_res.to(orig_dtype)
    return qdq_res



class QuantLinear(nn.Module):
    QUANT_TYPE = "exllamav2"

    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    def __init__(self, bits, group_size, infeatures, outfeatures, bias, trainable=False, **kwargs):
        super().__init__()
        if bits != 4:
            raise ValueError(
                f"Exllamav2 kernel supports only bits=4, requested bits={bits}. Something is wrong in the model initialization."
            )
        if trainable:
            raise NotImplementedError("Exllamav2 kernel does not support training.")

        self.q_handle = None
        self.q_tensors = None
        self.padding = -outfeatures % 32

        self.infeatures = infeatures
        self.outfeatures = outfeatures + self.padding
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.trainable = trainable
        self.maxq = 2**self.bits - 1

        assert infeatures % 32 == 0
        assert infeatures % self.group_size == 0
        assert outfeatures % 32 == 0

        # I need to register the tensors, otherwise, we won't be able to load them easily using transformers ...
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
        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32),
        )
        self.register_buffer(
            "act_scales",
            torch.zeros(
                1,
                dtype=torch.float16,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def post_init(self, temp_dq):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None
        self.q_tensors = {
            "qweight": self.qweight,
            "qzeros": self.qzeros,
            "scales": self.scales,
            "g_idx": self.g_idx,
        }
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
        self.q_handle = ext_make_q_matrix(self.q_tensors, temp_dq)

    def forward(self, x, force_cuda=False):
        x = torch.clip(x, -65504, 65504)
        if x.dtype != torch.float16:
            logger.warning_once(
                f"The exllama v2 kernel for GPTQ requires a float16 input activation, while {x.dtype} was passed. Casting to float16.\nMake sure you loaded your model with torch_dtype=torch.float16, that the model definition does not inadvertently cast to float32, or disable AMP Autocast that may produce float32 intermediate activations in the model."
            )

            x = x.half()

        x = input_qdq(x, self.act_scales)
        x = torch.clip(x, -65504, 65504)

        output = ext_gemm_half_q_half(x, self.q_handle, self.outfeatures, force_cuda)

        if self.bias is not None:
            output.add_(self.bias)
        output = torch.clip(output, -65504, 65504)
        return output

    def temp_dq_size(self):
        return self.infeatures * self.outfeatures * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.outfeatures * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)