# Copyright (c) 2024 Intel Corporation
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

import torch
import transformers
from auto_round.data_type import get_quant_func
from .utils import (
    check_to_quantized,
    get_scale_shape,
    set_module,
    logger
)


def reshape_tensor(v, group_size=-1):
    """Reshapes the tensor based on the group size.

    Args:
        v (torch.Tensor): The input tensor to be reshaped.
        group_size (int, optional): The number of elements to group together.

    Returns:
        torch.Tensor: The reshaped tensor. If padding is applied, the padded tensor is returned.
    """
    if group_size == -1 or v.shape[1] < group_size:
        return v
    if v.shape[1] % group_size == 0:
        v = v.reshape(-1, group_size)
    else:
        pad_len = (v.shape[1] + group_size - 1) // group_size * group_size - v.shape[1]
        v = torch.nn.functional.pad(v, (0, pad_len))
        v = v.reshape(-1, group_size)
    return v


def quant_tensor(
        quant_func, data, bits=4, group_size=-1, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16,
        weight_min=None, weight_max=None, q_scale_thresh=1e-5, **kwargs,
):
    """Quantizes and dequantizes weight, handing the group size issue .

    Args:
        data: Tensor containing the weight to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: The number of elements shares scale and zero point
        sym: Sym or asym
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        weight_min (Tensor, optional): Minimum weight value for quantization. Defaults to None.
        weight_max (Tensor, optional): Maximum weight value for quantization. Defaults to None.

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    orig_shape = data.shape
    if len(data.shape) > 2:
        data = data.reshape(-1, orig_shape[-1])
    if group_size == -1 or data.shape[1] < group_size:
        data, scale, zp = quant_func(data, bits, v=v, min_scale=min_scale, max_scale=max_scale,
                                     scale_dtype=scale_dtype, weight_min=weight_min, weight_max=weight_max,
                                     q_scale_thresh=q_scale_thresh, **kwargs)
        data = data.reshape(orig_shape)
        return data, scale, zp

    if data.shape[1] % group_size == 0:
        data = data.reshape(-1, group_size)
        data, scale, zp = quant_func(data, bits, v=v, min_scale=min_scale, max_scale=max_scale,
                                     scale_dtype=scale_dtype, weight_min=weight_min, weight_max=weight_max,
                                     q_scale_thresh=q_scale_thresh, **kwargs)
        data = data.reshape(orig_shape)
        return data, scale, zp

    else:
        tmp_shape = data.shape
        pad_len = (data.shape[1] + group_size - 1) // group_size * group_size - data.shape[1]
        data_new = torch.nn.functional.pad(data, (0, pad_len))
        data_new = data_new.reshape(-1, group_size)
        data_new, scale, zp = quant_func(data_new, bits, v=v, min_scale=min_scale,
                                         max_scale=max_scale, scale_dtype=scale_dtype, weight_min=weight_min,
                                         weight_max=weight_max, q_scale_thresh=q_scale_thresh, **kwargs)
        data_new = data_new.reshape(tmp_shape[0], -1)
        data_new = data_new[:, :-pad_len]
        data_new = data_new.reshape(orig_shape)
        return data_new, scale, zp


class WrapperWALayer(torch.nn.Module):
    def __init__(self, orig_layer):
        super(WrapperWALayer, self).__init__()
        self.orig_layer = orig_layer
        self.act_quant_func = self.orig_layer.act_quant_func

    def forward(self, x):
        x, _, _ = quant_tensor(self.orig_layer.act_quant_func, x, self.orig_layer.act_bits,
                               self.orig_layer.group_size,
                               scale_dtype=self.orig_layer.scale_dtype,
                               q_scale_thresh=self.orig_layer.q_scale_thresh,
                               data_type=self.orig_layer.act_data_type)
        return self.orig_layer.forward(x)


class WrapperLayerNorm(torch.nn.Module):
    """A wrapper for layer normalization with quantized weights.

       This class wraps a given layer normalization module and applies quantization without round
       to its weights. The quantization is parameterized by the number of bits and
       an optional group size.
    """

    def __init__(self, orig_layer, bit=4, group_size=-1, device="cpu"):
        super(WrapperLayerNorm, self).__init__()
        self.orig_layer = orig_layer
        self.bits = bit
        self.group_size = group_size
        self.device = device
        weight_dtype = torch.float32
        self.q_scale_thresh = 1e-5
        self.v = torch.nn.Parameter(
            reshape_tensor(
                torch.zeros(self.orig_layer.weight.shape, device=self.device, dtype=weight_dtype),
                self.group_size),
            requires_grad=True)
        self.params = {"v": self.v}
        from auto_round.data_type.int import quant_tensor_asym_wo_round
        self.quant_func = quant_tensor_asym_wo_round

    def unwrapper(self, best_params):
        if best_params is None:
            return self.orig_layer
        v = best_params['v']
        weight_q, _, _ = quant_tensor(self.quant_func, self.orig_layer.weight, self.bits, self.group_size,
                                      v, q_scale_thresh=self.q_scale_thresh)
        self.orig_layer.q_scale_thresh = self.q_scale_thresh
        self.orig_layer.weight.data.copy_(weight_q)
        return self.orig_layer

    def forward(self, input):
        weight_q, _, _ = quant_tensor(self.quant_func, self.orig_layer.weight, self.bits, self.group_size,
                                      self.v, q_scale_thresh=self.q_scale_thresh)
        import torch.nn.functional as F
        return F.layer_norm(
            input, self.orig_layer.normalized_shape, weight_q, self.orig_layer.bias, self.orig_layer.eps)


class WrapperLlamaNorm(torch.nn.Module):
    """A wrapper for Llama normalization in HF with fake quantized weights without rounding.

       This class wraps a given layer normalization module and applies quantization without rounding
       to its weights. The quantization is parameterized by the number of bits and
       an optional group size.
    """

    def __init__(self, orig_layer, bit=4, group_size=-1, device="cpu"):
        super(WrapperLlamaNorm, self).__init__()
        self.orig_layer = orig_layer
        self.bits = bit
        self.group_size = group_size
        self.device = device
        weight_dtype = torch.float32
        self.q_scale_thresh = 1e-5
        self.v = torch.nn.Parameter(
            reshape_tensor(
                torch.zeros(self.orig_layer.weight.shape, device=self.device, dtype=weight_dtype),
                self.group_size),
            requires_grad=True)
        self.params = {"v": self.v}
        from auto_round.data_type.int import quant_tensor_asym_wo_round
        self.quant_func = quant_tensor_asym_wo_round

    def unwrapper(self, best_params):
        if best_params is None:
            return self.orig_layer
        v = best_params['v']
        weight_q, _, _ = quant_tensor(self.quant_func, self.orig_layer.weight, self.bits, self.group_size,
                                      v, q_scale_thresh=self.q_scale_thresh)
        self.orig_layer.q_scale_thresh = self.q_scale_thresh
        self.orig_layer.weight.data.copy_(weight_q)
        return self.orig_layer

    def forward(self, hidden_states):
        weight_q, _, _ = quant_tensor(self.quant_func, self.orig_layer.weight, self.bits, self.group_size,
                                      self.v, q_scale_thresh=self.q_scale_thresh)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.orig_layer.variance_epsilon)
        return weight_q * hidden_states.to(input_dtype)


norm_mapping = {}
norm_mapping["LayerNorm"] = WrapperLayerNorm
norm_mapping["LlamaRMSNorm"] = WrapperLlamaNorm
norm_mapping["Qwen2RMSNorm"] = WrapperLlamaNorm
norm_mapping["Phi3RMSNorm"] = WrapperLlamaNorm
norm_mapping["MistralRMSNorm"] = WrapperLlamaNorm


class WrapperLinear(torch.nn.Module):
    def __init__(self, orig_layer, enable_minmax_tuning=True, enable_norm_bias_tuning=False, device='cpu'):
        """A wrapper module for linear layers that enables quantization and min-max tuning of weights.

        Args:
        - orig_layer (torch.nn.Module): The original linear layer to be wrapped.
        - enable_minmax_tuning (bool): Whether to enable min-max scaling tuning. Default is True.

        Attributes:
        - orig_layer (torch.nn.Module): The original linear layer being wrapped.
        - bits (int): The number of bits for quantization.
        - group_size (int): The size of the groups for quantization.
        - sym (bool): Whether the symmetric quantization is to be used.
        - value (torch.nn.Parameter): The learnable parameter for quantization.
        - enable_minmax_tuning (bool): Whether min-max scaling tuning is enabled.
        - min_scale (torch.nn.Parameter or torch.Tensor): The minimum scale for min-max tuning.
        - max_scale (torch.nn.Parameter or torch.Tensor): The maximum scale for min-max tuning.
        """
        super(WrapperLinear, self).__init__()
        self.orig_layer = orig_layer
        self.device = device
        self.bits = self.orig_layer.bits
        self.group_size = self.orig_layer.group_size
        self.scale_dtype = self.orig_layer.scale_dtype
        self.sym = self.orig_layer.sym
        self.data_type = self.orig_layer.data_type
        self.weight_quant_func, self.data_type = get_quant_func(self.orig_layer.data_type, self.bits, self.sym)
        self.act_bits = self.orig_layer.act_bits
        self.act_group_size = self.orig_layer.act_group_size
        self.act_sym = self.orig_layer.act_sym
        self.act_dynamic = self.orig_layer.act_dynamic
        self.act_quant = self.act_bits <= 8
        self.params = {}

        if self.act_quant:
            self.act_quant_func, self.act_data_type = get_quant_func(self.orig_layer.data_type, self.act_bits,
                                                                     self.act_sym)

        self.q_scale_thresh = 1e-5

        weight_dtype = torch.float32
        orig_layer_weight = self.orig_layer.weight if not hasattr(self.orig_layer, 'get_weight') \
            else self.orig_layer.get_weight()
        self.value = torch.nn.Parameter(
            reshape_tensor(
                torch.zeros(self.orig_layer.weight.shape, device=self.device, dtype=weight_dtype),
                self.group_size),
            requires_grad=True)
        self.params["v"] = self.value
        weight_reshape = reshape_tensor(orig_layer_weight.data, self.group_size)
        self.weight_min = torch.clamp(weight_reshape.min(1)[0], max=0)
        self.weight_max = torch.clamp(weight_reshape.max(1)[0], min=0)

        self.enable_minmax_tuning = enable_minmax_tuning
        shape = get_scale_shape(self.orig_layer.weight, self.group_size)
        if self.enable_minmax_tuning:
            self.min_scale = torch.nn.Parameter(
                torch.ones(shape, device=self.device, dtype=weight_dtype), requires_grad=True
            )
            self.max_scale = torch.nn.Parameter(
                torch.ones(shape, device=self.device, dtype=weight_dtype), requires_grad=True
            )
            self.params["min_scale"] = self.min_scale
            self.params["max_scale"] = self.max_scale
        else:
            self.min_scale = torch.tensor(1.0, device=self.device, dtype=weight_dtype)
            self.max_scale = torch.tensor(1.0, device=self.device, dtype=weight_dtype)
        self.enable_norm_bias_tuning = False
        if enable_norm_bias_tuning and self.orig_layer.bias is not None:
            self.enable_norm_bias_tuning = True
            self.bias_bits = 4  ## hard code
            self.bias_group_size = -1
            self.bias_v = torch.nn.Parameter(
                reshape_tensor(
                    torch.zeros(self.orig_layer.bias.shape, device=self.device, dtype=weight_dtype),
                    self.bias_group_size),
                requires_grad=True)
            from auto_round.data_type.int import quant_tensor_asym_wo_round
            self.bias_quant_func = quant_tensor_asym_wo_round
            self.params["bias_v"] = self.bias_v

    def unwrapper(self, best_params):
        """Unwrapper the layer to the original layer.

        Args:
        - v (torch.Tensor): The rounding v parameter for quantization.
        - min_scale (torch.nn.Parameter or torch.Tensor): The minimum scale for min-max tuning.
        - max_scale (torch.nn.Parameter or torch.Tensor): The maximum scale for min-max tuning.

        Returns:
        - torch.nn.Module: The original linear layer with updated weights after quantization and dequantization.
        """
        min_scale = torch.tensor(1.0)
        max_scale = torch.tensor(1.0)
        v = torch.tensor(0.0)
        if best_params is not None:
            min_scale = best_params.get('min_scale', min_scale)
            max_scale = best_params.get('max_scale', max_scale)
            v = best_params.get('v', v)

        min_scale.clamp_(0, 1.0)
        max_scale.clamp_(0, 1.0)
        v = v.to(self.device)
        min_scale = min_scale.to(self.device)
        max_scale = max_scale.to(self.device)

        if self.orig_layer.weight.device.type == 'meta':
            self.orig_layer.to(self.device)
        qdq_weight, scale, zp = quant_tensor(self.weight_quant_func, self.orig_layer.weight, self.bits,
                                             self.group_size, v,
                                             min_scale, max_scale, self.scale_dtype, self.weight_min, self.weight_max,
                                             data_type=self.data_type, q_scale_thresh=self.q_scale_thresh)
        scale = scale.reshape(qdq_weight.shape[0], -1)
        if zp is not None:
            zp = zp.reshape(qdq_weight.shape[0], -1)

        self.orig_layer.weight.data.copy_(qdq_weight)
        self.orig_layer.weight.grad = None
        self.orig_layer.scale = scale.to("cpu")
        self.orig_layer.zp = zp.to("cpu") if zp is not None else None
        if self.enable_norm_bias_tuning and "bias_v" in best_params.keys():  ##fake quant
            bias_v = best_params["bias_v"]
            bias, _, _ = quant_tensor(self.bias_quant_func, self.orig_layer.bias, self.bias_bits, self.bias_group_size,
                                      bias_v, q_scale_thresh=self.q_scale_thresh)
            self.orig_layer.bias.grad = None
            self.orig_layer.bias.data.copy_(bias)

        self.orig_layer.q_scale_thresh = self.q_scale_thresh
        self.orig_layer.data_type = self.data_type
        if self.act_quant:
            self.orig_layer.act_data_type = self.act_data_type
            self.orig_layer.act_quant_func = self.act_quant_func
            wrapper_layer = WrapperWALayer(self.orig_layer)
            return wrapper_layer

        if hasattr(self.orig_layer, 'update'):
            self.orig_layer.update()
            self.orig_layer.to('meta')

        return self.orig_layer

    def forward(self, x):
        """Performs forward pass through the wrapped linear layer with quantized weights.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor after applying the linear transformation with quantized weights.
        """
        from torch.functional import F

        weight = self.orig_layer.weight
        if weight.device.type == 'meta':
            weight = self.orig_layer.get_weight().to(self.device)
        self.min_scale.data.copy_(torch.clamp(self.min_scale.data, 0, 1.0))
        self.max_scale.data.copy_(torch.clamp(self.max_scale.data, 0, 1.0))
        weight_q, _, _ = quant_tensor(self.weight_quant_func, weight, self.bits, self.group_size, self.value,
                                      self.min_scale,
                                      self.max_scale, self.scale_dtype, self.weight_min, self.weight_max,
                                      data_type=self.data_type, q_scale_thresh=self.q_scale_thresh)
        weight_q = weight_q.to(weight.dtype)
        if self.act_quant:
            x, _, _ = quant_tensor(self.act_quant_func, x, self.act_bits, self.act_group_size,
                                   scale_dtype=self.scale_dtype, q_scale_thresh=self.q_scale_thresh,
                                   data_type=self.act_data_type)
        # pylint: disable=not-callable
        bias = self.orig_layer.bias
        if bias is not None and bias.device.type == 'meta':
            bias = self.orig_layer.get_bias().to(self.device)
        if self.enable_norm_bias_tuning:
            bias, _, _ = quant_tensor(self.bias_quant_func, bias, self.bias_bits, self.bias_group_size, self.bias_v,
                                      q_scale_thresh=self.q_scale_thresh)

        return F.linear(x, weight_q, bias)


class WrapperTransformerConv1d(torch.nn.Module):
    def __init__(self, orig_layer, enable_minmax_tuning=True, enable_norm_bias_tuning=False, device='cpu'):
        """A wrapper module for transformers 1D convolutional layers used in transformers,
        enabling quantization and min-max tuning of weights.

        Args:
        - orig_layer (torch.nn.Module): The original 1D convolutional layer to be wrapped.
        - bits (int): The number of bits for quantization.
        - group_size (int): The size of the groups for quantization.
        - sym (bool): Whether symmetric quantization is to be used.
        - enable_minmax_tuning (bool): Whether to enable min-max scaling tuning. Default is True.

        Attributes:
        - orig_layer (torch.nn.Module): The original 1D convolutional layer being wrapped.
        - bits (int): The number of bits for quantization.
        - group_size (int): The size of the groups for quantization.
        - sym (bool): Whether symmetric quantization is to be used.
        - weight_t (torch.Tensor): Transposed weight tensor of the original layer.
        - value (torch.nn.Parameter): The learnable parameter for quantization.
        - enable_minmax_tuning (bool): Whether min-max scaling tuning is enabled.
        - min_scale (torch.nn.Parameter or torch.Tensor): The minimum scale for min-max tuning.
        - max_scale (torch.nn.Parameter or torch.Tensor): The maximum scale for min-max tuning.
        """
        super(WrapperTransformerConv1d, self).__init__()
        self.orig_layer = orig_layer
        self.bits = self.orig_layer.bits
        self.group_size = self.orig_layer.group_size
        self.sym = self.orig_layer.sym
        self.scale_dtype = self.orig_layer.scale_dtype
        self.data_type = self.orig_layer.data_type
        self.act_bits = self.orig_layer.act_bits
        self.act_group_size = self.orig_layer.act_group_size
        self.act_sym = self.orig_layer.act_sym
        self.act_dynamic = self.orig_layer.act_dynamic
        self.act_quant = self.act_bits <= 8
        self.weight_quant_func, self.data_type = get_quant_func(self.orig_layer.data_type, self.bits, self.sym)
        if self.act_quant:
            self.act_quant_func, self.act_data_type = get_quant_func(self.orig_layer.data_type, self.act_bits,
                                                                     self.act_sym)

        self.q_scale_thresh = 1e-5
        weight_dtype = torch.float32
        self.device = device
        self.params = {}
        if hasattr(self.orig_layer, 'get_weight'):
            self.weight_t = self.orig_layer.get_weight().t()
        else:
            self.weight_t = self.orig_layer.weight.t()
        self.weight_t = self.weight_t.to(self.device)
        self.value = torch.nn.Parameter(
            reshape_tensor(torch.zeros(self.weight_t.shape, device=device, dtype=weight_dtype),
                           group_size=self.group_size),
            requires_grad=True
        )
        self.params["v"] = self.value
        weight_reshape = reshape_tensor(self.weight_t, self.group_size)
        self.weight_min = torch.clamp(weight_reshape.min(1)[0], max=0)
        self.weight_max = torch.clamp(weight_reshape.max(1)[0], min=0)

        shape = get_scale_shape(self.weight_t, self.group_size)

        if enable_minmax_tuning:
            self.min_scale = torch.nn.Parameter(
                torch.ones(shape, device=device, dtype=weight_dtype), requires_grad=True
            )
            self.max_scale = torch.nn.Parameter(
                torch.ones(shape, device=device, dtype=weight_dtype), requires_grad=True
            )
            self.params["min_scale"] = self.min_scale
            self.params["max_scale"] = self.max_scale

        else:
            self.min_scale = torch.tensor(1.0, device=device, dtype=weight_dtype)
            self.max_scale = torch.tensor(1.0, device=device, dtype=weight_dtype)

        self.enable_norm_bias_tuning = False
        if enable_norm_bias_tuning and self.orig_layer.bias is not None:
            self.enable_norm_bias_tuning = True
            self.bias_bits = 4  ## hard code
            self.bias_group_size = -1
            self.bias_v = torch.nn.Parameter(
                reshape_tensor(
                    torch.zeros(self.orig_layer.bias.shape, device=self.device, dtype=weight_dtype),
                    self.bias_group_size),
                requires_grad=True)
            from auto_round.data_type.int import quant_tensor_asym_wo_round
            self.bias_quant_func = quant_tensor_asym_wo_round
            self.params["bias_v"] = self.bias_v

    def unwrapper(self, best_params):
        """Unwrapper the layer to the original conv1d layer.

        Args:
        - v (torch.Tensor): The scaling parameter for quantization.
        - min_scale (torch.nn.Parameter or torch.Tensor): The minimum scale for min-max tuning.
        - max_scale (torch.nn.Parameter or torch.Tensor): The maximum scale for min-max tuning.

        Returns:
        - torch.nn.Module: The original 1D convolutional layer with updated weights after inverse quantization.
        """
        min_scale = torch.tensor(1.0)
        max_scale = torch.tensor(1.0)
        v = torch.tensor(0.0)
        if best_params is not None:
            min_scale = best_params.get('min_scale', min_scale)
            max_scale = best_params.get('max_scale', max_scale)
            v = best_params.get('v', v)

        min_scale.clamp_(0, 1.0)
        max_scale.clamp_(0, 1.0)
        v = v.to(self.device)
        min_scale = min_scale.to(self.device)
        max_scale = max_scale.to(self.device)

        qdq_weight, scale, zp = quant_tensor(self.weight_quant_func, self.weight_t, self.bits, self.group_size, v,
                                             min_scale,
                                             max_scale, self.scale_dtype, self.weight_min, self.weight_max,
                                             data_type=self.data_type, q_scale_thresh=self.q_scale_thresh)
        scale = scale.reshape(qdq_weight.shape[0], -1)
        if zp is not None:
            zp = zp.reshape(qdq_weight.shape[0], -1)
        if self.orig_layer.weight.device.type == 'meta':
            self.orig_layer.weight.to(self.device)
        self.orig_layer.weight.data.copy_(qdq_weight.t())
        self.orig_layer.weight.grad = None

        if self.enable_norm_bias_tuning and "bias_v" in best_params.keys():  ##fake quant
            bias_v = best_params["bias_v"]
            bias, _, _ = quant_tensor(self.bias_quant_func, self.orig_layer.bias, self.bias_bits, self.bias_group_size,
                                      bias_v, q_scale_thresh=self.q_scale_thresh)
            self.orig_layer.bias.grad = None
            self.orig_layer.bias.data.copy_(bias)

        self.orig_layer.scale = scale.to("cpu")
        self.orig_layer.zp = zp.to("cpu")
        self.orig_layer.q_scale_thresh = self.q_scale_thresh
        self.orig_layer.data_type = self.data_type
        if self.act_quant:
            self.orig_layer.act_quant_func = self.act_quant_func
            self.orig_layer.act_data_type = self.act_data_type
            wrapper_layer = WrapperWALayer(self.orig_layer)
            return wrapper_layer
        if hasattr(self.orig_layer, 'update'):
            self.orig_layer.update()
            self.orig_layer.to('meta')
        return self.orig_layer

    def forward(self, x):
        """Performs forward pass through the wrapped 1D convolutional layer with quantized weights.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor after applying the convolutional transformation with quantized weights.
        """
        with torch.no_grad():
            self.min_scale.clamp_(0, 1.0)
            self.max_scale.clamp_(0, 1.0)
        weight_q, _, _ = quant_tensor(self.weight_quant_func, self.weight_t, self.bits, self.group_size, self.value,
                                      self.min_scale, self.max_scale, self.scale_dtype, self.weight_min,
                                      self.weight_max, data_type=self.data_type, q_scale_thresh=self.q_scale_thresh)
        weight_q = weight_q.to(self.weight_t.dtype)
        size_out = x.size()[:-1] + (self.orig_layer.nf,)
        if self.act_quant:
            x, _, _ = quant_tensor(self.act_quant_func, x, self.act_bits, self.act_group_size,
                                   scale_dtype=self.scale_dtype, q_scale_thresh=self.q_scale_thresh,
                                   data_type=self.act_data_type)
        bias = self.orig_layer.bias
        if self.enable_norm_bias_tuning:
            bias, _, _ = quant_tensor(self.bias_quant_func, bias, self.bias_bits, self.bias_group_size, self.bias_v,
                                      q_scale_thresh=self.q_scale_thresh)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight_q.t())
        x = x.view(*size_out)
        return x


class WrapperMultiblock(torch.nn.Module):
    """A wrapper for a list of modules to be act as a single block.

    Args:
    module_list: The list of modules to wrap.
    """

    def __init__(self, module_list):
        super(WrapperMultiblock, self).__init__()
        self.layers = torch.nn.ModuleList(module_list)

    def forward(self, x, **kwargs):
        hidden_states = x
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(hidden_states, **kwargs)
            hidden_states = layer_outputs
            if isinstance(hidden_states, tuple) or isinstance(hidden_states, list):
                hidden_states = layer_outputs[0]
        return hidden_states


def wrapper_block(block, enable_minmax_tuning, enable_norm_bias_tuning, device='cpu'):
    """Wraps the layers in the given block with a custom Wrapper module.

    Args:
        block: The input block containing linear and conv1d layers to be wrapped.
        enable_minmax_tuning: A boolean indicating whether min-max tuning is enabled.

    Returns:
        list: A list of names of the wrapped layers and unwrapped layers.
    """
    quantized_layers = []
    unquantized_layers = []
    for n, m in block.named_modules():
        if isinstance(m, torch.nn.Linear):
            if not check_to_quantized(m):
                unquantized_layers.append(n)
                continue
            new_m = WrapperLinear(m, enable_minmax_tuning=enable_minmax_tuning,
                                  enable_norm_bias_tuning=enable_norm_bias_tuning, device=device)
            set_module(block, n, new_m)
            quantized_layers.append(n)

        if isinstance(m, transformers.modeling_utils.Conv1D):
            if not check_to_quantized(m):
                unquantized_layers.append(n)
                continue
            new_m = WrapperTransformerConv1d(m, enable_minmax_tuning=enable_minmax_tuning, device=device)
            set_module(block, n, new_m)
            quantized_layers.append(n)

        if enable_norm_bias_tuning:
            if "norm" in m.__class__.__name__.lower():
                if m.__class__.__name__ in norm_mapping.keys():
                    wrapper_layer_class = norm_mapping[m.__class__.__name__]
                    new_m = wrapper_layer_class(m, device=device)
                    setattr(block, n, new_m)
                else:
                    logger.warning_once(f"{m.__class__.__name__} is not supported")

    return quantized_layers, unquantized_layers


@torch.no_grad()
def unwrapper_layer(model, layer, layer_name, best_params):
    """Unwraps the WrapperLinear and WrapperTransformerConv1d modules in the given block.

    Args:
    block: The input block containing wrapped modules to be unwrapped.
    vs: A dictionary of scaling parameters for the wrapped modules.
    min_scales: A dictionary of minimum scaling values for the wrapped modules.
    max_scales: A dictionary of maximum scaling values for the wrapped modules.
    """

    if hasattr(layer, "orig_layer"):
        orig_layer = layer.unwrapper(best_params)
        orig_layer = orig_layer.to("cpu")
        set_module(model, layer_name, orig_layer)


@torch.no_grad()
def unwrapper_block(block, best_params):
    """Unwraps the WrapperLinear and WrapperTransformerConv1d modules in the given block.

    Args:
    block: The input block containing wrapped modules to be unwrapped.
    vs: A dictionary of scaling parameters for the wrapped modules.
    min_scales: A dictionary of minimum scaling values for the wrapped modules.
    max_scales: A dictionary of maximum scaling values for the wrapped modules.
    """
    for n, m in block.named_modules():
        if hasattr(m, "orig_layer"):
            if n in best_params.keys():
                best_param = best_params[n]
            else:
                best_param = None
            orig_layer = m.unwrapper(best_param)
            set_module(block, n, orig_layer)
