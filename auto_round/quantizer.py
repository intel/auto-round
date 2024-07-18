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

from .utils import (
    check_to_quantized,
    get_scale_shape,
    set_module
)


def round_ste(x: torch.Tensor):
    """Straight-Through Estimator for rounding.
    This function is adapted from omniquant.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor
    """
    return (x.round() - x).detach() + x


def quant_tensor_asym(weight, num_bits=4, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16,
                      weight_min=None, weight_max=None, q_scale_thresh=0.0):
    """Quantizes and dequantizes weight asymmetrically.

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        weight_min (Tensor, optional): Minimum weight value for quantization. Defaults to None.
        weight_max (Tensor, optional): Maximum weight value for quantization. Defaults to None.

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** num_bits - 1)
    if isinstance(min_scale, torch.Tensor):
        if weight_min is None or weight_max is None:
            wmin_tmp = torch.clamp(weight.min(1)[0], max=0)
            wmax_tmp = torch.clamp(weight.max(1)[0], min=0)
        else:
            wmin_tmp = weight_min
            wmax_tmp = weight_max
        wmin_tmp = wmin_tmp * min_scale
        wmax_tmp = wmax_tmp * max_scale
        wmax = torch.maximum(wmax_tmp, wmin_tmp)
        wmin = torch.minimum(wmax_tmp, wmin_tmp)
    else:
        if weight_min is None or weight_max is None:
            wmin = torch.clamp(weight.min(1)[0], max=0)
            wmax = torch.clamp(weight.max(1)[0], min=0)
        else:
            wmin = weight_min
            wmax = weight_max

    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    scale = ((wmax - wmin) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh)
    zp = round_ste(-wmin / scale)  # pylint: disable=E1130
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(weight / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    return scale * (q - zp), scale, zp


def quant_tensor_sym(weight, num_bits=4, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16, weight_min=None,
                     weight_max=None, q_scale_thresh=0.0):
    """Quantizes and dequantizes weight symmetrically.

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        weight_min (Tensor, optional): Minimum weight value for quantization. Defaults to None.
        weight_max (Tensor, optional): Maximum weight value for quantization. Defaults to None.

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** num_bits - 1)
    if isinstance(min_scale, torch.Tensor):
        if weight_min is None or weight_max is None:
            wmin_tmp = torch.clamp(weight.min(1)[0], max=0)
            wmax_tmp = torch.clamp(weight.max(1)[0], min=0)
        else:
            wmin_tmp = weight_min
            wmax_tmp = weight_max
        wmin_tmp = wmin_tmp * min_scale
        wmax_tmp = wmax_tmp * max_scale
        wmax = torch.maximum(wmax_tmp, wmin_tmp)
        wmin = torch.minimum(wmax_tmp, wmin_tmp)
    else:
        if weight_min is None or weight_max is None:
            wmin = torch.clamp(weight.min(1)[0], max=0)
            wmax = torch.clamp(weight.max(1)[0], min=0)
        else:
            wmin = weight_min
            wmax = weight_max

    wmax_new = torch.max(wmin.abs(), wmax)
    tmp = wmin < 0
    wmin_new = wmin.clone()  ##must clone, otherwise inplace backward will occur
    if torch.any(tmp):
        wmin_new[tmp] = -wmax_new[tmp]

    tmp = (wmin_new == 0) & (wmax_new == 0)
    wmin_new[tmp] = -1
    wmax_new[tmp] = +1
    scale = ((wmax_new - wmin_new) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh)
    scale = scale.unsqueeze(dim=-1)
    zp = torch.full_like(scale, (maxq + 1) / 2)

    int_w = round_ste(weight / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    return scale * (q - zp), scale, zp


def quant_tensor_actor(weight, num_bits, sym, v, min_scale, max_scale, scale_dtype=torch.float16, weight_min=None,
                       weight_max=None, q_scale_thresh=0.0):
    """Quantizes and dequantizes weight symmetrically or asymmetrically .

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        sym: Sym or asym
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        weight_min (Tensor, optional): Minimum weight value for quantization. Defaults to None.
        weight_max (Tensor, optional): Maximum weight value for quantization. Defaults to None.

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    assert num_bits > 0, "num_bits should be larger than 0"
    if sym:
        return quant_tensor_sym(weight, num_bits, v, min_scale, max_scale, scale_dtype, weight_min, weight_max,
                                q_scale_thresh)
    else:
        return quant_tensor_asym(weight, num_bits, v, min_scale, max_scale, scale_dtype, weight_min, weight_max,
                                 q_scale_thresh)


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
    return v


def quant_tensor(
        data, num_bits=4, group_size=-1, sym=False, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16,
        weight_min=None, weight_max=None, q_scale_thresh=0.0
):
    """Quantizes and dequantizes weight, handing the group size issue .

    Args:
        data: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
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
        data, scale, zp = quant_tensor_actor(data, num_bits, sym=sym, v=v, min_scale=min_scale, max_scale=max_scale,
                                             scale_dtype=scale_dtype, weight_min=weight_min, weight_max=weight_max,
                                             q_scale_thresh=q_scale_thresh)
        data = data.reshape(orig_shape)
        return data, scale, zp

    if data.shape[1] % group_size == 0:
        data = data.reshape(-1, group_size)
        data, scale, zp = quant_tensor_actor(data, num_bits, sym=sym, v=v, min_scale=min_scale, max_scale=max_scale,
                                             scale_dtype=scale_dtype, weight_min=weight_min, weight_max=weight_max,
                                             q_scale_thresh=q_scale_thresh)
        data = data.reshape(orig_shape)
        return data, scale, zp

    else:
        tmp_shape = data.shape
        pad_len = (data.shape[1] + group_size - 1) // group_size * group_size - data.shape[1]
        data_new = torch.nn.functional.pad(data, (0, pad_len))
        data_new = data_new.reshape(-1, group_size)
        data_new, scale, zp = quant_tensor_actor(data_new, num_bits, sym=sym, v=v, min_scale=min_scale,
                                                 max_scale=max_scale, scale_dtype=scale_dtype, weight_min=weight_min,
                                                 weight_max=weight_max, q_scale_thresh=q_scale_thresh)
        data_new = data_new.reshape(tmp_shape[0], -1)
        data_new = data_new[:, :-pad_len]
        data_new = data_new.reshape(orig_shape)
        return data_new, scale, zp


class WrapperWALayer(torch.nn.Module):
    def __init__(self, orig_layer):
        super(WrapperWALayer, self).__init__()
        self.orig_layer = orig_layer

    def forward(self, x):
        x, _, _ = quant_tensor(x, self.orig_layer.act_bits, self.orig_layer.group_size, self.orig_layer.sym,
                               scale_dtype=self.orig_layer.scale_dtype, q_scale_thresh=self.orig_layer.q_scale_thresh)
        return self.orig_layer.forward(x)


class WrapperLinear(torch.nn.Module):
    def __init__(self, orig_layer, enable_minmax_tuning=True, device='cpu'):
        """A wrapper module for linear layers that enables quantization and min-max tuning of weights.

        Args:
        - orig_layer (torch.nn.Module): The original linear layer to be wrapped.
        - enable_minmax_tuning (bool): Whether to enable min-max scaling tuning. Default is True.

        Attributes:
        - orig_layer (torch.nn.Module): The original linear layer being wrapped.
        - num_bits (int): The number of bits for quantization.
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
        self.num_bits = self.orig_layer.bits
        self.group_size = self.orig_layer.group_size
        self.scale_dtype = self.orig_layer.scale_dtype
        self.sym = self.orig_layer.sym
        self.act_bits = self.orig_layer.act_bits
        self.act_group_size = self.orig_layer.act_group_size
        self.act_sym = self.orig_layer.act_sym
        self.act_dynamic = self.orig_layer.act_dynamic
        self.act_quant = self.act_bits <= 8
        self.q_scale_thresh = 1e-5

        weight_dtype = torch.float32
        orig_layer_weight = self.orig_layer.weight if not hasattr(self.orig_layer, 'get_weight') \
            else self.orig_layer.get_weight()
        self.value = torch.nn.Parameter(
            reshape_tensor(
                torch.zeros(self.orig_layer.weight.shape, device=self.device, dtype=weight_dtype),
                self.group_size),
            requires_grad=True)
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
        else:
            self.min_scale = torch.tensor(1.0, device=self.device, dtype=weight_dtype)
            self.max_scale = torch.tensor(1.0, device=self.device, dtype=weight_dtype)

    def unwrapper(self, v, min_scale, max_scale):
        """Unwrapper the layer to the original layer.

        Args:
        - v (torch.Tensor): The rounding v parameter for quantization.
        - min_scale (torch.nn.Parameter or torch.Tensor): The minimum scale for min-max tuning.
        - max_scale (torch.nn.Parameter or torch.Tensor): The maximum scale for min-max tuning.

        Returns:
        - torch.nn.Module: The original linear layer with updated weights after quantization and dequantization.
        """
        min_scale.clamp_(0, 1.0)
        max_scale.clamp_(0, 1.0)

        if self.orig_layer.weight.device.type == 'meta':
            self.orig_layer.to(self.device)
        qdq_weight, scale, zp = quant_tensor(self.orig_layer.weight, self.num_bits, self.group_size, self.sym, v,
                                             min_scale, max_scale, self.scale_dtype, self.weight_min, self.weight_max)
        scale = scale.reshape(qdq_weight.shape[0], -1)
        if zp is not None:
            zp = zp.reshape(qdq_weight.shape[0], -1)
        self.orig_layer.weight.data.copy_(qdq_weight)
        self.orig_layer.weight.grad = None
        self.orig_layer.scale = scale.to("cpu")
        self.orig_layer.zp = zp.to("cpu") if zp is not None else None
        self.orig_layer.q_scale_thresh = self.q_scale_thresh
        if hasattr(self.orig_layer, 'update'):
            self.orig_layer.update()
            self.orig_layer.to('meta')
        # if self.act_quant:
        #     wrapper_layer = WrapperWALayer(self.orig_layer)
        #     return wrapper_layer
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
        weight_q, _, _ = quant_tensor(weight, self.num_bits, self.group_size, self.sym, self.value, self.min_scale,
                                      self.max_scale, self.scale_dtype, self.weight_min, self.weight_max)
        weight_q = weight_q.to(weight.dtype)
        if self.act_quant:
            x, _, _ = quant_tensor(x, self.act_bits, self.act_group_size, self.act_sym,
                                   scale_dtype=self.scale_dtype, q_scale_thresh=self.q_scale_thresh)
        # pylint: disable=not-callable
        bias = self.orig_layer.bias
        if bias is not None and bias.device.type == 'meta':
            bias = self.orig_layer.get_bias().to(self.device)
        return F.linear(x, weight_q, bias)


class WrapperTransformerConv1d(torch.nn.Module):
    def __init__(self, orig_layer, enable_minmax_tuning=True, device='cpu'):
        """A wrapper module for transformers 1D convolutional layers used in transformers,
        enabling quantization and min-max tuning of weights.

        Args:
        - orig_layer (torch.nn.Module): The original 1D convolutional layer to be wrapped.
        - num_bits (int): The number of bits for quantization.
        - group_size (int): The size of the groups for quantization.
        - sym (bool): Whether symmetric quantization is to be used.
        - enable_minmax_tuning (bool): Whether to enable min-max scaling tuning. Default is True.

        Attributes:
        - orig_layer (torch.nn.Module): The original 1D convolutional layer being wrapped.
        - num_bits (int): The number of bits for quantization.
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
        self.num_bits = self.orig_layer.bits
        self.group_size = self.orig_layer.group_size
        self.sym = self.orig_layer.sym
        self.scale_dtype = self.orig_layer.scale_dtype
        self.act_bits = self.orig_layer.act_bits
        self.act_group_size = self.orig_layer.act_group_size
        self.act_sym = self.orig_layer.act_sym
        self.act_dynamic = self.orig_layer.act_dynamic
        self.act_quant = self.act_bits <= 8
        self.q_scale_thresh = 1e-5
        weight_dtype = torch.float32
        self.device = device
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
        else:
            self.min_scale = torch.tensor(1.0, device=device, dtype=weight_dtype)
            self.max_scale = torch.tensor(1.0, device=device, dtype=weight_dtype)

    def unwrapper(self, v=0, min_scale=1.0, max_scale=1.0):
        """Unwrapper the layer to the original conv1d layer.

        Args:
        - v (torch.Tensor): The scaling parameter for quantization.
        - min_scale (torch.nn.Parameter or torch.Tensor): The minimum scale for min-max tuning.
        - max_scale (torch.nn.Parameter or torch.Tensor): The maximum scale for min-max tuning.

        Returns:
        - torch.nn.Module: The original 1D convolutional layer with updated weights after inverse quantization.
        """
        min_scale.clamp_(0, 1.0)
        max_scale.clamp_(0, 1.0)
        qdq_weight, scale, zp = quant_tensor(self.weight_t, self.num_bits, self.group_size, self.sym, v, min_scale,
                                             max_scale, self.scale_dtype, self.weight_min, self.weight_max)
        scale = scale.reshape(qdq_weight.shape[0], -1)
        if zp is not None:
            zp = zp.reshape(qdq_weight.shape[0], -1)
        if self.orig_layer.weight.device.type == 'meta':
            self.orig_layer.weight.to(self.device)
        self.orig_layer.weight.data.copy_(qdq_weight.t())
        self.orig_layer.weight.grad = None
        self.orig_layer.scale = scale.to("cpu")
        self.orig_layer.zp = zp.to("cpu")
        self.orig_layer.q_scale_thresh = self.q_scale_thresh
        if self.act_quant:
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
        weight_q, _, _ = quant_tensor(self.weight_t, self.num_bits, self.group_size, self.sym, self.value,
                                      self.min_scale, self.max_scale, self.scale_dtype, self.weight_min,
                                      self.weight_max)
        weight_q = weight_q.to(self.weight_t.dtype)
        size_out = x.size()[:-1] + (self.orig_layer.nf,)
        if self.act_quant:
            x, _, _ = quant_tensor(x, self.act_bits, self.act_group_size, self.act_sym,
                                   scale_dtype=self.scale_dtype, q_scale_thresh=self.q_scale_thresh)
        x = torch.addmm(self.orig_layer.bias, x.view(-1, x.size(-1)), weight_q.t())
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


def wrapper_block(block, enable_minmax_tuning, device='cpu'):
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
            new_m = WrapperLinear(m, enable_minmax_tuning=enable_minmax_tuning, device=device)
            set_module(block, n, new_m)
            quantized_layers.append(n)

        if isinstance(m, transformers.modeling_utils.Conv1D):
            if not check_to_quantized(m):
                unquantized_layers.append(n)
                continue
            new_m = WrapperTransformerConv1d(m, enable_minmax_tuning=enable_minmax_tuning, device=device)
            set_module(block, n, new_m)
            quantized_layers.append(n)

    return quantized_layers, unquantized_layers


@torch.no_grad()
def unwrapper_layer(model, layer, layer_name, v=0, min_scale=0, max_scale=0):
    """Unwraps the WrapperLinear and WrapperTransformerConv1d modules in the given block.

    Args:
    block: The input block containing wrapped modules to be unwrapped.
    vs: A dictionary of scaling parameters for the wrapped modules.
    min_scales: A dictionary of minimum scaling values for the wrapped modules.
    max_scales: A dictionary of maximum scaling values for the wrapped modules.
    """

    if hasattr(layer, "orig_layer"):

        if isinstance(min_scale, torch.Tensor):
            min_scale = torch.clamp(min_scale, 0, 1.0)
            max_scale = torch.clamp(max_scale, 0, 1.0)

        else:
            min_scale = torch.tensor(1.0)
            max_scale = torch.tensor(1.0)
        orig_layer = layer.unwrapper(v, min_scale, max_scale)
        orig_layer = orig_layer.to("cpu")
        set_module(model, layer_name, orig_layer)


@torch.no_grad()
def unwrapper_block(block, vs, min_scales, max_scales):
    """Unwraps the WrapperLinear and WrapperTransformerConv1d modules in the given block.

    Args:
    block: The input block containing wrapped modules to be unwrapped.
    vs: A dictionary of scaling parameters for the wrapped modules.
    min_scales: A dictionary of minimum scaling values for the wrapped modules.
    max_scales: A dictionary of maximum scaling values for the wrapped modules.
    """
    for n, m in block.named_modules():
        if hasattr(m, "orig_layer"):
            v = 0
            min_scale = torch.tensor(1.0)
            max_scale = torch.tensor(1.0)
            if isinstance(vs, dict):
                v = vs[n]
            if isinstance(min_scales, dict):
                min_scale = min_scales[n]
                min_scale = torch.clamp(min_scale, 0, 1.0)
            if isinstance(max_scales, dict):
                max_scale = max_scales[n]
                max_scale = torch.clamp(max_scale, 0, 1.0)
            orig_layer = m.unwrapper(v, min_scale, max_scale)
            set_module(block, n, orig_layer)
