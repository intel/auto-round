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

import re
import copy
import torch
from torch.functional import F
import transformers
from auto_round_diff.data_type import get_quant_func, get_static_quant_func
from .utils import (
    check_to_quantized,
    get_scale_shape,
    set_module,
    logger
)
import logging
logger = logging.getLogger("autoround")

def reshape_and_pad_tensor(v, group_size=-1):
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


class WrapperLinear(torch.nn.Module):
    """A wrapper for linear/conv1d layers to enable quantization and tuning.

    This module wraps an existing linear or conv1d layer and provides additional functionality
    for quantization, parameter tuning, and activation/bias normalization.

    Args:
        orig_layer (torch.nn.Module): The original layer to be wrapped (linear or conv1d).
        enable_minmax_tuning (bool): Whether to enable min-max scale tuning.
        enable_norm_bias_tuning (bool): Whether to enable normalization and tuning of the bias term.
        device (str): Device on which to run computations (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, orig_layer, keys, device='cpu', **kwargs):
        """Initializes the WrapperLinear module.

        Args:
            orig_layer (torch.nn.Module): The original layer to wrap.
            enable_minmax_tuning (bool): Whether to enable min-max scale tuning.
            enable_norm_bias_tuning (bool): Whether to enable normalization and tuning for the bias term.
            device (str): The computation device, such as 'cpu' or 'cuda'.
        """
        super(WrapperLinear, self).__init__()
        self.keys = keys
        self.orig_layer = orig_layer
        self.split = 0
        self.output_device = device
        self.enable_act_quant = self.orig_layer.quant_act
        self.q_scale_thresh = 1e-5
        self._init_quant_params_and_func()
        self.use_weight_quant = False
        self.use_act_quant = False

        if isinstance(self.orig_layer, torch.nn.Conv2d):
            self.orig_kwargs = dict(stride=self.orig_layer.stride, padding=self.orig_layer.padding,
                                   dilation=self.orig_layer.dilation, groups=self.orig_layer.groups)
            self.orig_forward = F.conv2d
        elif isinstance(self.orig_layer, torch.nn.Conv1d):
            self.orig_kwargs = dict(stride=self.orig_layer.stride, padding=self.orig_layer.padding,
                                   dilation=self.orig_layer.dilation, groups=self.orig_layer.groups)
            self.orig_forward = F.conv1d
        else:
            self.orig_kwargs = dict()
            self.orig_forward = F.linear

    def _init_quant_params_and_func(self):
        self.w_quant_params, self.act_quant_params = {"inited": False, "scale": None, "zp": None, }, {"inited": False, "scale": None, "zp": None}
        orig_layer = self.orig_layer
        for key in self.keys:
            if hasattr(self.orig_layer, key):
                if bool(re.search(r'w_|weight_', key)):
                    new_key = re.sub(r'w_|weight_', '', key)
                    self.w_quant_params[new_key] = getattr(orig_layer, key)
                elif bool(re.search(r'a_|act_', key)):
                    new_key = re.sub(r'a_|act_', '', key)
                    self.act_quant_params[new_key] = getattr(orig_layer, key)

        orig_weight = getattr(orig_layer, "get_weight", lambda: orig_layer.weight)()
        if isinstance(self.orig_layer, torch.nn.Conv1d):
            orig_weight = orig_weight.t()
    
        self.weight_quant_func, self.data_type_w = get_static_quant_func(orig_layer.data_type_w, orig_layer.weight_bits,
                                                                orig_layer.sym_w)

        if self.enable_act_quant:
            self.act_quant_func, self.data_type_act = get_static_quant_func(orig_layer.data_type_act,
                                                                     orig_layer.act_bits,
                                                                     orig_layer.sym_act)

    def _qdq_weight(self):
        """Quantizes and dequantizes weights with tuning parameters.

        Args:
            value (torch.Tensor): Value added for rounding for tuning.
            min_scale (torch.Tensor): Minimum scale for the min value of quantization.
            max_scale (torch.Tensor): Maximum scale for the max value of quantization.

        Returns:
            tuple: Quantized weight, scale, and zero point.
        """
        weight = self.orig_layer.weight
        if isinstance(self.orig_layer, torch.nn.Conv1d):
            weight = weight.t()

        if self.split != 0:
                weight_0, self.w_quant_params["scale"], self.w_quant_params["zp"], self.w_quant_params["inited"] = self.weight_quant_func(weight[:, :self.split, ...], **self.w_quant_params)
                weight_1, self.w_quant_params_0["scale"], self.w_quant_params_0["zp"], self.w_quant_params_0["inited"] = self.weight_quant_func(weight[:, self.split:, ...], **self.w_quant_params_0)
                weight_q = torch.cat([weight_0, weight_1], dim=1)
        else:
            weight_q, self.w_quant_params["scale"], self.w_quant_params["zp"], self.w_quant_params["inited"] = self.weight_quant_func(weight, **self.w_quant_params)
        
        weight_q = weight_q.to(weight.dtype)
        
        if isinstance(self.orig_layer, torch.nn.Conv1d):
            weight_q = weight_q.t()
        return weight_q

    def _qdq_act(self, x):
        """Quantizes and dequantizes activations.

        Args:
            x (torch.Tensor): Input activations.
            act_max_scale (torch.Tensor): Maximum scale for the act_max
            act_max (torch.Tensor, optional): Maximum value for activation quantization. Defaults to None.

        Returns:
            tuple: Quantized activation, scale, and zero point.
        """
        if self.split != 0:
            x, scale, zp = self.act_quant_func(x[:, :self.split, :, :], **self.act_quant_params)
            x_0, scale_0, zp_0 = self.act_quant_func(x[:, self.split:, :, :], **self.act_quant_params_0)
            x = torch.cat([x, x_0], dim=1)

            self.act_quant_params['scale'], self.act_quant_params['zp'] = scale, zp
            self.act_quant_params0['zp'], self.act_quant_params0['zp'] = scale_0, zp_0 
        else:
            x, scale, zp = self.act_quant_func(x, **self.act_quant_params)
            self.act_quant_params['scale'], self.act_quant_params['zp'] = zp

        return x, scale, zp, x_0, scale_0, zp_0

    def _qdq_bias(self, bias, bias_v):
        """Quantizes and dequantizes bias.

        Args:
            bias (torch.Tensor): Bias tensor to be quantized.
            bias_v (torch.Tensor): Value added for rounding for tuning.

        Returns:
            tuple: Quantized bias, scale, and zero point.
        """
        bias_bits = 4  ## hard code
        bias_group_size = -1
        bias, scale, zp = self.bias_quant_func(bias, bits=bias_bits, group_size=bias_group_size, v=bias_v,
                                               q_scale_thresh=self.q_scale_thresh)
        return bias, scale, zp

    def unwrapper(self, best_params):
        """Restores the original layer by applying the best tuning parameters.

        Args:
            best_params (dict): Dictionary containing the best tuning parameters.

        Returns:
            torch.nn.Module: The unwrapped and restored original layer.
        """
        best_params = best_params or {}
        v = best_params.get('value', torch.tensor(0.0)).to(self.device)
        min_scale = best_params.get('min_scale', torch.tensor(1.0)).to(self.device)
        max_scale = best_params.get('max_scale', torch.tensor(1.0)).to(self.device)

        if self.orig_layer.weight.device.type == 'meta':
            self.orig_layer.to(self.device)
        ##unwrapper weight
        qdq_weight, scale, zp = self._qdq_weight(v, min_scale, max_scale)

        self.orig_layer.weight.data.copy_(qdq_weight)
        self.orig_layer.weight.grad = None

        shape = qdq_weight.shape
        if isinstance(self.orig_layer, transformers.modeling_utils.Conv1D):
            shape = qdq_weight.t().shape

        def _set_dict_attr(attr_dict, attr_name):
            for key in attr_dict.keys():
                if key == attr_name:
                    setattr(self.orig_layer, attr_name, attr_dict[key].reshape(shape[0], -1).to("cpu"))
                else:
                    name = "w_" + key
                    setattr(self.orig_layer, name, attr_dict[key].to("cpu"))

        if isinstance(scale, dict):
            _set_dict_attr(scale, "scale")
        else:
            self.orig_layer.scale = scale.reshape(shape[0], -1).to("cpu")

        if zp is not None:
            if isinstance(zp, dict):
                _set_dict_attr(zp, "zp")
            else:
                zp = zp.reshape(shape[0], -1)
                self.orig_layer.zp = zp.to("cpu") if zp is not None else None
        else:
            self.orig_layer.zp = None

        ##unwrapper bias
        if self.enable_norm_bias_tuning and "bias_v" in best_params.keys():  ##fake quant
            bias_v = best_params["bias_v"].to(self.device)
            bias = self.orig_layer.bias
            if bias is not None and bias.device.type == 'meta':
                bias = self.orig_layer.get_bias().to(self.device)
            bias, _, _ = self._qdq_bias(bias, bias_v)
            self.orig_layer.bias.grad = None
            self.orig_layer.bias.data.copy_(bias)

        if hasattr(self.orig_layer, 'update'):
            self.orig_layer.update()
            self.orig_layer.to('meta')

        ##unwrapper act
        if self.enable_act_quant:
            if not self.orig_layer.act_dynamic:
                act_max_scale = best_params.get('act_max_scale', torch.tensor(1.0)).to(self.device)
                act_max = self.orig_layer.act_max if hasattr(self.orig_layer, "act_max") else None
                tmp_shape = (1)
                if self.orig_layer.act_group_size > 1:
                    tmp_shape = (1, self.orig_layer.act_group_size)
                _, act_scale, _ = self._qdq_act(torch.zeros(tmp_shape).to(self.device),
                                                act_max_scale=self.act_max_scale, act_max=act_max)
                self.orig_layer.act_max = torch.tensor(self.orig_layer.act_max * act_max_scale.item()).to("cpu")
                self.orig_layer.act_scale = act_scale.to("cpu")

            self.orig_layer.q_scale_thresh = self.q_scale_thresh
            self.orig_layer.data_type = self.data_type

            self.orig_layer.act_data_type = self.act_data_type
            self.orig_layer.act_quant_func = self.act_quant_func
            wrapper_layer = WrapperWALayer(self.orig_layer)
            return wrapper_layer

        return self.orig_layer

    def set_split(self):
        self.w_quant_params_0 = copy.deepcopy(self.w_quant_params)
        self.act_quant_params_0 = copy.deepcopy(self.act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_running_stat(self, running_stat_a: bool):
        self.act_quant_params["running_stat_a"] = running_stat_a
        if self.split != 0:
            self.act_quant_params_0["running_stat_a"] = running_stat_a

    def forward(self, x, split=0):
        """Executes the forward pass with quantized weights and optional bias/activation quantization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the wrapped layer.
        """
        x = x.to(self.output_device)

        if split != 0 and self.split != 0:
            assert(split == self.split)
        elif split != 0:
            logger.info(f"split at {split}!")
            self.split = split
            self.set_split()

        if self.enable_act_quant and self.use_act_quant:
            x, _, _, _, _, _ = self._qdq_act(x)
        if self.use_weight_quant:
            weight = self._qdq_weight()
        else:
            weight = self.org_weight
        output = self.orig_forward(x, weight, self.orig_layer.bias, **self.orig_kwargs).to(self.output_device)
       
        return output


class WrapperWALayer(torch.nn.Module):
    def __init__(self, orig_layer):
        super(WrapperWALayer, self).__init__()
        self.orig_layer = orig_layer
        self.act_quant_func = self.orig_layer.act_quant_func

    def forward(self, x):
        act_max = self.orig_layer.act_max if hasattr(self.orig_layer, "act_max") else None
        x, _, _ = self.orig_layer.act_quant_func(x, bits=self.orig_layer.act_bits,
                                                 group_size=self.orig_layer.group_size,
                                                 scale_dtype=self.orig_layer.scale_dtype,
                                                 q_scale_thresh=self.orig_layer.q_scale_thresh,
                                                 data_type=self.orig_layer.act_data_type,
                                                 tensor_max=act_max)
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
        self.device = self.orig_layer.tuning_device if hasattr(self.orig_layer, "tuning_device") else device
        self.output_device = device
        weight_dtype = torch.float32
        self.q_scale_thresh = 1e-5
        self.v = torch.nn.Parameter(
            reshape_and_pad_tensor(
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
        weight_q, _, _ = self.quant_func(self.orig_layer.weight, self.bits, self.group_size,
                                         v, q_scale_thresh=self.q_scale_thresh)
        self.orig_layer.q_scale_thresh = self.q_scale_thresh
        self.orig_layer.weight.data.copy_(weight_q)
        return self.orig_layer

    def forward(self, input):
        input = input.to(self.device)
        weight_q, _, _ = self.quant_func(self.orig_layer.weight, self.bits, self.group_size,
                                         self.v, q_scale_thresh=self.q_scale_thresh)
        import torch.nn.functional as F
        return F.layer_norm(
            input, self.orig_layer.normalized_shape, weight_q, self.orig_layer.bias, self.orig_layer.eps).to(
            self.output_device)


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
        self.device = self.orig_layer.tuning_device if hasattr(self.orig_layer, "tuning_device") else device
        self.output_device = device
        weight_dtype = torch.float32
        self.q_scale_thresh = 1e-5
        self.v = torch.nn.Parameter(
            reshape_and_pad_tensor(
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
        weight_q, _, _ = self.quant_func(self.orig_layer.weight, self.bits, self.group_size,
                                         v, q_scale_thresh=self.q_scale_thresh)
        self.orig_layer.q_scale_thresh = self.q_scale_thresh
        self.orig_layer.weight.data.copy_(weight_q)
        return self.orig_layer

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(self.device)
        weight_q, _, _ = self.quant_func(self.orig_layer.weight, self.bits, self.group_size,
                                         self.v, q_scale_thresh=self.q_scale_thresh)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.orig_layer.variance_epsilon)
        return (weight_q * hidden_states.to(input_dtype)).to(self.output_device)


norm_mapping = {}
norm_mapping["LayerNorm"] = WrapperLayerNorm
norm_mapping["LlamaRMSNorm"] = WrapperLlamaNorm
norm_mapping["Qwen2RMSNorm"] = WrapperLlamaNorm
norm_mapping["Phi3RMSNorm"] = WrapperLlamaNorm
norm_mapping["MistralRMSNorm"] = WrapperLlamaNorm


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


def wrapper_block(block, enable_minmax_tuning, enable_norm_bias_tuning, device='cpu', **kwargs):
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
        if isinstance(m, (torch.nn.Linear, transformers.modeling_utils.Conv1D)):
            if not check_to_quantized(m):
                unquantized_layers.append(n)
                continue
            new_m = WrapperLinear(m, enable_minmax_tuning=enable_minmax_tuning,
                                  enable_norm_bias_tuning=enable_norm_bias_tuning, device=device,
                                  **kwargs,
                                  )
            set_module(block, n, new_m)
            quantized_layers.append(n)

        if enable_norm_bias_tuning:
            if "norm" in m.__class__.__name__.lower():
                if m.__class__.__name__ in norm_mapping.keys():
                    wrapper_layer_class = norm_mapping[m.__class__.__name__]
                    new_m = wrapper_layer_class(m, device=device)
                    setattr(block, n, new_m)
                elif "RMSNorm" in m.__class__.__name__:
                    logger.warning_once(
                        f"use LlamaRMSNorm to wrap {m.__class__.__name__}, please check the correctness yourself")
                    wrapper_layer_class = norm_mapping["LlamaRMSNorm"]
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
