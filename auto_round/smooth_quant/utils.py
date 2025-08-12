import copy
from collections import UserDict, defaultdict

import torch 
from tqdm import tqdm

from auto_round.utils import logger
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad

def get_module(model, key):
    """Get module from model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    module = model
    name_list = key.split(".")
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
        elif hasattr(module, "sq_linear"):  # for peft models
            module = getattr(module, "sq_linear")
            module = getattr(module, name)
        elif hasattr(module, "orig_layer"):  # for peft models and auto alpha
            module = getattr(module, "orig_layer")
            module = getattr(module, name)
        else:
            module = module
    return module

def set_module(model, key, new_module):
    """Set new module into model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
        new_module (torch.nn.Module): new module to be inserted
    """
    module = model
    name_list = key.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
        elif hasattr(module, ("sq_linear")):  # for peft models that Linears are contained in Linear
            module = getattr(module, "sq_linear")
            module = getattr(module, name)
        elif hasattr(module, ("orig_layer")):  # for peft models and auto alpha
            module = getattr(module, "orig_layer")
            module = getattr(module, name)
        else:
            module = module

    if hasattr(module, "sq_linear") and name_list[-1] != "sq_linear":  # for peft models
        module = getattr(module, "sq_linear")
    if hasattr(module, "orig_layer") and name_list[-1] != "orig_layer":  # for peft models and auto alpha
        module = getattr(module, "orig_layer")
    setattr(module, name_list[-1], new_module)

def mul_scale(tensor, scale, group_size=-1):
    ori_shape = tensor.shape
    if len(scale.shape) == 2 and scale.shape[1] == 1:
        tensor = tensor.reshape(scale.shape[0], -1)
    else:
        tensor = tensor.reshape(-1, scale.shape[-1])

    tensor *= scale
    return tensor.reshape(ori_shape)

def reshape_scale_as_input(layer, scale):
    """Reshape the scale for input feature in channel
    :param layer:

    :param scale:
    :return:
    """
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer
    if isinstance(layer, torch.nn.Conv2d):
        scale = scale.view(1, scale.shape[0], 1, 1)

    elif isinstance(layer, torch.nn.Linear):
        scale = scale.view(1, scale.shape[0])

    return scale


def reshape_scale_as_weight(layer, scale):
    """Reshape the scale for weight input channel, depthwise output channel
    :param layer:  torch module
    :param scale: orig scale
    :return: reshaped scale."""
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer
    if isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:  ##only depthwise conv could hit here
        scale = scale.view(scale.shape[0], 1, 1, 1)  ##mount on output channel

    elif isinstance(layer, torch.nn.Conv2d):
        scale = scale.view(1, scale.shape[0], 1, 1)

    elif isinstance(layer, torch.nn.Linear):
        scale = scale.view(1, scale.shape[0])
    return scale

def move_input_to_device(input, device=torch.device("cpu")):
    if isinstance(input, dict) or isinstance(input, UserDict):
        tmp_input = {}
        for k, inp in input.items():
            tmp_input[k] = move_input_to_device(inp, device)
        input = tmp_input
    elif isinstance(input, list) or isinstance(input, tuple):
        is_tuple = isinstance(input, tuple)
        tmp_input = []
        for inp in input:
            tmp_input.append(move_input_to_device(inp, device))
        input = tuple(tmp_input) if is_tuple else tmp_input
    elif isinstance(input, torch.Tensor):
        input = input.to(device)  # pylint: disable=no-member
    return input

def forward_wrapper(model, input, device=torch.device("cpu")):
    try:
        model = model.to(device)
        input = move_input_to_device(input, device)
    except Exception as e:
        logger.warning(e)
        logger.warning("Please check the input device if the error raised.")
    if isinstance(input, dict) or isinstance(input, UserDict):
        output = model(**input)
    elif isinstance(input, list) or isinstance(input, tuple):
        try:
            output = model(*input)
        except:
            output = model(input)
    else:
        output = model(input)
    return output


def model_forward_per_sample(model, sample, device):
    try:
        output = forward_wrapper(model, sample, device)
        return output

    except Exception as e:
        output = forward_wrapper(model, sample[0], device)
        return output


def model_forward(model, dataloader, iters, device):
    cnt = 0
    pbar = tqdm(dataloader, total=iters)
    pbar.set_description("SmoothQuant Calibrating")
    for idx, input in enumerate(pbar):
        output = forward_wrapper(model, input, device)
        cnt += 1
        if iters != -1 and cnt > iters:
            break
    pbar.close()

def cal_scale(input_max_abs, weights, alpha, weight_max_lb=1e-5, group_size=-1):
    weights = torch.cat(weights, dim=0)
    weights, _, _ = reshape_pad_tensor_by_group_size(weights, group_size)
    weight_max = torch.max(torch.abs(weights), dim=0)[0]
    weight_max = torch.clip(weight_max, weight_max_lb)
    input_power = torch.pow(input_max_abs, alpha)
    # logger.debug(f"{max(input_max_abs)}, {min(input_max_abs)}")
    weight_power = torch.pow(weight_max, 1 - alpha)
    weight_scale = torch.clip(input_power / weight_power, min=1e-5)
    weight_scale[input_power == 0] = 1.0
    return weight_scale


def reshape_in_channel_to_last(layer_name, model):
    """Move the input channel to the last dim
    :param layer_name: Layer name
    :return: The reshaped weight."""
    layer = get_module(model, layer_name)
    if layer.__class__.__name__ == "WrapperLayer":
        layer = layer.orig_layer

    weight = layer.weight  ##TODO oc*ic, support transposed conv
    if len(weight.shape) == 4:
        weight = weight.permute(0, 2, 3, 1)
        weight = weight.reshape(-1, weight.shape[-1])
    return weight


def enough_memo_store_scale(device, need_space):
    if device == "cuda":  # pragma: no cover
        current_gpu_index = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory
        used_memory = torch.cuda.memory_allocated(current_gpu_index)
        free_space = total_memory - used_memory
    else:
        import psutil

        free_space = psutil.virtual_memory().free
    return free_space >= need_space


def quant_dequant(m, num_bits=4, group_size=32, data_type='mx_fp4', sym=True):
    from auto_round.data_type.utils import get_quant_func
    # data_type = 'int_asym'
    data_type = 'mx_fp4'
    tensor = m.weight if hasattr(m, "weight") else m
    quant_func, data_type = get_quant_func(data_type, num_bits, sym)
    # print(quant_func, num_bits)
    data_new, scale, zp = quant_func(tensor, bits=num_bits, group_size=group_size, v=0, max_scale=1.0)
    return data_new.to(tensor.dtype)

class WrapperLayer(torch.nn.Module):
    def __init__(self, layer, input_min, input_max, save_q_input=False, group_size=-1):
        super(WrapperLayer, self).__init__()
        if hasattr(layer, "orig_layer"):
            layer = layer.orig_layer
        self.add_module("orig_layer", layer)  # set orig_layer in get/set_module
        self.quant = False
        self.q_input = None
        self.fp32_output = None
        self.input_max = input_max
        self.input_min = input_min
        self.weight_scale = None
        self.input_scale = None
        self.absorb_scale = None
        self.save_q_input = save_q_input
        self.do_blockwise = False
        self.group_size = group_size

    def enable_quant(self):
        self.quant = True

    def disable_quant(self):
        self.quant = False

    def update_scale(self, input_scale, weight_scale, absorb_scale=None):
        self.input_scale = input_scale
        self.weight_scale = weight_scale
        self.absorb_scale = absorb_scale

    ##TODO better tradeoff performance and memory, currently it's too slow
    def q_dq_forward(self, x, input_scale, weight_scale, absorb_scale):
        layer_copy = copy.deepcopy(self.orig_layer)
        if absorb_scale is not None:
            ori_shape = layer_copy.weight.shape
            layer_copy.weight.data = mul_scale(layer_copy.weight, absorb_scale, group_size=self.group_size)
            layer_copy.weight.data = layer_copy.weight.view(ori_shape)
        if weight_scale is not None:
            ori_shape = layer_copy.weight.shape
            # layer_copy.weight *= weight_scale
            layer_copy.weight.data = mul_scale(layer_copy.weight, weight_scale, group_size=self.group_size)
            layer_copy.weight.data = layer_copy.weight.view(ori_shape)
        # q_dq_weight = quant_dequant_w_v1(layer_copy)
        q_dq_weight = quant_dequant(layer_copy)
        layer_copy.weight.data.copy_(q_dq_weight)
        if input_scale is None:
            # x = quant_dequant_x_v1(x, self.input_min, self.input_max)
            x = quant_dequant(x)
        else:
            ori_shape = x.shape
            # x = input_scale * x
            x = mul_scale(x, input_scale)
            # x = quant_dequant_x_v1(x, self.input_min * input_scale, self.input_max * input_scale)  ##FIXME
            x = quant_dequant(x)  ##FIXME
        output = layer_copy(x)
        return output

    def q_dq_forward_blockwise(self, x, input_scale):
        layer_copy = copy.deepcopy(self.orig_layer)
        if input_scale is None:
            # x = quant_dequant_x_v1(x, self.input_min, self.input_max)
            x = quant_dequant(x)
        else:
            x, orig_shape, pad_len = reshape_pad_tensor_by_group_size(x, self.group_size)
            x = input_scale * x
            x = revert_tensor_by_pad(x, orig_shape, pad_len)
            # x = quant_dequant_x_v1(x, self.input_min * input_scale, self.input_max * input_scale)  ##FIXME
            x = quant_dequant(x)  ##FIXME
        output = layer_copy(x)
        return output

    def forward(self, x):
        if self.quant:
            # self.q_input = x * scale ##save the q_input
            if self.save_q_input:
                self.q_input = x
            if not self.do_blockwise:
                output = self.q_dq_forward(x, self.input_scale, self.weight_scale, self.absorb_scale)
            else:
                output = self.q_dq_forward_blockwise(x, self.input_scale)

        else:
            output = self.orig_layer(x)
        self.output = output
        return output
