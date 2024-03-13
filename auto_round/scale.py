# [x] insert scale calculator ar `WrapperLinear`
#   [x] `init` insert `self.input_scale_calculator = ScaleCalculatorV(module.in_features, module.weight.device)`
#   [x] add parameter of `self.input_scale_calculator` into optimizer
#   [x] `forward` transform `input` and `weight`
# [x] at the `unwrapper` stage, replace the original `Linear` with `MulLinear`
# [ ] save and export



import torch
from .utils import logger

def get_scale_param_from_block(block: torch.nn.Module):
    scale_params = []
    for name, mod in block.named_modules():
        if hasattr(mod, "weight_scale_calculator"):
            scale_params.extend( mod.weight_scale_calculator.parameters())
    return scale_params

def _transform_weight(weight, weight_scale):
    updated_weight = weight * weight_scale.reshape(1, -1)
    return updated_weight
    
def _transform_input(x, weight_scale):
    input_scale_target_shape = (1,) * (len(x.shape) - 1) + (-1,)
    input_scale_for_x = weight_scale.reshape(input_scale_target_shape)
    updated_x = torch.div(x, input_scale_for_x)
    return updated_x

def equalization_transform(weight, x, weight_scale):
    updated_x = _transform_input(x, weight_scale)
    updated_weight = _transform_weight(weight, weight_scale)
    return updated_weight, updated_x

class MulLinear(torch.nn.Module):
    def __init__(self, module, weight_scale=None):
        """A forward hook to save input max of a module
        :param module: the linear module
        :param input_scale: scale for weight."""
        super().__init__()
        if weight_scale is None:
            weight_scale = torch.ones(module.in_features)
        self.register_buffer("weight_scale", weight_scale)
        module.weight *= weight_scale.reshape(1, -1)
        self.add_module("linear", module)
        logger.info(f"MulLinear: {module} has been wrapped as `MulLinear`.")

    def forward(self, X):
        updated_x = _transform_input(X, self.weight_scale)
        y = self.linear(updated_x)
        return y

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, weight):
        self.linear.weight = weight

    @property
    def bias(self):
        return self.linear.bias

    @bias.setter
    def bias(self, bias):
        self.linear.bias = bias




def replace_linear_with_smoothed_linear(module, weight_scale):
    from .scale import MulLinear
    logger.info(f"Replace {module} with `MulLinear`.")
    logger.info(f"weight_scale shape: {weight_scale.shape}, weight scale min: {weight_scale.min()}, weight scale max: {weight_scale.max()}")
    return MulLinear(module, weight_scale)

class ScaleCalculator(torch.nn.Module):
    def __init__(self, shape: int, device):
        super().__init__()
        self.shape = shape
        self.device = device
        tensor1 = torch.ones(shape, device=device) * 0.5
        tensor2 = torch.ones(shape, device=device) * 0.5
        self.scale1 = torch.nn.Parameter(tensor1, requires_grad=True)
        self.scale2 = torch.nn.Parameter(tensor2, requires_grad=True)

    def forward(self, x):
        update_scale = torch.clip(self.scale1, min=0.0, max=1.0) / torch.clip(self.scale2, min=1e-5, max=1.0)
        # TODO: add more complex logic here
        return update_scale

    def get_final_scale(self):
        update_scale = torch.clip(self.scale1, min=0.0, max=1.0) / torch.clip(self.scale2, min=1e-5, max=1.0)
        # TODO: add more complex logic here
        return update_scale

# ScaleCalculatorVanilla
class ScaleCalculatorV(torch.nn.Module):
    def __init__(self, shape: int, device):
        super().__init__()
        self.shape = shape
        self.device = device
        tensor1 = torch.ones(shape, device=device)
        # tensor2 = torch.ones(shape, device=device)
        # torch.nn.init.normal_(tensor1)
        # torch.nn.init.normal_(tensor2)
        self.scale1 = torch.nn.Parameter(tensor1, requires_grad=True)
        # self.scale2 = torch.nn.Parameter(tensor2, requires_grad=True)

    def forward(self, x):
        update_scale = self.scale1
        # TODO: add more complicated logic here
        return update_scale

    def get_final_scale(self):
        # TODO: add more complicated logic here
        return self.scale1
