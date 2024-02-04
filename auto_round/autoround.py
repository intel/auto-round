# Copyright (c) 2023 Intel Corporation
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


import logging
import torch

logger = logging.getLogger("autoround")
logger.setLevel(logging.INFO)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s L%(lineno)d: %(message)s',
                                 "%Y-%m-%d %H:%M:%S")
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

import copy
import time
from torch.amp import autocast
from functools import partial
from torch.functional import F
from .utils import (quant_weight, set_module, get_module, get_block_names, block_forward, sampling_inputs,
                    get_dataloader, get_scale_shape, move_input_to_device, check_is_cpu, collect_round_v,
                    collect_minmax_scale, get_batch_dim)


class SaveInputs:
    """Cache the inputs of the first block."""

    def __init__(self, model, dataloader, seqlen=256, block_name=None):
        """Initializes the SaveInputs class.

        Args:
            model: The model to be used.
            dataloader: The dataloader for the input data.
            seqlen (int): The sequence length.
            block_name (str): The name of the block.
        """
        self.model = model.eval()
        self.dataloader = dataloader
        self.inputs = {}
        self.block_name = block_name
        self.seqlen = seqlen

    @torch.no_grad()
    def get_forward_func(self, name):
        """Gets the forward function.

        Args:
            name (str): The name of the function.

        Returns:
            function: The forward function.
        """

        def forward(_, hidden_states, *positional_args, **kwargs):
            dim = int((hasattr(self.model, "config") and "chatglm" in self.model.config.model_type))
            if name in self.inputs:
                data = torch.cat([self.inputs[name]["input_ids"], hidden_states.to("cpu")], dim=dim)
                self.inputs[name]["input_ids"] = data
            else:
                self.inputs[name] = {}
                self.inputs[name]["input_ids"] = hidden_states.to("cpu")

            if "positional_inputs" not in self.inputs[name]:
                self.inputs[name]["positional_inputs"] = []
            for idx, item in enumerate(positional_args):
                self.inputs[name]["positional_inputs"] = move_input_to_device(positional_args)

            for key in kwargs.keys():
                if isinstance(kwargs[key], torch.Tensor) or isinstance(kwargs[key], list) or (key == "alibi"):
                    if "attention_mask" in key:
                        if key not in self.inputs[name].keys():
                            self.inputs[name][key] = None
                        if kwargs[key] is not None:
                            if self.inputs[name][key] is not None:
                                self.inputs[name][key] = torch.cat(
                                    [self.inputs[name][key], kwargs[key].to("cpu")], dim=0
                                )
                            else:
                                self.inputs[name][key] = kwargs[key].to("cpu")
                    elif "alibi" in key:
                        if key not in self.inputs[name].keys():
                            self.inputs[name][key] = None
                        if isinstance(kwargs[key], torch.Tensor):
                            alibi = kwargs[key]
                            batch = kwargs["attention_mask"].shape[0]
                            alibi = alibi.reshape(batch, -1, alibi.shape[1], alibi.shape[2])
                            if self.inputs[name][key] is not None:
                                self.inputs[name][key] = torch.cat([self.inputs[name][key], alibi.to("cpu")], dim=0)
                            else:
                                self.inputs[name][key] = alibi.to("cpu")
                    elif key not in self.inputs[name].keys():
                        self.inputs[name][key] = move_input_to_device(kwargs[key], device=torch.device("cpu"))
            raise NotImplementedError

        return forward

    @torch.no_grad()
    def get_inputs(self, n_samples=512):
        """Gets the inputs.

        Args:
            n_samples (int): The number of samples.

        Returns:
            dict: The inputs.
        """
        total_cnt = 0
        self._replace_forward()
        for data in self.dataloader:
            if data is None:
                continue
            if isinstance(data, torch.Tensor):
                input_ids = data.to(self.model.device)
            else:
                input_ids = data["input_ids"].to(self.model.device)
            if input_ids.shape[-1] < self.seqlen:
                continue
            if total_cnt + input_ids.shape[0] > n_samples:
                input_ids = input_ids[: n_samples - total_cnt, ...]
            try:
                self.model(input_ids)
            except NotImplementedError:
                pass
            except Exception as error:
                logger.error(error)
            total_cnt += input_ids.shape[0]
            if total_cnt >= n_samples:
                break
        self._recover_forward()
        if total_cnt == 0:
            logger.error(
                f"no data has been cached, please provide more data with sequence length >={self.seqlen} in the "
                f"dataloader or decease the sequence length"
            )
            exit()
        elif total_cnt < n_samples:
            logger.warning(
                f"Insufficient number of samples collected may affect the quantification. "
                f"Effective samples size:{total_cnt}, Target sample size:{n_samples}"
            )
        res = self.inputs[self.block_name]
        if "input_ids" in res.keys():
            total_samples = res["input_ids"].shape[0]
            if total_samples < n_samples:
                logger.warning("only cache {total_samples}")

        return res

    def _recover_forward(self):
        """Recovers the forward function."""
        for n, m in self.model.named_modules():
            if n == self.block_name:
                m.forward = m.orig_forward
                delattr(m, "orig_forward")
                break

    def _replace_forward(self):
        """Replaces the forward function."""
        for n, m in self.model.named_modules():
            if n == self.block_name:
                m.orig_forward = m.forward
                m.forward = partial(self.get_forward_func(n), m)
                break


class WrapperLinear(torch.nn.Module):
    def __init__(self, orig_layer, enable_minmax_tuning=True):
        """A wrapper module for linear layers that enables quantization and min-max tuning of weights.

        Args:
        - orig_layer (torch.nn.Module): The original linear layer to be wrapped.
        - enable_minmax_tuning (bool): Whether to enable min-max scaling tuning. Default is True.

        Attributes:
        - orig_layer (torch.nn.Module): The original linear layer being wrapped.
        - num_bits (int): The number of bits for quantization.
        - group_size (int): The size of the groups for quantization.
        - scheme (str): The quantization scheme to use.
        - value (torch.nn.Parameter): The learnable parameter for quantization.
        - enable_minmax_tuning (bool): Whether min-max scaling tuning is enabled.
        - min_scale (torch.nn.Parameter or torch.Tensor): The minimum scale for min-max tuning.
        - max_scale (torch.nn.Parameter or torch.Tensor): The maximum scale for min-max tuning.
        """
        super(WrapperLinear, self).__init__()
        self.orig_layer = orig_layer
        self.num_bits = self.orig_layer.bits
        self.group_size = self.orig_layer.group_size
        self.scale_dtype = self.orig_layer.scale_dtype
        self.scheme = self.orig_layer.scheme
        self.value = torch.nn.Parameter(
            torch.zeros(self.orig_layer.weight.shape, device=self.orig_layer.weight.device), requires_grad=True
        )
        self.enable_minmax_tuning = enable_minmax_tuning
        shape = get_scale_shape(self.orig_layer.weight, self.group_size)

        if self.enable_minmax_tuning:
            self.min_scale = torch.nn.Parameter(
                torch.zeros(shape, device=self.orig_layer.weight.device), requires_grad=True
            )
            self.max_scale = torch.nn.Parameter(
                torch.zeros(shape, device=self.orig_layer.weight.device), requires_grad=True
            )
        else:
            self.min_scale = torch.tensor(0, device=self.orig_layer.weight.device)
            self.max_scale = torch.tensor(0, device=self.orig_layer.weight.device)

    def unwrapper(self, v, min_scale, max_scale):
        """Unwrapper the layer to the original layer.

        Args:
        - v (torch.Tensor): The rounding v parameter for quantization.
        - min_scale (torch.nn.Parameter or torch.Tensor): The minimum scale for min-max tuning.
        - max_scale (torch.nn.Parameter or torch.Tensor): The maximum scale for min-max tuning.

        Returns:
        - torch.nn.Module: The original linear layer with updated weights after quantization and dequantization.
        """
        min_scale.clamp_(-1, 0)
        max_scale.clamp_(-1, 0)

        q_dq_weight, scale, zp = quant_weight(
            self.orig_layer.weight,
            self.num_bits,
            self.group_size,
            self.scheme,
            v,
            min_scale,
            max_scale,
            self.scale_dtype,
        )
        self.orig_layer.weight.data.copy_(q_dq_weight)
        self.orig_layer.weight.grad = None  ##clear grad
        self.orig_layer.scale = scale.to("cpu")
        self.orig_layer.zp = zp.to("cpu") if zp is not None else None
        return self.orig_layer

    def forward(self, x):
        """Performs forward pass through the wrapped linear layer with quantized weights.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor after applying the linear transformation with quantized weights.
        """
        weight = self.orig_layer.weight
        self.min_scale.data.copy_(torch.clamp(self.min_scale.data, -1, 0))
        self.max_scale.data.copy_(torch.clamp(self.max_scale.data, -1, 0))
        weight_q, _, _ = quant_weight(
            weight, self.num_bits, self.group_size, self.scheme, self.value, self.min_scale, self.max_scale,
            self.scale_dtype
        )
        weight_q = weight_q.to(weight.dtype)
        return F.linear(x, weight_q, self.orig_layer.bias)


class WrapperTransformerConv1d(torch.nn.Module):
    def __init__(self, orig_layer, enable_minmax_tuning=True):
        """A wrapper module for transformers 1D convolutional layers used in transformers,
        enabling quantization and min-max tuning of weights.

        Args:
        - orig_layer (torch.nn.Module): The original 1D convolutional layer to be wrapped.
        - num_bits (int): The number of bits for quantization.
        - group_size (int): The size of the groups for quantization.
        - scheme (str): The quantization scheme to use.
        - enable_minmax_tuning (bool): Whether to enable min-max scaling tuning. Default is True.

        Attributes:
        - orig_layer (torch.nn.Module): The original 1D convolutional layer being wrapped.
        - num_bits (int): The number of bits for quantization.
        - group_size (int): The size of the groups for quantization.
        - scheme (str): The quantization scheme to use.
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
        self.scheme = self.orig_layer.scheme
        device = self.orig_layer.weight.device
        self.weight_t = self.orig_layer.weight.t()
        self.value = torch.nn.Parameter(torch.zeros(self.weight_t.shape, device=device), requires_grad=True)
        shape = get_scale_shape(self.weight_t, self.group_size)

        if enable_minmax_tuning:
            self.min_scale = torch.nn.Parameter(torch.zeros(shape, device=device), requires_grad=True)
            self.max_scale = torch.nn.Parameter(torch.zeros(shape, device=device), requires_grad=True)
        else:
            self.min_scale = torch.tensor(0, device=device)
            self.max_scale = torch.tensor(0, device=device)

    def unwrapper(self, v, min_scale, max_scale):
        """Unwrapper the layer to the original conv1d layer.

        Args:
        - v (torch.Tensor): The scaling parameter for quantization.
        - min_scale (torch.nn.Parameter or torch.Tensor): The minimum scale for min-max tuning.
        - max_scale (torch.nn.Parameter or torch.Tensor): The maximum scale for min-max tuning.

        Returns:
        - torch.nn.Module: The original 1D convolutional layer with updated weights after inverse quantization.
        """
        min_scale.clamp_(-1, 0)
        max_scale.clamp_(-1, 0)
        weight_q, scale, zp = quant_weight(
            self.weight_t,
            self.num_bits,
            self.group_size,
            self.scheme,
            v,
            min_scale,
            max_scale,
            self.scale_dtype
        )
        self.orig_layer.weight.data.copy_(weight_q.t())
        self.orig_layer.weight.grad = None
        self.orig_layer.scale = scale.to("cpu")
        self.orig_layer.zp = zp.to("cpu")
        return self.orig_layer

    def forward(self, x):
        """Performs forward pass through the wrapped 1D convolutional layer with quantized weights.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor after applying the convolutional transformation with quantized weights.
        """
        with torch.no_grad():
            self.min_scale.clamp_(-1, 0)
            self.max_scale.clamp_(-1, 0)
        weight_q, _, _ = quant_weight(
            self.weight_t,
            self.num_bits,
            self.group_size,
            self.scheme,
            self.value,
            self.min_scale,
            self.max_scale,
            self.scale_dtype
        )
        weight_q = weight_q.to(self.weight_t.dtype)
        size_out = x.size()[:-1] + (self.orig_layer.nf,)
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


def wrapper_block(block, enable_minmax_tuning):
    """Wraps the layers in the given block with a custom Wrapper module.

    Args:
        block: The input block containing linear and conv1d layers to be wrapped.
        enable_minmax_tuning: A boolean indicating whether min-max tuning is enabled.

    Returns:
        list: A list of names of the wrapped layers.
    """
    names = []
    for n, m in block.named_modules():
        if isinstance(m, torch.nn.Linear):
            if "fp" in m.data_type or "float" in m.data_type:
                continue
            new_m = WrapperLinear(m, enable_minmax_tuning=enable_minmax_tuning)
            set_module(block, n, new_m)
            names.append(n)

        try:
            import transformers

            if isinstance(m, transformers.modeling_utils.Conv1D):
                if "fp" in m.data_type or "float" in m.data_type:
                    continue
                new_m = WrapperTransformerConv1d(m, enable_minmax_tuning=enable_minmax_tuning)
                set_module(block, n, new_m)
                names.append(n)
        except:
            pass
    return names


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
            min_scale = 0
            max_scale = 0
            if isinstance(vs, dict):
                v = vs[n]
            if isinstance(min_scales, dict):
                min_scale = min_scales[n]
                min_scale = torch.clamp(min_scale, -1, 0)
            if isinstance(max_scales, dict):
                max_scale = max_scales[n]
                max_scale = torch.clamp(max_scale, -1, 0)
            orig_layer = m.unwrapper(v, min_scale, max_scale)
            set_module(block, n, orig_layer)


class AutoRound(object):
    """This is Signround+ which is an advanced version of Signround. For more information,
     please refer to Cheng, Wenhua, et al. "Optimize weight rounding via signed gradient descent
     for the quantization of llms." arXiv preprint arXiv:2309.05516 (2023).

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: Tokenizer for processing input data. Temporarily set as a mandatory parameter.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        scheme (str): The quantization scheme to be used (default is "asym").
        weight_config (dict): Configuration for weight quantization (default is an empty dictionary).
        weight_config={
                   'layer1':##layer_name
                   {
                       'data_type': 'int',
                       'bits': 4,
                       'group_size': 32,
                       'scheme': "asym", ## or sym
                   }
                   ...
               }
        enable_full_range (bool): Whether to enable full range quantization (default is False).
        bs (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for tuning (default is "cuda:0").
        lr_scheduler: The learning rate scheduler to be used.
        dataloader: The dataloader for input data (to be supported in future).
        dataset_name (str): The default dataset name (default is "NeelNanda/pile-10k").
        dataset_split (str): The split of the dataset to be used (default is "train").
        use_quant_input (bool): Whether to use quantized input data (default is True).
        enable_minmax_tuning (bool): Whether to enable min-max tuning (default is True).
        lr (float): The learning rate (default is 0.005).
        minmax_lr (float): The learning rate for min-max tuning (default is None).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is True).
        iters (int): Number of iterations (default is 200).
        seqlen (int): Length of the sequence.
        n_samples (int): Number of samples (default is 512).
        sampler (str): The sampling method (default is "rand").
        seed (int): The random seed (default is 42).
        n_blocks (int): Number of blocks (default is 1).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        not_use_best_mse (bool): Whether to use mean squared error (default is False).
        dynamic_max_gap (int): The dynamic maximum gap (default is -1).
        data_type (str): The data type to be used (default is "int").
        **kwargs: Additional keyword arguments.

    Returns:
        The quantized model.
    """

    def __init__(
            self,
            model,
            tokenizer,
            bits: int = 4,
            group_size: int = 128,
            scheme: str = "asym",
            weight_config: dict = {},
            enable_full_range: bool = False,  ##for symmetric, TODO support later
            bs: int = 8,
            amp: bool = True,
            device="cuda:0",
            lr_scheduler=None,
            dataloader=None,  ## to support later
            dataset_name: str = "NeelNanda/pile-10k",
            dataset_split: str = "train",
            use_quant_input: bool = True,
            enable_minmax_tuning: bool = True,
            lr: float = None,
            minmax_lr: float = None,
            low_gpu_mem_usage: bool = True,
            iters: int = 200,
            seqlen: int = 2048,
            n_samples: int = 512,
            sampler: str = "rand",
            seed: int = 42,
            n_blocks: int = 1,
            gradient_accumulate_steps: int = 1,
            not_use_best_mse: bool = False,
            dynamic_max_gap: int = -1,
            data_type: str = "int",  ##only support data_type
            scale_dtype="fp16",
            **kwargs,
    ):
        self.model = model
        self.model = self.model.to("cpu")
        self.amp = amp
        self.use_quant_input = use_quant_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.n_samples = n_samples
        self.n_blocks = n_blocks
        self.bits = bits
        self.group_size = group_size
        self.scheme = scheme
        self.low_gpu_mem_usage = low_gpu_mem_usage
        self.data_type = data_type
        self.supported_types = [torch.nn.Linear]
        try:
            import transformers

            self.supported_types.append(transformers.modeling_utils.Conv1D)
        except:
            pass
        self.weight_config = weight_config
        assert tokenizer is not None
        self.dataset_split = dataset_split
        self.seed = seed
        self.tokenizer = tokenizer
        self.seqlen = seqlen
        self.train_bs = bs
        self.n_blocks = n_blocks
        self.device = device

        if scale_dtype == 'fp16':
            self.scale_dtype = torch.float16
        elif scale_dtype == 'bf16':
            self.scale_dtype = torch.bfloat16
        else:
            self.scale_dtype = torch.float32

        self.amp_dtype = torch.float16
        if self.model.dtype != torch.float32:
            self.amp_dtype = self.model.dtype
        if self.device == "cpu":
            self.amp_dtype = torch.bfloat16
        if self.amp:
            self.model = self.model.to(self.amp_dtype)
            logger.info(f"using {self.amp_dtype}")
        elif self.device == "cpu" and self.model.dtype == torch.float16:
            self.model = self.model.to(torch.float32)
            logger.info(f"using {torch.float32} for cpu quantization")
        self.dataset_name = dataset_name

        if dataloader is None:
            self.dataloader = get_dataloader(
                self.tokenizer,
                self.seqlen,
                seed=self.seed,
                bs=self.train_bs,
                split=self.dataset_split,
                data_name=self.dataset_name,
            )
        else:
            self.dataloader = dataloader
        self.iters = iters
        if self.iters <= 0:
            logger.warning("iters must be positive, reset it to 200")
            self.iters = 200
        self.lr = lr
        if self.lr is None:
            self.lr = 1.0 / self.iters
        self.minmax_lr = minmax_lr
        if self.minmax_lr is None:
            self.minmax_lr = self.lr

        self.sampler = sampler
        self.gradient_accumulate_steps = gradient_accumulate_steps
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.enable_full_range = enable_full_range
        assert self.enable_full_range is False, "only support enable_full_range=False currently"
        self.lr_scheduler = lr_scheduler
        self.set_layerwise_config(self.weight_config)
        self.optimizer = self.get_optimizer(None)
        self.check_configs()

    def get_optimizer(self, optimizer):
        """Returns the specified optimizer. In SignRound, we fix the optimizer.

        Args:
        optimizer: The optimizer to be used.

        Returns:
        The specified optimizer.
        """
        from auto_round.sign_sgd import SGD

        return SGD

    def get_scaler(self):
        """Returns scaler, in SignRound, no need to use scaler."""
        return None

    def scale_loss_and_backward(self, scaler, loss):
        """Scales the loss and performs backward pass.

        Args:
        scaler: The scaler to be used.
        loss: The loss to be scaled.

        Returns:
        The scaled loss.
        """
        scale_loss = loss * 1000
        scale_loss.backward()
        return scale_loss

    def step(self, scaler, optimizer, lr_schedule):
        """Performs a step in the optimization process.

        Args:
        scaler: The scaler to be used.
        optimizer: The optimizer for the step.
        lr_schedule: The learning rate schedule.

        Returns:
        None
        """
        optimizer.step()
        optimizer.zero_grad()
        lr_schedule.step()

    def check_configs(self):
        """Checks if the configurations are valid.

        Raises:
        AssertionError: If any of the configurations are invalid.
        """
        assert isinstance(self.model, torch.nn.Module)
        assert self.bits > 0, "bits must be positive"
        assert self.group_size == -1 or self.group_size >= 1, "only supports positive group_size or -1(per channel)"
        assert self.train_bs > 0, "batch size must be positive"
        assert self.iters > 0, "iters must be positive"
        assert self.seqlen > 0, "seqlen must be positive"
        assert self.n_blocks > 0, "n_blocks must be positive"
        assert self.gradient_accumulate_steps > 0, "gradient accumulate step must be positive"

    def set_layerwise_config(self, weight_config):
        """Sets the layer-wise configuration based on the provided weight_config.

        Args:
        weight_config: The weight configuration.

        Returns:
        None
        """
        for n, m in self.model.named_modules():
            is_supported_type = False
            for supported_type in self.supported_types:
                if isinstance(m, supported_type):
                    is_supported_type = True
                    break
            if not is_supported_type:
                continue
            if n not in weight_config.keys():
                weight_config[n] = {}
                weight_config[n]["data_type"] = self.data_type
                weight_config[n]["bits"] = self.bits
                weight_config[n]["group_size"] = self.group_size
                weight_config[n]["scheme"] = self.scheme
            else:
                if "data_type" not in weight_config[n].keys():
                    weight_config[n]["data_type"] = self.data_type
                if "bits" not in weight_config[n].keys():
                    weight_config[n]["bits"] = self.bits
                if "group_size" not in weight_config[n].keys():
                    weight_config[n]["group_size"] = self.group_size
                if "scheme" not in weight_config[n].keys():
                    weight_config[n]["scheme"] = self.scheme
            weight_config[n]["scale_dtype"] = self.scale_dtype

            m.data_type = weight_config[n]["data_type"]
            m.bits = weight_config[n]["bits"]
            m.group_size = weight_config[n]["group_size"]
            m.scheme = weight_config[n]["scheme"]
            m.scale_dtype = weight_config[n]["scale_dtype"]

    def get_batch_dim(self, input_others):
        """Get the batch dimension of the input tensor.

        Args:
        input_others: A dictionary containing input data, including positional_inputs.

        Returns:
        The batch dimension of the input tensor.
        """
        dim = int(len(input_others["positional_inputs"]) > 0)
        return dim

    @torch.no_grad()
    def get_block_outputs(self, block, input_ids, input_others, bs, device, cache_device, batch_dim):
        """Compute the output of a given block of the model for a given input.

        Args:
        block: The block of the model.
        input_ids: The input tensor containing tokenized input ids.
        input_others: A dictionary containing additional input data.
        bs: The batch size for computing the output.
        device: The device for computation.
        cache_device: The device for storing the output.
        batch_dim: The batch dimension of the output tensor.

        Returns:
        The output tensor of the block.
        """

        output = []
        for i in range(0, self.n_samples, bs):
            end_index = min(self.n_samples, i + bs)
            indices = torch.arange(i, end_index).to(torch.long)
            tmp_input_ids, tmp_input_others = sampling_inputs(input_ids, input_others, indices, self.seqlen)
            tmp_output = block_forward(block, tmp_input_ids, tmp_input_others, self.amp, self.amp_dtype, device).to(
                cache_device
            )
            output.append(tmp_output)
        output = torch.cat(output, dim=batch_dim)
        torch.cuda.empty_cache()
        return output

    def quant_block(self, block, input_ids, input_others, q_input=None, device=torch.device("cpu")):
        """Quantize the weights of a given block of the model.

        Args:
        block: The block of the model to be quantized.
        input_ids: The input tensor containing tokenized input ids.
        input_others: A dictionary containing additional input data.
        q_input: The quantized input tensor.
        device: The device for quantization.

        Returns:
        Tuple: (q_outputs, output) if self.use_quant_input is True, else (None, output)
        """
        batch_dim = get_batch_dim(input_others)
        if not self.low_gpu_mem_usage and input_ids.device != device:
            input_ids = move_input_to_device(input_ids, device)
            input_others = move_input_to_device(input_others, device)
        cache_device = device
        if self.low_gpu_mem_usage:
            cache_device = "cpu"
        output = self.get_block_outputs(block, input_ids, input_others, self.train_bs, device, cache_device, batch_dim)

        if q_input is not None:
            input_ids = q_input.to(cache_device)

        names = wrapper_block(block, self.enable_minmax_tuning)
        logger.info(names)

        round_params = []
        minmax_params = []
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                round_params.append(m.value)
                minmax_params.append(m.min_scale)
                minmax_params.append(m.max_scale)

        if self.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": self.minmax_lr}], lr=self.lr, weight_decay=0
            )
        else:
            optimizer = self.optimizer(round_params, lr=self.lr, weight_decay=0)

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters, verbose=False
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)

        pick_samples = self.train_bs
        if len(input_ids.shape) == 3:
            n_samples = input_ids.shape[batch_dim]
        else:
            n_samples = input_ids.shape[0] // self.seqlen
        if self.sampler != "rand":
            indices = torch.randperm(n_samples)[:pick_samples]
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        mse_loss = torch.nn.MSELoss().to(device)
        scaler = self.get_scaler()  # pylint: disable=assignment-from-none
        for i in range(self.iters):
            if self.sampler == "rand":
                indices = torch.randperm(n_samples)[:pick_samples]

            total_loss = 0
            for _ in range(self.gradient_accumulate_steps):
                current_input_ids, current_input_others = sampling_inputs(
                    input_ids, input_others, indices, seqlen=self.seqlen
                )
                if len(input_ids.shape) == 3:
                    if batch_dim == 0:
                        current_output = output[indices, :, :]
                    elif batch_dim == 1:
                        current_output = output[:, indices, :]
                    else:
                        current_output = output[:, :, indices]
                else:
                    current_output = output.view(n_samples, self.seqlen, -1)
                    current_output = current_output[indices, :, :]
                    current_output = current_output.reshape(-1, current_output.shape[-1])
                current_output = move_input_to_device(current_output, device)

                output_q = block_forward(
                    block, current_input_ids, current_input_others, self.amp, self.amp_dtype, device
                )
                if self.amp and not check_is_cpu(device):
                    with autocast(device_type="cuda", dtype=self.amp_dtype):
                        loss = mse_loss(output_q, current_output)
                else:
                    loss = mse_loss(output_q.to(torch.float32), current_output.to(torch.float32))

                total_loss += loss.item() / self.gradient_accumulate_steps
                self.scale_loss_and_backward(scaler, loss)

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    # print(f"get better result at iter {i}, the loss is {total_loss}", flush=True)
                    best_v = collect_round_v(block)
                    best_min_scale, best_max_scale = collect_minmax_scale(block)
                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_v = collect_round_v(block)
                best_min_scale, best_max_scale = collect_minmax_scale(block)

            if not self.not_use_best_mse:
                if self.dynamic_max_gap > 0 and i - last_best_iter >= self.dynamic_max_gap:
                    break
            self.step(scaler, optimizer, lr_schedule)
        unwrapper_block(block, best_v, best_min_scale, best_max_scale)
        if self.use_quant_input:
            q_outputs = self.get_block_outputs(
                block, input_ids, input_others, self.train_bs, device, cache_device, batch_dim
            )

            return q_outputs, output

        else:
            return None, output

    def qdq_weight_round(
            self,
            model: torch.nn.Module,
            inputs,
            block_names,
            n_blocks=1,
            device=torch.device("cpu"),
    ):
        """Quantize and dequantize the weights of the specified blocks in the model.

        Args:
        model: The PyTorch model to be quantized.
        inputs: The input data for quantization.
        block_names: The names of the blocks to be quantized and dequantized.
        n_blocks: The number of blocks to quantize and dequantize.
        device: The device for quantization and dequantization.

        Returns:
        None
        """
        q_input = None
        torch.cuda.empty_cache()
        for n, m in model.named_parameters():
            m.requires_grad_(False)
        input_ids = inputs["input_ids"]
        inputs.pop("input_ids", None)
        input_others = inputs
        torch.cuda.empty_cache()
        for i in range(0, len(block_names), n_blocks):
            if n_blocks == 1:
                n = block_names[i]
                logger.info(f"quantizing {i + 1}/{len(block_names)}, {n}")
                m = get_module(model, n)
            else:
                names = block_names[i: i + n_blocks]
                logger.info(names)
                modules = [get_module(model, n) for n in names]
                m = WrapperMultiblock(modules)

            m = m.to(device)

            q_input, input_ids = self.quant_block(
                m,
                input_ids,
                input_others,
                q_input=q_input,
                device=device,
            )
            m.to("cpu")
            torch.cuda.empty_cache()

        del q_input
        del input_ids
        del input_others
        del inputs

        torch.cuda.empty_cache()

    def export(self, output_dir, target="itrex", **kwargs):
        if target == "itrex":
            self.export_to_itrex(output_dir)
        elif target == "auto_gptq":
            self.export_to_autogptq(output_dir, **kwargs)
        else:
            logger.error("export only supports itrex and auto_gptq now")

    def export_to_autogptq(self, output_dir, use_triton=False):
        """
        Export the model to autogptq format to easily leverage cuda kernel
        """
        if not self.quantized:
            logger.warning("please run autoround.quantize first")
            return
        model = copy.deepcopy(self.model.to("cpu"))  ##TODO avoid this deepcopy

        from auto_gptq.modeling._utils import pack_model
        if self.bits == 3 or use_triton is False:
            if self.bits == 3 and use_triton is True:
                logger.warning("triton does not support 3 bits, reset it to False")
            quantizers = {}
            for key in self.weight_config:
                info = self.weight_config[key]
                if info["bits"] > 8:
                    continue
                quantizers[key] = (None, info['scale'], info['zp'], info['g_idx'])
            pack_model(model, quantizers, self.bits, self.group_size, use_cuda_fp16=True, desc_act=False,
                       force_layer_back_to_cpu=True, use_triton=False)
        else:
            quantizers = {}
            for key in self.weight_config:
                info = self.weight_config[key]
                if info["bits"] > 8:
                    continue
                quantizers[key] = (None, info['scale'].to(torch.float32), info['zp'].to(torch.float32), info['g_idx'])

            pack_model(model, quantizers, self.bits, self.group_size, use_cuda_fp16=True, desc_act=False,
                       force_layer_back_to_cpu=True, use_triton=True)
        from auto_round import save_quantized_to_autogptq
        sym = self.scheme == "sym"
        save_quantized_to_autogptq(model, output_dir, bits=self.bits, group_size=self.group_size, sym=sym,
                                   iters=self.iters, lr=self.lr, minmax_lr=self.minmax_lr,
                                   enable_minmax_tuning=self.enable_minmax_tuning, use_quant_input=self.use_quant_input,
                                   use_safetensors=True)

    def export_to_itrex(self, output_dir):
        """Save configure file and weights for CPU backend inference."""
        from .export_to_itrex import compress_model
        compressed_model, quantize_config = compress_model(self.model, self.weight_config)
        if quantize_config is not None:
            config = compressed_model.config
            setattr(config, "quantization_config", quantize_config.to_dict())
            config.save_pretrained(output_dir)
            quantize_config.save_pretrained(output_dir)

        try:
            compressed_model.save_pretrained(output_dir, safe_serialization=True)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            logger.info("Saved config file and weights of quantized model to {}.".format(output_dir))
        except IOError as e:  # pragma: no cover
            logger.error("Fail to save configure file and weights due to {}.".format(e))

    def quantize(self):
        """Quantize the model and return the quantized model along with weight configurations.

        Returns:
        The quantized model and weight configurations.
        """
        start_time = time.time()
        # logger.info("cache block input")
        block_names = get_block_names(self.model)
        if len(block_names) == 0:
            logger.warning("could not find blocks, exit with original model")
            return

        if self.amp:
            self.model = self.model.to(self.amp_dtype)
        if not self.low_gpu_mem_usage:
            self.model = self.model.to(self.device)

        save_input_actor = SaveInputs(self.model, self.dataloader, self.seqlen, block_names[0])
        inputs = save_input_actor.get_inputs(n_samples=self.n_samples)
        del save_input_actor
        if "input_ids" in inputs.keys():
            total_samples = inputs["input_ids"].shape[0]
            if total_samples < self.train_bs:
                self.train_bs = total_samples
                logger.warning(f"force the train batch size to {total_samples} ")
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()
        self.qdq_weight_round(
            self.model,
            inputs,
            block_names,
            n_blocks=self.n_blocks,
            device=self.device,
        )
        for n, m in self.model.named_modules():
            if n in self.weight_config.keys():
                if hasattr(m, "scale"):
                    self.weight_config[n]["scale"] = m.scale
                    self.weight_config[n]["zp"] = m.zp
                    self.weight_config[n]["g_idx"] = torch.tensor(
                        [i // self.group_size for i in range(m.weight.shape[1])], dtype=torch.int32, device="cpu")
                    delattr(m, "scale")
                    delattr(m, "zp")
                else:
                    self.weight_config[n]["data_type"] = "float"
                    if self.amp_dtype == torch.bfloat16:
                        self.weight_config[n]["data_type"] = "bfloat"
                    self.weight_config[n]["bits"] = 16
                    self.weight_config[n]["group_size"] = None
                    self.weight_config[n]["sym"] = None

        end_time = time.time()
        cost_time = end_time - start_time
        logger.info(f"quantization tuning time {cost_time}")
        self.quantized = True
        ##self.export_to_autogptq("test_export")
        return self.model, self.weight_config


class AutoOPTRound(AutoRound):
    """Class for automatic rounding-based quantization with optimizers like adamw of a PyTorch model.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        scheme (str): The quantization scheme to be used (default is "asym").
        weight_config (dict): Configuration for weight quantization (default is an empty dictionary).
        enable_full_range (bool): Whether to enable full range quantization (default is False).
        bs (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for training (default is "cuda:0").
        lr_scheduler: The learning rate scheduler to be used.
        dataloader: The dataloader for input data (to be supported in future).
        dataset_name (str): The default dataset name (default is "NeelNanda/pile-10k").
        dataset_split (str): The split of the dataset to be used (default is "train").
        use_quant_input (bool): Whether to use quantized input data (default is True).
        enable_minmax_tuning (bool): Whether to enable min-max tuning (default is True).
        lr (float): The learning rate (default is 0.005).
        minmax_lr (float): The learning rate for min-max tuning (default is None).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is True).
        iters (int): Number of iterations (default is 200).
        seqlen (int): Length of the sequence.
        n_samples (int): Number of samples (default is 512).
        sampler (str): The sampling method (default is "rand").
        seed (int): The random seed (default is 42).
        n_blocks (int): Number of blocks (default is 1).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        not_use_best_mse (bool): Whether to use mean squared error (default is False).
        dynamic_max_gap (int): The dynamic maximum gap (default is -1).
        data_type (str): The data type to be used (default is "int").
        optimizer: string or object
        **kwargs: Additional keyword arguments.

    Returns:
        The quantized model.
    """

    def __init__(
            self,
            model,
            tokenizer=None,
            bits: int = 4,
            group_size: int = 128,
            scheme: str = "asym",
            weight_config: dict = {},
            enable_full_range: bool = False,
            bs: int = 8,
            amp: bool = True,
            device="cuda:0",
            lr_scheduler=None,
            dataloader=None,
            dataset_name: str = "NeelNanda/pile-10k",
            dataset_split: str = "train",
            use_quant_input: bool = True,
            enable_minmax_tuning: bool = True,
            lr: float = None,
            minmax_lr: float = None,
            low_gpu_mem_usage: bool = True,
            iters: int = 200,
            seqlen: int = 2048,
            n_samples: int = 512,
            sampler: str = "rand",
            seed: int = 42,
            n_blocks: int = 1,
            gradient_accumulate_steps: int = 1,
            not_use_best_mse: bool = False,
            dynamic_max_gap: int = -1,
            data_type: str = "int",
            optimizer="AdamW",
            **kwargs,
    ):
        super(AutoOPTRound, self).__init__(
            model,
            tokenizer,
            bits,
            group_size,
            scheme,
            weight_config,
            enable_full_range,
            bs,
            amp,
            device,
            lr_scheduler,
            dataloader,
            dataset_name,
            dataset_split,
            use_quant_input,
            enable_minmax_tuning,
            lr,
            minmax_lr,
            low_gpu_mem_usage,
            iters,
            seqlen,
            n_samples,
            sampler,
            seed,
            n_blocks,
            gradient_accumulate_steps,
            not_use_best_mse,
            dynamic_max_gap,
            data_type,
            **kwargs,
        )

        self.optimizer = self.get_optimizer(optimizer)

    def get_optimizer(self, optimizer):
        if optimizer is None:
            optimizer = torch.optim.AdamW
        elif isinstance(optimizer, str):
            optimizer = getattr(torch.optim, optimizer)
        else:
            optimizer = optimizer
        return optimizer

    def get_scaler(self):
        scaler = None
        if self.amp and not check_is_cpu(self.device):
            from torch.cuda.amp import GradScaler

            scaler = GradScaler(init_scale=1024, growth_interval=100000)
        return scaler

    def scale_loss_and_backward(self, scaler, loss):
        if scaler is not None:
            scale_loss = scaler.scale(loss)
            scale_loss.backward()
            return scale_loss
        else:
            loss.backward()
            return loss

    def step(self, scaler, optimizer, lr_schedule):
        if scaler is not None:
            scaler.step(optimizer)
            optimizer.zero_grad()
            lr_schedule.step()
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad()
            lr_schedule.step()


class AutoAdamRound(AutoOPTRound):
    """Class for automatic rounding-based quantization with optimizers like adamw of a PyTorch model.
    The default lr has been changed.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        scheme (str): The quantization scheme to be used (default is "asym").
        weight_config (dict): Configuration for weight quantization (default is an empty dictionary).
        enable_full_range (bool): Whether to enable full range quantization (default is False).
        bs (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for training (default is "cuda:0").
        lr_scheduler: The learning rate scheduler to be used.
        dataloader: The dataloader for input data (to be supported in future).
        dataset_name (str): The default dataset name (default is "NeelNanda/pile-10k").
        dataset_split (str): The split of the dataset to be used (default is "train").
        use_quant_input (bool): Whether to use quantized input data (default is True).
        enable_minmax_tuning (bool): Whether to enable min-max tuning (default is True).
        lr (float): The learning rate (default is 0.005).
        minmax_lr (float): The learning rate for min-max tuning (default is None).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is True).
        iters (int): Number of iterations (default is 200).
        seqlen (int): Length of the sequence.
        n_samples (int): Number of samples (default is 512).
        sampler (str): The sampling method (default is "rand").
        seed (int): The random seed (default is 42).
        n_blocks (int): Number of blocks (default is 1).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        not_use_best_mse (bool): Whether to use mean squared error (default is False).
        dynamic_max_gap (int): The dynamic maximum gap (default is -1).
        data_type (str): The data type to be used (default is "int").
        optimizer: string or object
        **kwargs: Additional keyword arguments.

    Returns:
        The quantized model.
    """

    def __init__(
            self,
            model,
            tokenizer=None,
            bits: int = 4,
            group_size: int = 128,
            scheme: str = "asym",
            weight_config: dict = {},
            enable_full_range: bool = False,
            bs: int = 8,
            amp: bool = True,
            device="cuda:0",
            lr_scheduler=None,
            dataloader=None,
            dataset_name: str = "NeelNanda/pile-10k",
            dataset_split: str = "train",
            use_quant_input: bool = True,
            enable_minmax_tuning: bool = True,
            lr: float = None,
            minmax_lr: float = None,
            low_gpu_mem_usage: bool = True,
            iters: int = 200,
            seqlen: int = 2048,
            n_samples: int = 512,
            sampler: str = "rand",
            seed: int = 42,
            n_blocks: int = 1,
            gradient_accumulate_steps: int = 1,
            not_use_best_mse: bool = False,
            dynamic_max_gap: int = -1,
            data_type: str = "int",
            optimizer="AdamW",
            **kwargs,
    ):
        super(AutoAdamRound, self).__init__(
            model,
            tokenizer,
            bits,
            group_size,
            scheme,
            weight_config,
            enable_full_range,
            bs,
            amp,
            device,
            lr_scheduler,
            dataloader,
            dataset_name,
            dataset_split,
            use_quant_input,
            enable_minmax_tuning,
            lr,
            minmax_lr,
            low_gpu_mem_usage,
            iters,
            seqlen,
            n_samples,
            sampler,
            seed,
            n_blocks,
            gradient_accumulate_steps,
            not_use_best_mse,
            dynamic_max_gap,
            data_type,
            optimizer,
            **kwargs,
        )

