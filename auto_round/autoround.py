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


import copy
import time

import torch
from torch import autocast

from .calib_dataset import get_dataloader
from .special_model_handler import check_hidden_state_dim, check_share_attention_mask
from .utils import (
    CpuInfo,
    block_forward,
    check_is_cpu,
    check_memory_availability,
    check_to_quantized,
    collect_minmax_scale,
    collect_round_v,
    detect_device,
    get_block_names,
    get_module,
    get_scale_shape,
    htcore,
    is_hpu_available,
    is_local_path,
    logger,
    move_input_to_device,
    quant_weight,
    sampling_inputs,
    set_module,
)

if is_hpu_available:
    import habana_frameworks.torch.core as htcore  # pylint: disable=E0401
    import habana_frameworks.torch.hpu as hthpu  # pylint: disable=E0401


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
        - sym (bool): Whether the symmetric quantization is to be used.
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
        self.sym = self.orig_layer.sym
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
            self.sym,
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
        from torch.functional import F

        weight = self.orig_layer.weight
        self.min_scale.data.copy_(torch.clamp(self.min_scale.data, -1, 0))
        self.max_scale.data.copy_(torch.clamp(self.max_scale.data, -1, 0))
        weight_q, _, _ = quant_weight(
            weight,
            self.num_bits,
            self.group_size,
            self.sym,
            self.value,
            self.min_scale,
            self.max_scale,
            self.scale_dtype,
        )
        weight_q = weight_q.to(weight.dtype)
        # pylint: disable=not-callable
        return F.linear(x, weight_q, self.orig_layer.bias)


class WrapperTransformerConv1d(torch.nn.Module):
    def __init__(self, orig_layer, enable_minmax_tuning=True):
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

    def unwrapper(self, v=0, min_scale=0, max_scale=0):
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
            self.weight_t, self.num_bits, self.group_size, self.sym, v, min_scale, max_scale, self.scale_dtype
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
            self.sym,
            self.value,
            self.min_scale,
            self.max_scale,
            self.scale_dtype,
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
        list: A list of names of the wrapped layers and unwrapped layers.
    """
    quantized_layers = []
    unquantized_layers = []
    for n, m in block.named_modules():
        if isinstance(m, torch.nn.Linear):
            if not check_to_quantized(m):
                unquantized_layers.append(n)
                continue
            new_m = WrapperLinear(m, enable_minmax_tuning=enable_minmax_tuning)
            set_module(block, n, new_m)
            quantized_layers.append(n)

        try:
            import transformers

            if isinstance(m, transformers.modeling_utils.Conv1D):
                if not check_to_quantized(m):
                    unquantized_layers.append(n)
                    continue
                new_m = WrapperTransformerConv1d(m, enable_minmax_tuning=enable_minmax_tuning)
                set_module(block, n, new_m)
                quantized_layers.append(n)
        except:
            pass
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
            min_scale = torch.clamp(min_scale, -1, 0)
            max_scale = torch.clamp(max_scale, -1, 0)

        else:
            min_scale = torch.tensor(0)
            max_scale = torch.tensor(0)
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
            min_scale = torch.tensor(0)
            max_scale = torch.tensor(0)
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
        tokenizer: An optional tokenizer for processing input data. If none is provided, a dataloader must be supplied.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (bool): Whether symmetric quantization is to be used (default is False).
        weight_config (dict): Configuration for weight quantization (default is an empty dictionary).
        weight_config={
                   'layer1':##layer_name
                   {
                       'data_type': 'int',
                       'bits': 4,
                       'group_size': 32,
                       'sym': False
                   }
                   ...
               }
        enable_full_range (bool): Whether to enable full range quantization (default is False).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for tuning (default is "auto").
        lr_scheduler: The learning rate scheduler to be used.
        dataloader: The dataloader for input data (to be supported in future).
        dataset_name (str): The default dataset name (default is "NeelNanda/pile-10k").
        dataset_split (str): The split of the dataset to be used (default is "train").
        use_quant_input (bool): Whether to use the output of the previous quantized block as the input for the current
                                block (default is True).
        enable_minmax_tuning (bool): Whether to enable weight min-max tuning (default is True).
        lr (float): The learning rate (default is None, will be set to 1.0/iters).
        minmax_lr (float): The learning rate for min-max tuning (default is None, it will be set to lr automatically).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is True).
        iters (int): Number of iterations (default is 200).
        seqlen (int): Data length of the sequence for tuning (default is 2048).
        n_samples (int): Number of samples (default is 512).
        sampler (str): The sampling method (default is "rand").
        seed (int): The random seed (default is 42).
        n_blocks (int): Number of blocks (default is 1).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        not_use_best_mse (bool): Whether to use mean squared error (default is False).
        dynamic_max_gap (int): The dynamic maximum gap (default is -1).
        data_type (str): The data type to be used (default is "int").
        scale_dtype (str): The data type of quantization scale to be used (default is "float32"), different kernels
                           have different choices.

    Returns:
        The quantized model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = False,
        weight_config: dict = {},
        enable_full_range: bool = False,  ##for symmetric, TODO support later
        batch_size: int = 8,
        amp: bool = True,
        device=None,
        lr_scheduler=None,
        dataloader=None,  ## to support later
        dataset: str = "NeelNanda/pile-10k",
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
        scale_dtype: str = "fp32",
        **kwargs,
    ):
        self.quantized = False
        self.model_orig_dtype = model.dtype
        self.model = model.eval().to("cpu")
        self.amp = amp
        self.use_quant_input = use_quant_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.n_samples = n_samples
        self.n_blocks = n_blocks
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.low_gpu_mem_usage = low_gpu_mem_usage
        self.data_type = data_type
        self.supported_types = [torch.nn.Linear]
        try:
            import transformers

            self.supported_types.append(transformers.modeling_utils.Conv1D)
        except:
            pass
        self.weight_config = weight_config
        self.dataset_split = dataset_split
        self.seed = seed
        self.tokenizer = tokenizer
        self.seqlen = seqlen
        self.train_bs = batch_size
        self.n_blocks = n_blocks
        self.device = detect_device(device)

        if scale_dtype == "fp16" or scale_dtype == "float16":
            self.scale_dtype = torch.float16
        elif scale_dtype == "bf16" or scale_dtype == "bfloat16":
            self.scale_dtype = torch.bfloat16
        else:
            self.scale_dtype = torch.float32

        self.amp_dtype = torch.float16
        if self.model.dtype != torch.float32:
            self.amp_dtype = self.model.dtype
        if self.device == "cpu" or "hpu" in self.device:
            self.amp_dtype = torch.bfloat16
        if self.amp:
            if self.device == "cpu" and not CpuInfo().bf16:
                self.amp = False
                self.model = self.model.to(torch.float32)
                logger.warning("amp is set to FALSE as the current" "device does not support the 'bf16' data type.")
            else:
                self.model = self.model.to(self.amp_dtype)
        else:
            self.model = self.model.to(torch.float32)
        logger.info(f"using {self.model.dtype} for quantization tuning")
        self.dataset_name = dataset

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
        self.share_attention_mask_flag = None
        self.hidden_dim_flag = None
        torch.set_printoptions(precision=3, sci_mode=True)

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
        if is_hpu_available:
            htcore.mark_step()
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
        # for hpu
        if is_hpu_available:
            htcore.mark_step()
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
        # assert self.tokenizer != None or self.dataloader != None

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
                weight_config[n]["sym"] = self.sym
            else:
                if "data_type" not in weight_config[n].keys():
                    weight_config[n]["data_type"] = self.data_type
                if "bits" not in weight_config[n].keys():
                    weight_config[n]["bits"] = self.bits
                if "group_size" not in weight_config[n].keys():
                    weight_config[n]["group_size"] = self.group_size
                if "sym" not in weight_config[n].keys():
                    weight_config[n]["sym"] = self.sym
            weight_config[n]["scale_dtype"] = self.scale_dtype

            m.data_type = weight_config[n]["data_type"]
            m.bits = weight_config[n]["bits"]
            m.group_size = weight_config[n]["group_size"]
            m.sym = weight_config[n]["sym"]
            m.scale_dtype = weight_config[n]["scale_dtype"]

    @torch.no_grad()
    def get_block_outputs(self, block, input_ids, input_others, bs, device, cache_device):
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
            tmp_input_ids, tmp_input_others = sampling_inputs(
                input_ids, input_others, indices, self.seqlen, self.share_attention_mask_flag, self.input_dim
            )
            tmp_output = block_forward(block, tmp_input_ids, tmp_input_others, self.amp, self.amp_dtype, device).to(
                cache_device
            )
            output.append(tmp_output)
        output = torch.cat(output, dim=self.input_dim)
        torch.cuda.empty_cache()
        return output

    @torch.no_grad()
    def calib(self, n_samples, bs):
        """Perform calibration for quantization.

        This method calibrates the model for quantization by processing a specified
        number of samples from the calibration dataset. It ensures that the data is
        properly formatted and feeds it to the model. If the number of samples processed
        is less than the specified number, it logs a warning. If no samples are processed,
        it logs an error and exits.
        Args:
            n_samples (int): The number of samples to use for calibration.
            bs (int): The number of samples to use for calibration
        """

        if self.dataloader is None:
            self.dataloader = get_dataloader(
                self.tokenizer,
                self.seqlen,
                self.dataset_name,
                self.dataset_split,
                self.seed,
                bs,
                self.n_samples,
            )
        total_cnt = 0
        for data in self.dataloader:
            if data is None:
                continue
            if isinstance(data, torch.Tensor):
                input_ids = data.to(self.model.device)
                data_new = input_ids

            elif isinstance(data, str):
                if self.tokenizer is None:
                    logger.error("for string input, please provide tokenizer")
                    exit()
                data = self.tokenizer(data, truncation=True, max_length=self.seqlen, return_tensors="pt").data
                data_new = {}
                for key in data.keys():
                    data_new[key] = data[key].to(self.model.device)
                input_ids = data_new["input_ids"]
            else:
                data_new = {}
                for key in data.keys():
                    data_new[key] = data[key].to(self.model.device)
                input_ids = data_new["input_ids"]
            if input_ids.shape[-1] < self.seqlen:
                continue

            try:
                if isinstance(data_new, torch.Tensor):
                    self.model(data_new)
                else:
                    self.model(**data_new)
            except NotImplementedError:
                pass
            except Exception as error:
                logger.error(error)
            total_cnt += input_ids.shape[0]
            if total_cnt >= n_samples:
                break
        if total_cnt == 0:
            logger.error(
                f"no data has been cached, please provide more data with sequence length >={self.seqlen} in the "
                f"dataloader or decease the sequence length"
            )
            exit()
        elif total_cnt < n_samples:
            logger.warning(
                f"Insufficient number of samples collected may affect the quantification. "
                f"Valid samples size:{total_cnt}, Target sample size:{n_samples}"
            )

    @torch.no_grad()
    def cache_inter_data(self, block_names, n_samples, layer_names=[], last_cache_name=None):
        """Save the inputs of block_name for calibration. For layers, we cache both of inputs and output.

        This method temporarily replaces the forward method of the model to capture
        the inputs passing through the specified block. It then calibrates the model
        using a specified number of samples. Finally, it restores the original forward
        method and returns the inputs for the specified block.
        Args:
            block_names (list): The names of the blocks for which inputs are to be saved.
            layer_names (list):The names of the layers for which inputs are to be saved.
            n_samples (int): The number of samples to use for calibration.
            last_cache_name (str, optional): The name of the last layer to be cached,
                                       we could break the forward in this layer to save time

        Returns:
            dict: A dictionary containing the inputs for the specified block.
        """
        self.inputs = {}
        self.to_cached_layers = block_names + layer_names
        tmp_dtype = None
        if (
            len(block_names) > 1 or len(layer_names) > 0
        ) and self.low_gpu_mem_usage:  ## have bug if block name is not the first block
            tmp_dtype = self.model.dtype
            self.model = (
                self.model.to(torch.bfloat16) if self.amp else self.model.to(torch.float32)
            )  ##force to dtype supported on cpu

        self.last_cache_name = last_cache_name
        if last_cache_name is None and len(block_names) + len(layer_names) == 1:
            self.last_cache_name = block_names[0] if len(block_names) == 1 else layer_names[0]
        calib_bs = self.train_bs
        if not self.low_gpu_mem_usage and len(layer_names) > 1:  ## persume has lm-head
            calib_bs = 1

        self.hook_handles = []
        self._replace_forward()
        self.calib(n_samples, calib_bs)
        self._recover_forward()
        res = self.inputs
        del self.last_cache_name
        del self.to_cached_layers
        if tmp_dtype is not None:
            self.model = self.model.to(tmp_dtype)

        return res

    @torch.no_grad()
    def get_block_forward_func(self, name):
        """Gets the forward function.

        Args:
            name (str): The name of the function.
        Returns:
            function: The forward function.
        """

        def forward(m, hidden_states, *positional_args, **kwargs):
            """Rewrite forward function, process and collect input data.

            Args:
                hidden_states (torch.Tensor): The hidden states tensor.
                *positional_args: Variable number of positional arguments.
                **kwargs: Variable number of keyword arguments.

            Returns:
                NotImplementedError: Getting the first layer inputs and then raise the error to save runtime.
            """
            if self.share_attention_mask_flag is None:
                self.input_dim = check_hidden_state_dim(self.model, positional_args)
                self.share_attention_mask_flag = check_share_attention_mask(self.model, hidden_states, **kwargs)
            if name in self.inputs:
                data = torch.cat([self.inputs[name]["input_ids"], hidden_states.to("cpu")], dim=self.input_dim)
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
                            if (not self.share_attention_mask_flag) and self.inputs[name][key] is not None:
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
                            if (not self.share_attention_mask_flag) and self.inputs[name][key] is not None:
                                self.inputs[name][key] = torch.cat([self.inputs[name][key], alibi.to("cpu")], dim=0)
                            else:
                                self.inputs[name][key] = alibi.to("cpu")
                    elif key not in self.inputs[name].keys():
                        self.inputs[name][key] = move_input_to_device(kwargs[key], device=torch.device("cpu"))
            if name == self.last_cache_name:
                raise NotImplementedError
            else:
                return m.orig_forward(hidden_states, *positional_args, **kwargs)

        return forward

    @torch.no_grad()
    def _get_cache_data_hook_for_layer(self, name):
        """A forward hook to save input max of a module
        :param name: the module name
        :return: A hook function."""

        def cache_input_hook(module, inputs, outputs):
            input = inputs
            if isinstance(inputs, tuple) or isinstance(input, list):
                input = inputs[0]
            if name in self.inputs:
                self.inputs[name] = torch.cat([self.inputs[name], input.to("cpu")], dim=0)
            else:
                self.inputs[name] = input.to("cpu")

        return cache_input_hook

    def _recover_forward(self):
        """Recovers the forward function."""
        for n, m in self.model.named_modules():
            if hasattr(m, "orig_forward"):
                m.forward = m.orig_forward
                delattr(m, "orig_forward")
        for hook_handle in self.hook_handles:
            hook_handle.remove()
        self.hook_handles = []

    def _replace_forward(self):
        """Replaces the forward function."""
        from functools import partial

        for n, m in self.model.named_modules():
            if n in self.to_cached_layers and not isinstance(m, tuple(self.supported_types)):  ##block
                m.orig_forward = m.forward
                m.forward = partial(self.get_block_forward_func(n), m)
            elif n in self.to_cached_layers:  ##linear layer or conv1d layer
                hook_func = self._get_cache_data_hook_for_layer(n)
                hook_handle = m.register_forward_hook(hook_func)
                self.hook_handles.append(hook_handle)

    def quant_layer(self, layer_name, inputs, q_inputs=None, device=torch.device("cpu")):
        logger.info(f"quantizing layer {layer_name}")
        with torch.no_grad():
            layer = get_module(self.model, layer_name)
            cache_device = "cpu"
            layer = layer.to(device)
            available_flag, seqlen, train_bs = check_memory_availability(
                device, inputs, layer.weight, self.seqlen, self.train_bs
            )

            if not available_flag:
                # do RTN quantization
                q_dq_weight, scale, zp = quant_weight(
                    layer.weight,
                    num_bits=layer.bits,
                    group_size=layer.group_size,
                    sym=layer.sym,
                    scale_dtype=layer.scale_dtype,
                )
                layer.weight.data.copy_(q_dq_weight)
                layer.scale = scale.to("cpu")
                layer.zp = zp.to("cpu") if zp is not None else None
                logger.warning(f"RTN is adopted to quantize  {layer_name} due to memory constraint")
                return
            if seqlen != self.seqlen or self.train_bs != train_bs:
                logger.warning(
                    f"the seqlen and bs for tuning {layer_name} have been adjusted to {seqlen} and "
                    f"{train_bs} respectively due to memory constraints"
                )
            inputs = inputs.to(layer.weight.dtype)
            inputs = inputs[:, :seqlen, :]
            if q_inputs is not None:
                q_inputs = q_inputs.to(layer.weight.dtype)
                if len(inputs.shape) == 3:
                    q_inputs = q_inputs[:, :seqlen, :]

            output = []
            for i in range(0, self.n_samples, train_bs):
                end_index = min(self.n_samples, i + train_bs)
                tmp_inputs = inputs[i:end_index, ...].to(device)
                tmp_output = layer.forward(tmp_inputs).to(cache_device)
                output.append(tmp_output)
                torch.cuda.empty_cache()  ##too large for lm head, maybe need to decrease n_sample

            output = torch.cat(output, dim=0)
            torch.cuda.empty_cache()

        wrapper_linear = WrapperLinear(layer, self.enable_minmax_tuning).to(device)
        round_params = []
        minmax_params = []
        round_params.append(wrapper_linear.value)
        minmax_params.append(wrapper_linear.min_scale)
        minmax_params.append(wrapper_linear.max_scale)
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

        pick_samples = train_bs

        n_samples = inputs.shape[0]
        if self.sampler != "rand":
            indices = torch.randperm(n_samples)[:pick_samples]
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        mse_loss = torch.nn.MSELoss().to(device)
        scaler = self.get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        best_v, best_min_scale, best_max_scale = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        gradient_accumulate_steps = self.train_bs // train_bs
        for i in range(self.iters):
            if self.sampler == "rand":
                indices = torch.randperm(n_samples)[:pick_samples]
            total_loss = 0
            try:
                OOM_flag = False
                for _ in range(gradient_accumulate_steps):
                    if q_inputs is not None:
                        current_input = q_inputs[indices, ...].to(device)
                    else:
                        current_input = inputs[indices, ...].to(device)

                    current_output = output[indices, ...].to(device)
                    if self.amp:
                        with autocast(device_type=device.split(":")[0], dtype=self.amp_dtype):
                            output_q = wrapper_linear(current_input)
                            loss = mse_loss(output_q, current_output)  # pylint: disable=not-callable
                    else:
                        output_q = WrapperLinear(current_input)
                        loss = mse_loss(  # pylint: disable=not-callable
                            output_q.to(torch.float32), current_output.to(torch.float32)
                        )

                    total_loss += loss.item() / gradient_accumulate_steps
                    if i == 0:
                        init_loss = total_loss

                    self.scale_loss_and_backward(scaler, loss)
                    torch.cuda.empty_cache()

                if total_loss < best_loss:
                    best_loss = total_loss
                    if not self.not_use_best_mse:
                        best_v = copy.deepcopy(wrapper_linear.value.data)
                        best_min_scale = copy.deepcopy(torch.clamp(wrapper_linear.min_scale.data, -1, 0))
                        best_max_scale = copy.deepcopy(torch.clamp(wrapper_linear.max_scale.data, -1, 0))

                        last_best_iter = i
                if self.not_use_best_mse and i == self.iters - 1:
                    best_v = copy.deepcopy(wrapper_linear.value.data)
                    best_min_scale = copy.deepcopy(torch.clamp(wrapper_linear.min_scale.data, -1, 0))
                    best_max_scale = copy.deepcopy(torch.clamp(wrapper_linear.max_scale.data, -1, 0))

                if not self.not_use_best_mse:
                    if self.dynamic_max_gap > 0 and i - last_best_iter >= self.dynamic_max_gap:
                        break
                self.step(scaler, optimizer, lr_schedule)
            except MemoryError:
                current_input = current_input.to("cpu")
                current_output = current_output.to("cpu")
                torch.cuda.empty_cache()
                OOM_flag = True
                break

        if not OOM_flag:
            last_loss = total_loss
            best_iter = self.iters
            if not self.not_use_best_mse:
                last_loss = best_loss
                best_iter = last_best_iter
            dump_info = f"quantized {layer_name},  loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
            logger.info(dump_info)
            with torch.no_grad():
                unwrapper_layer(self.model, wrapper_linear, layer_name, best_v, best_min_scale, best_max_scale)
        else:
            del best_loss, best_v, best_min_scale, best_max_scale
            with torch.no_grad():
                unwrapper_layer(self.model, wrapper_linear, layer_name)
            logger.warning(
                f"Due to memory constraints, the quantized layer {layer_name} is implemented using the RTN method."
            )

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
        cache_device = "cpu"  ## force cache device to "cpu"
        ##change to block dtype:
        tmp_dtype = self.amp_dtype if self.amp else torch.float32
        for (
            key
        ) in (
            input_others.keys()
        ):  ## as in calibration phase, we may use bf16 for calibration due to low_gpu_memory usage
            if isinstance(input_others[key], torch.Tensor) and (
                input_others[key].dtype == torch.float16 or input_others[key].dtype == torch.bfloat16
            ):
                input_others[key] = input_others[key].to(tmp_dtype)

        output = self.get_block_outputs(block, input_ids, input_others, self.train_bs, device, cache_device)

        if q_input is not None:
            input_ids = q_input.to(cache_device)

        quantized_layer_names, unquantized_layer_names = wrapper_block(block, self.enable_minmax_tuning)

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
            n_samples = input_ids.shape[self.input_dim]
        else:
            n_samples = input_ids.shape[0] // self.seqlen
        if self.sampler != "rand":
            indices = torch.randperm(n_samples)[:pick_samples]
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        mse_loss = torch.nn.MSELoss().to(device)
        scaler = self.get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        best_v, best_min_scale, best_max_scale = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        for i in range(self.iters):
            if self.sampler == "rand":
                indices = torch.randperm(n_samples)[:pick_samples]

            total_loss = 0
            for _ in range(self.gradient_accumulate_steps):
                current_input_ids, current_input_others = sampling_inputs(
                    input_ids,
                    input_others,
                    indices,
                    seqlen=self.seqlen,
                    share_attention_mask_flag=self.share_attention_mask_flag,
                    input_dim=self.input_dim,
                )
                if len(input_ids.shape) == 3:
                    if self.input_dim == 0:
                        current_output = output[indices, :, :]
                    elif self.input_dim == 1:
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
                    with autocast(device_type=device.split(":")[0], dtype=self.amp_dtype):
                        loss = mse_loss(output_q, current_output)  # pylint: disable=not-callable
                else:
                    loss = mse_loss(  # pylint: disable=not-callable
                        output_q.to(torch.float32), current_output.to(torch.float32)
                    )

                total_loss += loss.item() / self.gradient_accumulate_steps
                torch.cuda.empty_cache()
                if i == 0:
                    init_loss = total_loss

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

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        dump_info = (
            f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
            f"layers in the block, loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
        )
        logger.info(dump_info)
        if len(unquantized_layer_names) != 0:
            logger.info(f"{unquantized_layer_names} have not been quantized")
        with torch.no_grad():
            unwrapper_block(block, best_v, best_min_scale, best_max_scale)
        if self.use_quant_input:
            q_outputs = self.get_block_outputs(block, input_ids, input_others, self.train_bs, device, cache_device)

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
                names = block_names[i : i + n_blocks]
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
            m = m.to("cpu")
            torch.cuda.empty_cache()

        del q_input
        del input_ids
        del input_others
        del inputs

        torch.cuda.empty_cache()

    def save_quantized(self, output_dir=None, format="auto_gptq", inplace=True, **kwargs):
        if not self.quantized:
            logger.warning("please run autoround.quantize first")
            return
        from auto_round.export import EXPORT_FORMAT

        if format not in EXPORT_FORMAT:
            logger.error(f"export format only supports {EXPORT_FORMAT.keys()}")
            exit()
        save_quantized_as_format = EXPORT_FORMAT.get(format)
        compressed_model = save_quantized_as_format(
            output_dir,
            model=self.model,
            weight_config=self.weight_config,
            inplace=inplace,
            bits=self.bits,
            group_size=self.group_size,
            sym=self.sym,
            iters=self.iters,
            lr=self.lr,
            minmax_lr=self.minmax_lr,
            enable_minmax_tuning=self.enable_minmax_tuning,
            use_quant_input=self.use_quant_input,
            scale_dtype=self.scale_dtype,
            tokenizer=self.tokenizer,
            supported_types=self.supported_types,
            **kwargs,
        )
        return compressed_model

    @torch.no_grad()
    def gets_layer_names_outside_blocks(self):
        all_layer_names = set()
        for n, m in self.model.named_modules():
            if isinstance(m, tuple(self.supported_types)):
                m.tmp_name = n
                all_layer_names.add(n)
        block_names = get_block_names(self.model)
        all_layer_names_in_block = set()
        for block_name in block_names:
            block = get_module(self.model, block_name)
            for n, m in block.named_modules():
                if isinstance(m, tuple(self.supported_types)):
                    all_layer_names_in_block.add(m.tmp_name)

        res = all_layer_names - all_layer_names_in_block
        for n, m in self.model.named_modules():
            if isinstance(m, tuple(self.supported_types)):
                delattr(m, "tmp_name")
        return list(res)

    def quantize(self):
        """Quantize the model and return the quantized model along with weight configurations.

        Returns:
        The quantized model and weight configurations.
        """
        # logger.info("cache block input")
        block_names = get_block_names(self.model)
        if len(block_names) == 0:
            logger.warning("could not find blocks, exit with original model")
            return
        if self.amp:
            self.model = self.model.to(self.amp_dtype)
        if not self.low_gpu_mem_usage:
            self.model = self.model.to(self.device)

        layer_names = self.gets_layer_names_outside_blocks()
        self.start_time = time.time()
        all_inputs = self.cache_inter_data([block_names[0]], self.n_samples, layer_names=layer_names)
        inputs = all_inputs[block_names[0]]
        self.inputs.pop(block_names[0])
        if "input_ids" in inputs.keys():
            dim = int((hasattr(self.model, "config") and "chatglm" in self.model.config.model_type))
            total_samples = inputs["input_ids"].shape[dim]
            self.n_samples = total_samples
            if total_samples < self.train_bs:
                self.train_bs = total_samples
                logger.warning(f"force the train batch size to {total_samples} ")
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()
        # self.qdq_weight_round(
        #     self.model,
        #     inputs,
        #     block_names,
        #     n_blocks=self.n_blocks,
        #     device=self.device,
        # )

        ##TODO currently we take all the layers outside blocks as post block layers which is not optimal
        if len(layer_names) > 0:
            torch.cuda.empty_cache()
            layer_inputs = all_inputs
            del self.inputs
            q_layer_inputs = None
            if self.use_quant_input:
                if not self.low_gpu_mem_usage:
                    self.model = self.model.to(self.device)
                q_layer_inputs = self.cache_inter_data([], self.n_samples, layer_names=layer_names)
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()
            for layer_name in layer_names:
                q_layer_input = q_layer_inputs[layer_name] if self.use_quant_input else None
                self.quant_layer(layer_name, layer_inputs[layer_name], q_layer_input, device=self.device)
                torch.cuda.empty_cache()

        for n, m in self.model.named_modules():
            if n in self.weight_config.keys():
                if hasattr(m, "scale"):
                    self.weight_config[n]["scale"] = m.scale
                    self.weight_config[n]["zp"] = m.zp
                    if self.group_size <= 0:
                        self.weight_config[n]["g_idx"] = torch.tensor(
                            [0 for i in range(m.weight.shape[1])], dtype=torch.int32, device="cpu"
                        )
                    else:
                        self.weight_config[n]["g_idx"] = torch.tensor(
                            [i // self.group_size for i in range(m.weight.shape[1])], dtype=torch.int32, device="cpu"
                        )
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
        cost_time = end_time - self.start_time
        logger.info(f"quantization tuning time {cost_time}")
        ## dump a summary
        quantized_layers = []
        unquantized_layers = []
        for n, m in self.model.named_modules():
            if isinstance(m, tuple(self.supported_types)):
                if self.weight_config[n]["bits"] == 16:
                    unquantized_layers.append(n)
                else:
                    quantized_layers.append(n)
        summary_info = (
            f"Summary: quantized {len(quantized_layers)}/{len(quantized_layers) + len(unquantized_layers)} in the model"
        )
        if len(unquantized_layers) > 0:
            summary_info += f",  {unquantized_layers} have not been quantized"

        logger.info(summary_info)

        self.quantized = True
        self.model = self.model.to(self.model_orig_dtype)
        return self.model, self.weight_config


class AutoOPTRound(AutoRound):
    """Class for automatic rounding-based quantization with optimizers like adamw of a PyTorch model.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (bool): Whether sym to be used (default is False).
        weight_config (dict): Configuration for weight quantization (default is an empty dictionary).
        enable_full_range (bool): Whether to enable full range quantization (default is False).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for training (default is "auto").
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
        scale_dtype (str): The data type of quantization scale to be used (default is "float32"), different kernels
                           have different choices.
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
        sym: bool = False,
        weight_config: dict = {},
        enable_full_range: bool = False,
        batch_size: int = 8,
        amp: bool = True,
        device="auto",
        lr_scheduler=None,
        dataloader=None,
        dataset: str = "NeelNanda/pile-10k",
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
        scale_dtype: str = "fp32",
        optimizer="AdamW",
        **kwargs,
    ):
        super(AutoOPTRound, self).__init__(
            model,
            tokenizer,
            bits,
            group_size,
            sym,
            weight_config,
            enable_full_range,
            batch_size,
            amp,
            device,
            lr_scheduler,
            dataloader,
            dataset,
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
            scale_dtype,
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
            loss = scaler.scale(loss)

        loss.backward()
        if is_hpu_available:
            htcore.mark_step()
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
        if is_hpu_available:
            htcore.mark_step()


class AutoAdamRound(AutoOPTRound):
    """Class for automatic rounding-based quantization with optimizers like adamw of a PyTorch model.
    The default lr has been changed.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (str): Whether symmetric quantization to be used (default is False).
        weight_config (dict): Configuration for weight quantization (default is an empty dictionary).
        enable_full_range (bool): Whether to enable full range quantization (default is False).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for training (default is "auto").
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
        scale_dtype (str): The data type of quantization scale to be used (default is "float32"), different kernels
                           have different choices.

    Returns:
        The quantized model.
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = False,
        weight_config: dict = {},
        enable_full_range: bool = False,
        batch_size: int = 8,
        amp: bool = True,
        device="auto",
        lr_scheduler=None,
        dataloader=None,
        dataset: str = "NeelNanda/pile-10k",
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
        scale_dtype: str = "fp32",
        optimizer="AdamW",
        **kwargs,
    ):
        super(AutoAdamRound, self).__init__(
            model,
            tokenizer,
            bits,
            group_size,
            sym,
            weight_config,
            enable_full_range,
            batch_size,
            amp,
            device,
            lr_scheduler,
            dataloader,
            dataset,
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
            scale_dtype,
            optimizer,
            **kwargs,
        )
