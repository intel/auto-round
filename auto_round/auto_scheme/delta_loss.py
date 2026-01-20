# Copyright (c) 2025 Intel Corporation
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
from dataclasses import asdict
from functools import wraps
from typing import Iterable, Union

import torch
from accelerate import dispatch_model
from tqdm import tqdm

from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.auto_scheme.register import register_scheme_methods
from auto_round.auto_scheme.utils import (
    apply_quant_scheme,
    compute_avg_bits_for_scheme,
    compute_layer_bits,
    dispatch_model_by_all_available_devices,
    parse_shared_layers,
    remove_quant_scheme,
)
from auto_round.calib_dataset import get_dataloader
from auto_round.data_type.gguf import (
    quant_tensor_gguf_asym_dq,
    quant_tensor_gguf_sym_dq,
    search_gguf_scale_min_asym,
    search_gguf_scale_min_sym,
)
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme, preset_name_to_scheme
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    clear_memory,
    get_block_names,
    get_major_device,
    get_module,
    llm_load_model,
    parse_available_devices,
    set_avg_auto_device_map,
    set_module,
    set_non_auto_device_map,
    to_device,
)
from auto_round.wrapper import WrapperLinear

__all__ = ["gen_layer_config"]


class AutoSchemeWrapperLinear(WrapperLinear):
    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_round_tuning=True,
        need_weight_grad=False,
        enable_torch_compile=False,
        **kwargs,
    ):
        super().__init__(
            orig_layer,
            enable_minmax_tuning,
            enable_norm_bias_tuning,
            device,
            enable_round_tuning,
            enable_torch_compile=enable_torch_compile,
            **kwargs,
        )
        self.total_act_score = 0.0
        self.act_score = 0.0
        self.avg_act_score = 0.0
        self.act_cnt = 0.0
        self.weight_score = 0.0
        self.mix_score = 0.0
        self.super_qdq_func = super()._qdq_weight
        self.act_qdq_func = super()._qdq_act
        # self.device = device
        self.max_act_value = 0
        self.need_weight_grad = need_weight_grad
        self.grad_mode = False
        if self.need_weight_grad:
            self.orig_layer.weight.requires_grad = True

    def _qdq_act(self, x, act_max_scale, act_max=None):
        if hasattr(self.orig_layer, "act_bits") and self.orig_layer.act_bits > 8:
            return x, 1.0, None

        qdq_x, scale, zp = self.act_qdq_func(x, act_max_scale, act_max)
        if self.grad_mode:
            with torch.no_grad():
                self.max_act_value = torch.abs(x).max()
                if torch.abs(x).max() != 0:
                    self.act_cnt += 1
                x_diff = x - qdq_x
                self.x_diff = x_diff.to("cpu")

            def save_grad(grad):
                if self.max_act_value == 0:
                    if torch.abs(grad).max() != 0:
                        raise ValueError
                """
                this ut will cause NAN issue sometimes, need to investigate
                    @multi_card
                    def test_multi_card(self):
                     model_name = "/models/Qwen3-8B"
                """
                if torch.isnan(grad).any() or torch.isnan(self.x_diff).any():
                    self.act_cnt -= 1
                    return None

                self.total_act_score += torch.abs((grad * self.x_diff.to(grad.device))).sum().item()
                self.act_score = 0.0 if self.act_cnt <= 0 else self.total_act_score / self.act_cnt
                self.mix_score = self.weight_score + self.act_score
                self.x_diff = None
                return None

            qdq_x.register_hook(save_grad)
        return qdq_x, scale, zp

    def _qdq_weight(self, value, min_scale, max_scale):
        device = self.device
        if self.orig_layer.bits > 8 or not self.need_weight_grad:
            qdq_w, scale, zp = super()._qdq_weight(
                torch.tensor(0, device=device), torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)
            )

            return qdq_w, 1.0, None

        qdq_w, scale, zp = super()._qdq_weight(
            torch.tensor(0, device=device), torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)
        )

        if self.grad_mode:

            def save_grad(grad):
                qdq_w, scale, zp = self.super_qdq_func(
                    torch.tensor(0, device=device), torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)
                )
                w_diff = self.orig_layer.weight - qdq_w.to(self.orig_layer.weight.device)
                self.weight_score += torch.abs((grad.to(w_diff.device) * w_diff)).sum().item()
                act_score = 0.0 if self.act_cnt <= 0 else self.total_act_score / self.act_cnt
                self.mix_score = self.weight_score + act_score
                return None

            qdq_w.register_hook(save_grad)
        return qdq_w, 1.0, None


class AutoSchemeWrapperLinearForGGUFK(AutoSchemeWrapperLinear):
    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_round_tuning=True,
        need_weight_grad=False,
        **kwargs,
    ):
        super().__init__(
            orig_layer,
            enable_minmax_tuning,
            enable_norm_bias_tuning,
            device,
            enable_round_tuning,
            need_weight_grad,
            **kwargs,
        )
        with torch.no_grad():
            qdq_w, scale, zp = self.super_qdq_func(
                torch.tensor(0).to(device), torch.tensor(1.0).to(device), torch.tensor(1.0).to(device)
            )
        self.register_buffer("qdq_w", qdq_w.detach().clone().to(self.orig_layer.weight.device))

        if self.need_weight_grad:

            def save_grad(grad):
                w_diff = self.orig_layer.weight - self.qdq_w.to(self.orig_layer.weight.device)
                # TODO strange, grad could be in CPU
                self.weight_score += torch.abs((grad.to(w_diff.device) * w_diff)).sum().item()  # TODO add 2nd order
                act_score = 0.0 if self.act_cnt <= 0 else self.total_act_score / self.act_cnt
                self.mix_score = self.weight_score + act_score
                return None

            self.qdq_w.requires_grad_(True)
            self.orig_layer.weight.requires_grad_(False)

            self.qdq_w.register_hook(save_grad)

    def _qdq_weight(self, value, min_scale, max_scale):
        return self.qdq_w, 1.0, None


class AutoSchemeWrapperLinearForGGUFKImatrix(AutoSchemeWrapperLinear):
    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_round_tuning=True,
        need_weight_grad=False,
        enable_torch_compile=False,
        **kwargs,
    ):
        super().__init__(
            orig_layer,
            enable_minmax_tuning,
            enable_norm_bias_tuning,
            device,
            enable_round_tuning,
            need_weight_grad,
            enable_torch_compile=enable_torch_compile,
            **kwargs,
        )
        with torch.no_grad():
            qdq_w = self._init_scale().detach()
            self.register_buffer("qdq_w", qdq_w.detach().clone().to(self.orig_layer.weight.device))
        if self.need_weight_grad:

            def save_grad(grad):
                w_diff = self.orig_layer.weight - self.qdq_w.to(self.orig_layer.weight.device)
                self.weight_score += torch.abs((grad * w_diff.to(grad.device))).sum().item()
                act_score = 0.0 if self.act_cnt <= 0 else self.total_act_score / self.act_cnt
                self.mix_score = self.weight_score + act_score
                return None

            self.qdq_w.requires_grad_(True)
            self.orig_layer.weight.requires_grad_(False)

            self.qdq_w.register_hook(save_grad)

    @torch.no_grad()
    def _init_scale(self):
        tensor = self.orig_layer.weight.data
        bits = self.orig_layer.bits
        scale_dtype = self.orig_layer.scale_dtype
        imatrix = self.orig_layer.imatrix
        orig_dtype = tensor.dtype
        if self.orig_layer.bits in [2, 4, 5]:
            group_size = 16 if bits == 2 else 32
            tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
            scale, wmin, d_scale, d_wmin = search_gguf_scale_min_asym(tensor, bits, scale_dtype, imatrix)
            tensor = revert_tensor_by_pad(tensor, orig_shape=orig_shape, pad_len=pad_len)

            qdq_w, _, _ = quant_tensor_gguf_asym_dq(
                tensor=tensor,
                bits=bits,
                scale_dtype=scale_dtype,
                imatrix=imatrix,
                scale=scale,
                wmin=wmin,
                d_scale=d_scale,
                d_wmin=d_wmin,
            )
        elif bits in [3, 6]:
            group_size = 16
            tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
            scale, d_scale = search_gguf_scale_min_sym(tensor, bits, imatrix, scale_dtype)
            tensor = revert_tensor_by_pad(tensor, orig_shape=orig_shape, pad_len=pad_len)
            qdq_w, _, _ = quant_tensor_gguf_sym_dq(
                tensor=tensor, bits=bits, scale_dtype=scale_dtype, imatrix=imatrix, scale=scale, d_scale=d_scale
            )
        else:
            raise ValueError("bits must be in [2,3,4,5,6]")
        return qdq_w.to(orig_dtype)

    def _qdq_weight(self, value, min_scale, max_scale):
        return self.qdq_w, 1.0, None


@torch.no_grad()
def cal_imatrix(model, dataloader):
    def register_act_hook(model):
        """Registers hooks to accumulate activation squared norms into `imatrix`."""

        def get_imatrix_hook(module, input, output):
            input = input[0] if isinstance(input, (tuple, list)) else input
            flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
            squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)

            if not hasattr(module, "imatrix"):
                module.imatrix = squared.to("cpu")
            else:
                module.imatrix += squared.to(module.imatrix.device).to("cpu")

        hook_handles = []
        for name, module in model.named_modules():
            if isinstance(module, SUPPORTED_LAYER_TYPES):
                hook = module.register_forward_hook(get_imatrix_hook)
                hook_handles.append(hook)
        return hook_handles

    hooks = register_act_hook(model)
    for data in dataloader:
        model.forward(**to_device(data, model.device))
    for hook in hooks:
        hook.remove()


class MyCustomError(Exception):
    """用于中断反向传播的自定义异常"""

    def __init__(self, message):
        super().__init__(message)


last_grad_input = None


def prepare_model_low_gpu(model, block_inputs: dict = None, pbar=None, major_device="cpu"):
    block_inputs.clear()
    for n, m in model.named_modules():
        if hasattr(m, "grad_mode"):
            m.grad_mode = False

    block_names = get_block_names(model)[0]

    def wrap_forward(module, module_name):
        original_forward = module.forward

        @wraps(original_forward)
        def new_forward(*args, **kwargs):
            move_module_to_tuning_device(module, major_device=major_device)
            # 调用原始forward
            with torch.no_grad():
                result = original_forward(*args, **kwargs)
            # 保存输入信息，并确保张量在GPU上且需要梯度
            input_info = {
                "args": [
                    arg.detach().clone().to("cpu") if isinstance(arg, torch.Tensor) else arg for arg in args  # To CPU
                ],
                "kwargs": {
                    k: v.detach().clone().to("cpu") if isinstance(v, torch.Tensor) else v  # To CPU
                    for k, v in kwargs.items()
                },
            }
            block_inputs[module_name] = input_info

            module.to("cpu")
            # torch.cuda.empty_cache()
            if module.tmp_name == block_names[-1]:
                if isinstance(result, torch.Tensor):
                    result = result.requires_grad_(True)
                elif isinstance(result, tuple):
                    result = tuple(r.requires_grad_(True) if isinstance(r, torch.Tensor) else r for r in result)
            pbar.update(1)

            return result

        return new_forward

    # 为每个模块设置临时名称
    for n, m in model.named_modules():
        m.tmp_name = n

    # 包装每个block的forward方法
    for block_name in block_names:
        module = get_module(model, block_name)
        module.forward = wrap_forward(module, block_name)


def model_forward_low_gpu(model, dataloader, major_device="cuda", pbar=None):
    block_inputs = {}

    block_names = get_block_names(model)[0]
    for name in block_names:
        module = get_module(model, name)
        module.orig_forward = module.forward

    def backward_pre_hook(module, grad_input):
        """反向传播前的钩子函数"""
        global last_grad_input
        last_grad_input = grad_input
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        raise MyCustomError("Interrupt backward pass")
        return grad_input

    for data in dataloader:
        prepare_model_low_gpu(model, block_inputs, major_device=major_device, pbar=pbar)
        # 为最后一个block注册反向钩子
        last_block = get_module(model, block_names[-1])
        last_block_backward_hook = last_block.register_full_backward_pre_hook(backward_pre_hook)

        data = to_device(data, model.device)
        output = model.forward(**data, labels=data["input_ids"], use_cache=False)

        try:
            # 反向传播（会被钩子中断）
            output.loss.backward()
        except MyCustomError:
            pass

        current_grad = last_grad_input

        # 手动计算梯度
        last_block_backward_hook.remove()

        for name in block_names:
            module = get_module(model, name)
            module.forward = module.orig_forward

        for block_name in reversed(block_names):

            # 获取block输入
            block_input_info = block_inputs.get(block_name, {})

            block_input_args = to_device(block_input_info.get("args", []), major_device)
            block_input_kwargs = to_device(block_input_info.get("kwargs", {}), major_device)
            block_input_args[0].requires_grad_(True)
            # 获取block模块并移动到GPU
            block_module = get_module(model, block_name)
            for n, m in block_module.named_modules():
                if hasattr(m, "grad_mode"):
                    m.grad_mode = True
                if hasattr(m, "orig_layer"):
                    m.to(m.orig_layer.tuning_device)
                elif hasattr(m, "tuning_device"):
                    m.to(m.tuning_device)
                elif len(list(m.children())) == 0:
                    m.to(major_device)

            # 设置模型为训练模式以启用梯度计算
            block_module.eval()

            # 重新计算block输出
            block_output = block_module(*block_input_args, **block_input_kwargs)

            # 确保输出需要梯度
            if isinstance(block_output, tuple):
                # 对于元组输出，我们通常关心第一个元素（hidden states）
                main_output = block_output[0]
                main_output = main_output.requires_grad_(True)
            else:
                main_output = block_output.requires_grad_(True)

            # 计算梯度
            torch.autograd.backward(
                tensors=main_output,
                # inputs=block_input_args,
                grad_tensors=current_grad,
                retain_graph=True,  # False有些grad会为0，比如mxfp4
            )

            # 获取输入的梯度
            if block_input_args and isinstance(block_input_args[0], torch.Tensor):
                current_grad = block_input_args[0].grad.detach().clone()
            else:
                print(f"Warning: No suitable input gradient found for {block_name}")
                break

            block_module.to("cpu")
            # clear_memory()
            pbar.update(1)


def get_score_for_scheme(
    model,
    tokenizer,
    quant_layer_names,
    fixed_layer_scheme,
    dataset,
    ignore_scale_zp_bits=False,
    nsamples=16,
    seqlen=256,
    pbar=None,
    shared_layers=None,
    need_weight_grad=False,
    enable_torch_compile=False,
    low_gpu_mem_usage=True,
    major_device="cpu",
    batch_size=1,
    disable_opt_rtn=True,
):
    scores_dict = {}  # Key=name,Val=[quant_total_bits, loss]
    for n, m in model.named_modules():
        if type(m) in SUPPORTED_LAYER_TYPES:
            m.weight.requires_grad = False
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = False

    has_imatrix = False
    for name in quant_layer_names:
        if name in fixed_layer_scheme.keys():
            continue
        m = get_module(model, name)
        if hasattr(m, "imatrix") and m.imatrix is not None:
            has_imatrix = True
            break

    for name in quant_layer_names:
        if name in fixed_layer_scheme.keys():
            continue
        m = get_module(model, name)
        if not check_to_quantized(m):
            layer_bits, _ = compute_layer_bits(m, ignore_scale_zp_bits)
            scores_dict[name] = [layer_bits, 0.0]
            continue
        if m.act_bits > 8 and m.super_bits is not None:
            m.scale_dtype = torch.float32  # TODO set this via API
        elif m.act_bits > 8:
            m.scale_dtype = torch.float16
        else:
            m.scale_dtype = torch.bfloat16

        WrapperLayer = AutoSchemeWrapperLinear
        if hasattr(m, "super_group_size") and m.super_group_size is not None:
            if has_imatrix:
                WrapperLayer = AutoSchemeWrapperLinearForGGUFKImatrix
            else:
                WrapperLayer = AutoSchemeWrapperLinearForGGUFK

        with torch.no_grad():
            if low_gpu_mem_usage:
                device = m.tuning_device if hasattr(m, "tuning_device") else major_device
                if "cuda" in device or "xpu" in device:
                    device = major_device
            else:
                device = m.weight.device
                m.tuning_device = m.weight.device

            new_m = WrapperLayer(
                m,
                device=device,
                enable_minmax_tuning=False,  # TODO this should be change
                enable_norm_bias_tuning=False,
                enable_round_tuning=False,
                need_weight_grad=need_weight_grad,
                enable_torch_compile=enable_torch_compile,
                disable_opt_rtn=disable_opt_rtn,
            )
            set_module(model, name, new_m)
    if low_gpu_mem_usage:
        dataloader = get_dataloader(tokenizer, seqlen, dataset_name=dataset, seed=42, bs=batch_size, nsamples=nsamples)

        model_forward_low_gpu(model, dataloader, major_device=major_device, pbar=pbar)
    else:
        dataloader = get_dataloader(tokenizer, seqlen, dataset_name=dataset, seed=42, bs=batch_size, nsamples=nsamples)
        for data in dataloader:
            data = to_device(data, model.device)
            output = model.forward(**data, labels=data["input_ids"], use_cache=False)
            output.loss.backward()
            for n, m in model.named_parameters():  # This should be kept to reduce VRAM footprint
                m.grad = None
            if pbar is not None:
                pbar.update(1)

        for n, m in model.named_parameters():
            m.grad = None

    scores_dict = {}
    for n, m in model.named_modules():
        if hasattr(m, "mix_score"):
            if m.orig_layer.act_bits <= 8:
                if m.act_cnt == 0:
                    logger.warning_once(
                        "layer{n} max abs activation is 0, please use more data to improve the accuracy"
                    )
            layer_bits, _ = compute_layer_bits(m.orig_layer, ignore_scale_zp_bits=ignore_scale_zp_bits)
            scores_dict[n] = [layer_bits, m.mix_score]
    for n, m in model.named_modules():
        if hasattr(m, "orig_layer"):
            set_module(model, n, m.orig_layer)
    return scores_dict


def choose_bits_per_layer_with_path(layers: dict, P: int):
    """
    layers: 每层的候选选项列表 (bits, params, loss)
    P: 参数量上限
    返回: (最小loss, 最优选择路径 [每层的(bits, params, loss, name_list)]) 或 (None, None)
    """
    # dp: params_total -> (loss, path)
    # path 直接存 [选项...]
    dp: dict[int, tuple[float, list]] = {0: (0.0, [])}

    for layer_name, opts in layers.items():
        new_dp: dict[int, tuple[float, list]] = {}
        for cur_params, (cur_loss, cur_path) in dp.items():
            for opt in opts:
                scheme, bits_cost, loss_cost, layer_names = opt
                np_total = cur_params + bits_cost
                if np_total > P:
                    continue
                new_loss = cur_loss + loss_cost
                new_path = cur_path + [(layer_names, scheme)]
                # 取更小 loss 的路径
                if np_total not in new_dp or new_loss < new_dp[np_total][0]:
                    new_dp[np_total] = (new_loss, new_path)
        if not new_dp:
            return None, None

        # Pareto 剪枝
        items = sorted(new_dp.items(), key=lambda x: x[0])  # (params, (loss, path))
        pruned: dict[int, tuple[float, list]] = {}
        best_loss_so_far = float("inf")
        for params_val, (loss_val, path_val) in items:
            if loss_val < best_loss_so_far:
                pruned[params_val] = (loss_val, path_val)
                best_loss_so_far = loss_val
        dp = pruned

    # 选最优
    best_params = min(dp.keys(), key=lambda k: dp[k][0])
    best_loss, best_path = dp[best_params]
    return best_loss, best_path


def move_module_to_tuning_device(module, major_device="cpu"):
    for n, m in module.named_modules():
        if hasattr(m, "orig_layer"):
            m.to(m.orig_layer.tuning_device)
        elif hasattr(m, "tuning_device"):
            m.to(m.tuning_device)
        elif len(list(m.children())) == 0:
            m.to(major_device)


def _gen_layer_config(
    auto_scheme: AutoScheme,
    model: Union[str, torch.nn.Module],
    quant_layer_names: Iterable[str],
    fixed_layer_scheme: dict[str, dict],
    min_avg_bit_scheme,
    dataset: str = "pile-10k",
    tokenizer=None,
    device_map=None,
    enable_torch_compile=False,
    disable_opt_rtn=True,
    model_name=None,
    major_device="cpu",
    device_list=None,
):
    target_bits = auto_scheme.avg_bits
    model.eval()

    block_name = get_block_names(model)[0]  # TODO need change to support vlm
    for name in block_name:
        module = get_module(model, name)
        module.in_block = True
        for n, m in module.named_modules():
            m.in_block = True

    for n, m in model.named_modules():
        if len(list(m.children())) == 0:
            if not hasattr(m, "in_block"):
                m.in_block = False
            if not m.in_block and auto_scheme.low_gpu_mem_usage:
                m.to(major_device)

    total_scores = {}
    schemes = auto_scheme.options

    def check_bf16_scheme(scheme):
        if isinstance(scheme, str) and scheme.upper() == "BF16":
            return True
        if isinstance(scheme, QuantizationScheme):
            scheme = asdict(scheme)
            if scheme["bits"] >= 16 and scheme["act_bits"] >= 16:
                return True
        return False

    if auto_scheme.nsamples is not None:
        nsamples = auto_scheme.nsamples
    else:
        is_moe_model = False
        if hasattr(model, "config"):
            for key in model.config.to_dict().keys():
                if "moe" in key or "expert" in key:
                    is_moe_model = True
                    break
        if is_moe_model:
            logger.info(
                "The model appears to be an MoE  model. "
                "Using more samples to help generate a better auto-scheme recipe."
            )
            nsamples = 64
        else:
            nsamples = 16

    seqlen = auto_scheme.seqlen if auto_scheme.seqlen is not None else 256

    if auto_scheme.batch_size is not None:
        batch_size = auto_scheme.batch_size
    else:
        if auto_scheme.low_gpu_mem_usage:
            batch_size = 8
        else:
            batch_size = 1

    pbar_cnt = 0
    need_weight_grad = False
    for index, scheme in enumerate(schemes):
        if check_bf16_scheme(scheme):
            continue
        if isinstance(scheme, str):
            scheme = asdict(preset_name_to_scheme(scheme))
        elif isinstance(scheme, QuantizationScheme):
            scheme = asdict(scheme)
        bits = scheme.get("bits", 16)
        act_bits = scheme.get("act_bits", 16)
        if bits <= 8 < act_bits:
            need_weight_grad = True
        if not auto_scheme.low_gpu_mem_usage:
            pbar_cnt += nsamples
        if auto_scheme.low_gpu_mem_usage:
            pbar_cnt += len(block_name) * 2 * ((nsamples + batch_size - 1) // batch_size)  # forward backward
    shared_layers = parse_shared_layers(model, auto_scheme.shared_layers)

    embedding_layers_names = []
    for name in quant_layer_names:
        module = get_module(model, name)
        if isinstance(module, torch.nn.Embedding):
            embedding_layers_names.append(name)
    quant_layer_names = list(set(quant_layer_names) - set(embedding_layers_names))

    options_scores = []
    if auto_scheme.low_gpu_mem_usage and not disable_opt_rtn:
        logger.warning("low_gpu_mem_usage is enabled, disable_opt_rtn will be set to True automatically")
        disable_opt_rtn = True
    if not disable_opt_rtn:
        need_imatrix = False
        for scheme in schemes:
            if isinstance(scheme, str):
                scheme = asdict(preset_name_to_scheme(scheme))
            elif isinstance(scheme, QuantizationScheme):
                scheme = asdict(scheme)

            need_imatrix = scheme["super_bits"] is not None
            if need_imatrix:
                break
        if need_imatrix:  # TODO change to block way in low_gpu_mem_usage
            dataloader = get_dataloader(tokenizer, seqlen=256, dataset_name=dataset, seed=42, bs=8, nsamples=16)
            logger.info("start to compute imatrix for GGUF-K quantization in AutoScheme")
            cal_imatrix(model, dataloader)
            logger.info("finish calculating imatrix")

    pbar = tqdm(total=pbar_cnt, desc="Generating AutoScheme")
    for index, scheme in enumerate(schemes):
        apply_quant_scheme(
            model, quant_layer_names=quant_layer_names, fixed_layer_scheme=fixed_layer_scheme, scheme=scheme
        )
        scores = {}  # name: bits, loss
        if check_bf16_scheme(scheme):
            for n in quant_layer_names:
                if n in fixed_layer_scheme.keys():
                    continue
                m = get_module(model, n)
                bits, _ = compute_layer_bits(m, auto_scheme.ignore_scale_zp_bits)
                scores[n] = [bits, 0.0]
        else:
            scores = get_score_for_scheme(
                model,
                tokenizer,
                quant_layer_names,
                fixed_layer_scheme,
                dataset,
                ignore_scale_zp_bits=auto_scheme.ignore_scale_zp_bits,
                pbar=pbar,
                nsamples=nsamples,
                seqlen=seqlen,
                need_weight_grad=need_weight_grad,
                enable_torch_compile=enable_torch_compile,
                low_gpu_mem_usage=auto_scheme.low_gpu_mem_usage,
                major_device=major_device,
                batch_size=batch_size,
                disable_opt_rtn=auto_scheme.disable_opt_rtn,
            )
        new_scores = {}
        for share_layer in shared_layers:
            param_bits = 0
            tmp_loss = 0
            name_list = []
            for name in share_layer:
                if name in scores.keys():
                    param_bits += scores[name][0]
                    tmp_loss += scores[name][1]
                    name_list.append(name)
                    scores.pop(name)
            new_scores[name_list[0]] = [index, param_bits, tmp_loss, name_list]
        for name, item in scores.items():
            new_scores[name] = [index, item[0], item[1], [name]]
        options_total_loss = 0.0
        for key, item in new_scores.items():
            options_total_loss += item[2]
            if key in total_scores:
                total_scores[key].append(item)
            else:
                total_scores[key] = [item]
        options_scores.append(options_total_loss)
        clear_memory(device_list=device_list)

    total_params = 0
    for n, m in model.named_modules():
        if n in quant_layer_names + embedding_layers_names:
            total_params += m.weight.numel()

    target_params_cnt = int(total_params * target_bits)
    sorted_indices = sorted(range(len(options_scores)), key=lambda i: options_scores[i])
    # Layers that are not fixed in fixed_layer_scheme
    not_fixed_embedding_layers_names = [name for name in embedding_layers_names if name not in fixed_layer_scheme]
    for sorted_index in sorted_indices:
        tmp_scheme = schemes[sorted_index]
        if isinstance(tmp_scheme, str):
            tmp_scheme = asdict(preset_name_to_scheme(tmp_scheme))

        for embedding_layer_name in not_fixed_embedding_layers_names:

            fixed_layer_scheme[embedding_layer_name] = tmp_scheme
            embedding_layer = get_module(model, embedding_layer_name)
            for key, item in tmp_scheme.items():
                setattr(embedding_layer, key, item)
        current_avg_bits, _ = compute_avg_bits_for_scheme(
            model,
            quant_layer_names + embedding_layers_names,
            fixed_layer_scheme,
            min_avg_bit_scheme,
            ignore_scale_zp_bits=auto_scheme.ignore_scale_zp_bits,
            clean_scheme=False,
        )

        if current_avg_bits <= target_bits:  # compute_avg_bit remove setting
            for embedding_layer_name in not_fixed_embedding_layers_names:

                fixed_layer_scheme[embedding_layer_name] = tmp_scheme
                embedding_layer = get_module(model, embedding_layer_name)
                for key, item in tmp_scheme.items():
                    setattr(embedding_layer, key, item)
            break

    # Minus fixed_layer
    for name in fixed_layer_scheme.keys():  # The Scheme should have been applied
        m = get_module(model, name)
        layer_bits, _ = compute_layer_bits(m, auto_scheme.ignore_scale_zp_bits)
        target_params_cnt -= layer_bits
    if target_params_cnt <= 0:
        raise ValueError("Avg bits is too small")

    remove_quant_scheme(model)  # Must place after minus fixed_layer

    best_loss, best_path = choose_bits_per_layer_with_path(total_scores, target_params_cnt)
    # print(best_loss, best_path)  # TODO better log
    layer_config = copy.deepcopy(fixed_layer_scheme)
    options = copy.deepcopy(auto_scheme.options)
    # Replace scheme index with
    for index in range(len(options)):
        if isinstance(options[index], str):
            options[index] = preset_name_to_scheme(options[index])
    for item in best_path:
        layer_names = item[0]
        layer_scheme = options[item[1]]
        for layer_name in layer_names:
            layer_config[layer_name] = asdict(layer_scheme)

    if model_name is not None:
        model = None
        del model
    else:
        model = model.to("cpu")  # TODO this requires large ram
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            import accelerate

            accelerate.hooks.remove_hook_from_submodules(model)
            delattr(model, "hf_device_map")
        for n, m in model.named_modules():
            if hasattr(m, "scale_dtype"):  # TODO refine code
                delattr(m, "scale_dtype")
            if hasattr(m, "imatrix"):
                delattr(m, "imatrix")
            if hasattr(m, "tuning_device"):
                delattr(m, "tuning_device")
        for n, m in model.named_parameters():
            if hasattr(m, "grad"):
                m.grad = None
    global last_grad_input
    last_grad_input = None
    clear_memory(device_list=device_list)
    pbar.close()
    return layer_config


# Supports model with gradient clearing between iterations
@register_scheme_methods(("default", "DeltaLoss"))
def gen_layer_config(
    auto_scheme: AutoScheme,
    model: Union[str, torch.nn.Module],
    quant_layer_names: Iterable[str],
    fixed_layer_scheme: dict[str, dict],
    dataset: str = "pile-10k",
    tokenizer=None,
    device_map=None,
    enable_torch_compile=False,
    disable_opt_rtn=True,
    low_gpu_mem_usage=True,
    min_avg_bit_scheme=None,
    **kwargs,
):
    model_name = None
    if isinstance(model, str):
        model_name = model
        # auto does not consider tuning
        model, tokenizer, _ = llm_load_model(model_name, device_map="cpu")
    # Get major device
    major_device = get_major_device(device_map)
    if not low_gpu_mem_usage:
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            model = dispatch_model(model, device_map=model.hf_device_map)
        else:
            model = dispatch_model_by_all_available_devices(model, device_map)
    else:
        model.to("cpu")
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            import accelerate

            accelerate.hooks.remove_hook_from_submodules(model)
        if (isinstance(device_map, str) and "," in device_map) or device_map == "auto":
            set_avg_auto_device_map(model, device_map)
        else:
            set_non_auto_device_map(model, device_map)

        for n in quant_layer_names:
            m = get_module(model, n)
            if not hasattr(m, "tuning_device"):
                m.tuning_device = major_device

    device_list = parse_available_devices(device_map)

    # gradient checkpointing
    if model.supports_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    for name in quant_layer_names:
        m = get_module(model, name)
        m.tmp_name = name

    try:
        res = _gen_layer_config(
            auto_scheme,
            model,
            quant_layer_names,
            fixed_layer_scheme,
            dataset=dataset,
            tokenizer=tokenizer,
            model_name=model_name,
            enable_torch_compile=enable_torch_compile,
            disable_opt_rtn=disable_opt_rtn,
            device_map=device_map,
            major_device=major_device,
            device_list=device_list,
            min_avg_bit_scheme=min_avg_bit_scheme,
        )
    except torch.OutOfMemoryError:
        logger.warning(
            "Fallback to CPU for automatic scheme generation."
            " Using multiple devices is strongly recommended (e.g., --device_map 0,1,2,3)."
        )
        model.to("cpu")
        for n, m in model.named_modules():
            if hasattr(m, "orig_layer"):
                set_module(model, n, m.orig_layer)
        clear_memory(device_list=device_list)
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            import accelerate

            accelerate.hooks.remove_hook_from_submodules(model)
            delattr(model, "hf_device_map")
        res = _gen_layer_config(
            auto_scheme,
            model,
            quant_layer_names,
            fixed_layer_scheme,
            dataset=dataset,
            tokenizer=tokenizer,
            model_name=model_name,
            enable_torch_compile=enable_torch_compile,
            disable_opt_rtn=disable_opt_rtn,
            device_map=device_map,
            major_device=major_device,
            device_list=device_list,
            min_avg_bit_scheme=min_avg_bit_scheme,
        )

    return res
