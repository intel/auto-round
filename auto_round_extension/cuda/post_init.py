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

# MIT License
#
# Copyright (c) 2023 潘其威(William)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch

EXLLAMA_DEFAULT_MAX_INPUT_LENGTH = 2048


def autoround_post_init(model):
    """
    The max_input_length argument is specific to the exllama backend, that requires to initialize a buffer temp_state.
    """
    device_to_buffers_size = {}

    model_uses_exllama = False
    for name, submodule in model.named_modules():
        if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllama":
            model_uses_exllama = True
            device = submodule.qweight.device
            if device not in device_to_buffers_size:
                device_to_buffers_size[device] = {
                    "max_dq_buffer_size": 1,
                    "max_inner_outer_dim": 1,
                }

            submodule._use_act_order = False

            # Disable this heuristic for detecting act_order, but it could be used instead of the config.
            """
            if submodule.g_idx is None:
                submodule.act_order = False
            elif submodule.g_idx is not None and ((submodule.g_idx == 0).all() or
             torch.equal(submodule.g_idx.cpu(), 
             torch.tensor([i // submodule.group_size for i in range(submodule.g_idx.shape[0])], dtype=torch.int32))):
                submodule.g_idx = None
                submodule.act_order = False
            else:
                submodule.act_order = True
            """

            device_to_buffers_size[device]["max_dq_buffer_size"] = max(
                device_to_buffers_size[device]["max_dq_buffer_size"],
                submodule.qweight.numel() * 8,
            )

    if model_uses_exllama:
        # To be honest this is quite ugly, not proud of this.
        try:
            from exllama_kernels import prepare_buffers, set_tuning_params
        except ImportError as e:
            raise ImportError(
                f"Could not import exllama backend dependencies prepare_buffers, set_tuning_params with the following "
                f"error: {e}"
            )

        device_to_buffers = {}

        max_input_len = 1

        for device, buffers_size in device_to_buffers_size.items():
            # The temp_state buffer is required to reorder X in the act-order case.
            # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
            device_to_buffers[device] = {
                "temp_state": torch.zeros(
                    (max_input_len, buffers_size["max_inner_outer_dim"]),
                    dtype=torch.float16,
                    device=device,
                ),
                "temp_dq": torch.zeros(
                    (1, buffers_size["max_dq_buffer_size"]),
                    dtype=torch.float16,
                    device=device,
                ),
                "max_dq_buffer_size": buffers_size["max_dq_buffer_size"],
                "max_inner_outer_dim": buffers_size["max_inner_outer_dim"],
            }

        # Buffers need to be persistent to avoid any bug.
        model.device_to_buffers = device_to_buffers

        for device, buffers in model.device_to_buffers.items():
            prepare_buffers(device, buffers["temp_state"], buffers["temp_dq"])

        # Using the default from exllama repo here.
        matmul_recons_thd = 8
        matmul_fused_remap = False
        matmul_no_half2 = False
        set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

        # The buffers need to have been initialized first before calling make_q4.
        for name, submodule in model.named_modules():
            if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllama":
                submodule.post_init()

    ## exllamav2
    fixed_bytes = {}
    model_uses_exllamav2 = False

    for _, submodule in model.named_modules():
        if hasattr(submodule, "QUANT_TYPE"):
            if submodule.QUANT_TYPE == "exllamav2":
                model_uses_exllamav2 = True
                device = submodule.qweight.device
                scratch_fixed = submodule.scratch_space_fixed()
                fixed_bytes[device] = max(scratch_fixed, fixed_bytes.get(device, 0))
            elif submodule.QUANT_TYPE == "hpu":
                submodule.post_init()

    if model_uses_exllamav2:
        from auto_round_extension.cuda.qlinear_exllamav2 import ExLlamaV2DeviceTensors

        device_tensors = {}
        for device, scratch_bytes in fixed_bytes.items():
            device_tensors[device] = ExLlamaV2DeviceTensors(device.index, scratch_bytes)

        # have persistent buffers, otherwise we will get OOM
        model.device_tensors = device_tensors

        for _, submodule in model.named_modules():
            if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllamav2":
                device = submodule.qweight.device
                submodule.post_init(temp_dq=model.device_tensors[device])
    torch.cuda.empty_cache()

    return model
