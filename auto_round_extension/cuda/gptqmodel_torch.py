# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from transformers import PreTrainedModel

from gptqmodel.adapter.adapter import Adapter, Lora
from gptqmodel.models._const import DEVICE, PLATFORM
from gptqmodel.nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.logger import setup_logger
from gptqmodel.utils.torch import torch_compile
log = setup_logger()

class QuantLinear(PackableQuantLinear):
    SUPPORTS_BITS = [2, 3, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    # for transformers/optimum tests compat
    QUANT_TYPE = "torch"

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        desc_act: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.TORCH),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs)

        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8

        # if self.group_size != self.in_features:
        #     self.padded_in_features = self.in_features + (-self.in_features % self.group_size)
        # else:
        #     self.padded_in_features = self.in_features

    def post_init(self):
        # if self.padded_in_features != self.in_features:
        #     self.qweight.resize_(self.padded_in_features // self.pack_dtype_bits * self.bits, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.padded_in_features / self.group_size),
        #         self.out_features // self.pack_dtype_bits * self.bits
        #     )
        #     self.scales.resize_((math.ceil(self.padded_in_features / self.group_size), self.out_features), )
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_in_features)], dtype=torch.int32,
        #                               device=self.g_idx.device)

        super().post_init()

        # torch benefits the most from torch.compile, enable it by default
        self.optimize()

    def optimize(self, backend: str = None, mode: str = None, fullgraph: bool = False):
        if self.optimized:
            return

        if backend is None:
            # MPS doesn't support inductor.
            backend = "inductor" if self.list_buffers()[0].device.type != "mps" else "aot_eager"

        # compile dequantize
        self.dequantize_weight = torch_compile(self.dequantize_weight, backend=backend, mode=mode, fullgraph=fullgraph)

        if self.adapter:
            self.adapter.optimize(backend=backend, mode=mode, fullgraph=fullgraph)

        super().optimize()

    def train(self, mode: bool = True):
        old_train = self.training
        if mode == old_train:
            return self

        from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module

        # IPEX kernel will use Torch for training only and switches back to IPEX for eval/inference
        # If the kernel inherits Torch kernel only for training and can do its own inference in v1 (IPEX, Marlin) then
        # we can support training for all these v1 kernels by enabling this flag. We need to switch qzero states
        # by overriding module train() and swapping qzero back between v1 and v2 (Torch kernel requires v2)
        if self.SUPPORTS_TRAINING_USE_TORCH_KERNEL:
            # training starts
            if mode:
                # one time clone v1 qzeros and save both v1 and v2 qzeros in memory
                if self.qzero_format() == 1:
                    if not hasattr(self, "qzeros_data_v1"):
                        self.qzeros_data_v1 = self.qzeros.data.clone()
                        convert_gptq_v1_to_v2_format_module(self, bits=self.bits, pack_dtype=self.pack_dtype)
                        self.qzeros_data_v2 = self.qzeros.data
                    else:
                        self.qzeros.data = self.qzeros_data_v2
                        self.qzero_format(format=2)

            # training switching to inference/eval
            else:
                if hasattr(self, "qzeros_data_v1"):
                    # switch qzero back to v1 for inference/eval
                    self.qzeros.data = self.qzeros_data_v1
                    self.qzero_format(format=1)

        return super().train(mode=mode)

    def forward(self, x: torch.Tensor):
        # if x.size(-1) != self.padded_in_features:
        #     x = F.pad(x, (0, self.padded_in_features - self.in_features))

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        out = self._forward(x, out_shape)
        return out

    def _forward(self, x, out_shape):
        num_itr = self.g_idx.shape[0] // x.shape[-1]
        # make sure dequant dtype matches input x
        weights = self.dequantize_weight(num_itr=num_itr).to(x.dtype)

        out = torch.matmul(x, weights).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    # clear gptq only weights: useful in de-quantization
    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None

def dequantize_model(model: PreTrainedModel):
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchQuantLinear):
            raise ValueError(
                "Only models loaded using TorchQuantLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.TORCH."
            )

        if isinstance(module, TorchQuantLinear):
            # Create a new Linear layer with dequantized weights
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to("cpu", torch.float16))
            new_module.bias = torch.nn.Parameter(module.bias)

            # Replace the module in the model
            parent = model
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    del model.config.quantization_config
    return model

__all__ = ["QuantLinear"]
