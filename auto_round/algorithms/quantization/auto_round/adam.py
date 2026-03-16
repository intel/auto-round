# Copyright (c) 2026 Intel Corporation
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
from typing import Union

import torch

from auto_round.algorithms.quantization.auto_round.quantizer import ARQuantizer
from auto_round.compressors.base import BaseCompressor
from auto_round.schemes import QuantizationScheme
from auto_round.utils import check_is_cpu, htcore, is_hpex_available


class ARAdamQuantizer(ARQuantizer):
    def _get_optimizer(self, optimizer):
        if optimizer is None:
            optimizer = torch.optim.AdamW
        elif isinstance(optimizer, str):
            optimizer = getattr(torch.optim, optimizer)
        else:
            optimizer = optimizer
        return optimizer

    def _get_scaler(self):
        scaler = None
        if self.amp and not check_is_cpu(self.device):
            from torch.cuda.amp import GradScaler

            scaler = GradScaler(init_scale=1024, growth_interval=100000)
        return scaler

    def _scale_loss_and_backward(self, scaler, loss):
        if scaler is not None:
            loss = scaler.scale(loss)

        loss.backward()
        if is_hpex_available():
            htcore.mark_step()
        return loss

    def _step(self, scaler, optimizer, lr_schedule):
        if scaler is not None:
            scaler.step(optimizer)
            optimizer.zero_grad()
            lr_schedule.step()
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad()
            lr_schedule.step()
        if is_hpex_available():
            htcore.mark_step()
