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

from functools import lru_cache


@lru_cache(maxsize=None)
def is_distributed():
    import torch.distributed as dist

    return dist.is_initialized() and dist.get_world_size() > 1


def setup_ddp_if_needed_(ar, block, device_list):
    if not is_distributed():
        return
    from torch.nn.parallel import DistributedDataParallel as DDP

    block = DDP(block, device_ids=[device_list], find_unused_parameters=True)
