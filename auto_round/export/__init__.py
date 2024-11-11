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

from auto_round.export.register import EXPORT_FORMAT, register_format


@register_format("auto_gptq")
def _save_quantized_as_autogptq(*args, **kwargs):
    from auto_round.export.export_to_autogptq.export import save_quantized_as_autogptq

    return save_quantized_as_autogptq(*args, **kwargs)


@register_format("itrex")
def _save_quantized_as_itrex(*args, **kwargs):
    from auto_round.export.export_to_itrex.export import save_quantized_as_itrex

    return save_quantized_as_itrex(*args, **kwargs)


@register_format("itrex_xpu")
def _save_quantized_as_itrex_xpu(*args, **kwargs):
    from auto_round.export.export_to_itrex.export import save_quantized_as_itrex_xpu

    return save_quantized_as_itrex_xpu(*args, **kwargs)


@register_format("auto_round")
def _save_quantized_as_autoround(*args, **kwargs):
    from auto_round.export.export_to_autoround.export import save_quantized_as_autoround

    return save_quantized_as_autoround(*args, **kwargs)


@register_format("auto_awq")
def _save_quantized_as_autoawq(*args, **kwargs):
    from auto_round.export.export_to_awq.export import save_quantized_as_autoawq

    return save_quantized_as_autoawq(*args, **kwargs)
