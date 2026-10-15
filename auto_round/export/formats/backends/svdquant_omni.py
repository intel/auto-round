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

"""Standard format backend for vLLM-Omni SVDQuant NVFP4 checkpoints."""

from __future__ import annotations

from auto_round.export.formats.backends.svdquant_nunchaku import SVDQuantNunchakuFormat
from auto_round.export.formats.base import OutputFormat


@OutputFormat.register("svdquant_omni")
class SVDQuantOmniFormat(SVDQuantNunchakuFormat):
    """Reuse full-model SVDQuant export handling, but select the Omni tensor ABI."""

    support_schemes = ["NVFP4"]
    format_name = "svdquant_omni"

    @classmethod
    def check_scheme_args(cls, scheme) -> bool:
        from auto_round.export.svdquant_omni import validate_nvfp4_scheme

        return validate_nvfp4_scheme(scheme)

    def check_and_reset_format(self, scheme, ctx):
        self.check_scheme_args(scheme)
        self._validate_svd_layer_overrides(ctx.model, ctx.layer_config)
        # Do not resolve Nunchaku architecture mappings or reset W4A4 to fake.
        return None, scheme, ctx.layer_config, ctx.quant_block_list

    def save_quantized(
        self,
        output_dir,
        model=None,
        tokenizer=None,
        layer_config=None,
        inplace=True,
        device="cpu",
        serialization_dict=None,
        *,
        adapter=None,
        model_adapter=None,
        **kwargs,
    ):
        if adapter is not None and model_adapter is not None:
            raise TypeError("Pass only one of model_adapter and adapter.")
        if output_dir is None:
            return model
        from auto_round.export.svdquant_omni import save_svdquant_omni

        self._validate_svd_layer_overrides(model, layer_config)
        if adapter is None:
            adapter = model_adapter
        # Nunchaku's named adapters and transform-selected mapping describe a
        # different runtime ABI; do not resolve or inherit them for Omni.
        if isinstance(adapter, str):
            if adapter.strip().lower() not in {"auto", "identity"}:
                raise ValueError("svdquant_omni requires an Omni-compatible adapter object, not Nunchaku adapter names")
            adapter = None
        save_svdquant_omni(model, output_dir, device=device, adapter=adapter)
        return model
