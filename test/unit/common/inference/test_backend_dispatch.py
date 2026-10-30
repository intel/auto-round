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
"""Tests for the QuantLinear dispatch / requirement helpers in ``auto_round/inference/backend.py``.

Optional third-party kernels (gptqmodel, auto_gptq, awq, humming, ark, habana)
are replaced with fakes in ``sys.modules`` so every branch runs on CPU.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from auto_round.experimental import qmodules as ar_qmodules
from auto_round.inference import backend as be

CFG = {"bits": 4, "group_size": 128, "sym": True}


class TestDynamicImportInferenceLinear:
    @pytest.mark.parametrize(
        "backend_name, attr",
        [
            ("auto_round:torch_fp8_static", "WeightFP8ActFP8StaticQuantLinear"),
            ("auto_round:torch_mxfp8", "MXFP8QuantLinear"),
            ("auto_round:torch_mxint4", "MXINT4QuantLinear"),
            ("auto_round:torch_mxfp4", "MXFP4QuantLinear"),
            ("auto_round:cute_nvfp4_e5m3_gs", "CuteNVFP4E5M3GSQuantLinear"),
            ("auto_round:torch_nvfp4_e5m3_gs", "NVFP4E5M3GSQuantLinear"),
            ("auto_round:cute_nvfp4_e5m3", "CuteNVFP4E5M3QuantLinear"),
            ("auto_round:torch_nvfp4_e5m3", "NVFP4E5M3QuantLinear"),
            ("auto_round:torch_nvfp4", "NVFP4QuantLinear"),
            ("auto_round:fake", "FakeActQuantLinear"),
        ],
    )
    def test_builtin_qmodules(self, backend_name, attr):
        assert be.dynamic_import_inference_linear(backend_name, CFG) is getattr(ar_qmodules, attr)

    def test_torch_backends(self):
        from auto_round_extension.torch.qlinear_torch import QuantLinear
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear as QuantLinearZp

        assert be.dynamic_import_inference_linear("auto_round:torch", CFG) is QuantLinear
        assert be.dynamic_import_inference_linear("auto_round:torch_zp", CFG) is QuantLinearZp

    def test_mlx_with_gptq_checkpoint_uses_torch_linear(self):
        from auto_round_extension.torch.qlinear_torch import QuantLinear

        assert be.dynamic_import_inference_linear("auto_round:mlx", CFG, packing_format="auto_round") is QuantLinear

    @pytest.mark.parametrize(
        "backend_name, missing_module, match",
        [
            ("auto_round:humming", "humming", "humming"),
            ("auto_round:ark", "auto_round_kernel", "auto-round-lib"),
            ("auto_round:hpu", "habana_frameworks.torch.hpu", "hpu"),
        ],
    )
    def test_missing_optional_kernels_raise(self, backend_name, missing_module, match):
        with patch.dict(sys.modules, {missing_module: None}), pytest.raises(ImportError, match=match):
            be.dynamic_import_inference_linear(backend_name, CFG)

    @pytest.mark.parametrize(
        "backend_name, expected",
        [("ark_zp", "QuantLinearGPTQ"), ("ark_awq", "QuantLinearAWQ"), ("auto_round_kernel", "QuantLinear")],
    )
    def test_ark_variants(self, backend_name, expected):
        qlinear = types.SimpleNamespace(QuantLinearGPTQ="gptq", QuantLinearAWQ="awq", QuantLinear="plain")
        fake_ark = types.ModuleType("auto_round_kernel")
        fake_ark.qlinear = qlinear
        with patch.dict(sys.modules, {"auto_round_kernel": fake_ark, "auto_round_kernel.qlinear": qlinear}):
            assert be.dynamic_import_inference_linear(backend_name, CFG) == getattr(qlinear, expected)

    def test_gptqmodel_and_autogptq_delegation(self):
        with patch.object(be, "get_gptqmodel_infer_linear", return_value="gm") as gm:
            assert be.dynamic_import_inference_linear("gptqmodel:awq_marlin", CFG) == "gm"
        gm.assert_called_once_with("gptqmodel:awq_marlin", is_awq=True)

        with patch.object(be, "get_autogptq_infer_linear", return_value="ag") as ag:
            assert be.dynamic_import_inference_linear("auto_gptq:exllamav2", CFG) == "ag"
        ag.assert_called_once_with("auto_gptq:exllamav2", 4, 128, True)

    def test_autoawq_fallback(self):
        linear_mod = types.SimpleNamespace(WQLinear_GEMM="wq")
        with patch.dict(sys.modules, {"awq": types.ModuleType("awq"), "awq.modules": types.ModuleType("m")}):
            sys.modules["awq.modules.linear"] = linear_mod
            try:
                assert be.dynamic_import_inference_linear("auto_awq:gemm", CFG) == "wq"
            finally:
                sys.modules.pop("awq.modules.linear", None)

    def test_unsupported_backend(self):
        with pytest.raises(ValueError, match="unsupported backend"):
            be.dynamic_import_inference_linear("auto_round:does_not_exist", CFG)


class TestGptqmodelInferLinear:
    def _fake_module(self):
        names = [
            "AwqMarlinLinear",
            "AwqMarlinQuantLinear",
            "AwqGEMMLinear",
            "AwqGEMMQuantLinear",
            "ExllamaV2Linear",
            "ExllamaV2QuantLinear",
            "TorchLinear",
            "TorchQuantLinear",
        ]
        return types.SimpleNamespace(**{n: n for n in names})

    @pytest.mark.parametrize(
        "version, backend_name, is_awq, expected",
        [
            ("7.0.0", "gptqmodel:awq_marlin", True, "AwqMarlinLinear"),
            ("6.0.3", "gptqmodel:awq_marlin", True, "AwqMarlinQuantLinear"),
            ("6.0.3", "gptqmodel:awq_gemm", True, "AwqGEMMQuantLinear"),
            ("7.1.0", "gptqmodel:exllamav2", False, "ExllamaV2Linear"),
            ("6.0.3", "gptqmodel:exllamav2", False, "ExllamaV2QuantLinear"),
            ("6.0.3", "gptqmodel:torch", False, "TorchQuantLinear"),
        ],
    )
    def test_selects_class_by_version(self, version, backend_name, is_awq, expected):
        fake = types.SimpleNamespace(__version__=version)
        with (
            patch.object(be, "safe_import_gptqmodel", return_value=fake),
            patch.object(be, "import_module", return_value=self._fake_module()),
        ):
            assert be.get_gptqmodel_infer_linear(backend_name, is_awq=is_awq) == expected

    def test_marlin_uses_extension_factory(self):
        fake = types.SimpleNamespace(__version__="7.0.0")
        with (
            patch.object(be, "safe_import_gptqmodel", return_value=fake),
            patch("auto_round_extension.cuda.gptqmodel_marlin.get_marlin_layer", return_value="marlin_cls"),
        ):
            assert be.get_gptqmodel_infer_linear("gptqmodel:marlin", is_awq=False) == "marlin_cls"

    def test_unsupported(self):
        with (
            patch.object(be, "safe_import_gptqmodel", return_value=types.SimpleNamespace(__version__="7.0.0")),
            pytest.raises(ValueError, match="Unsupported"),
        ):
            be.get_gptqmodel_infer_linear("gptqmodel:unknown")

    def test_safe_import_restores_default_dtype(self):
        fake = types.ModuleType("gptqmodel")
        orig = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            with patch.dict(sys.modules, {"gptqmodel": fake}):
                assert be.safe_import_gptqmodel() is fake
            assert torch.get_default_dtype() == torch.float64
        finally:
            torch.set_default_dtype(orig)


class TestAutogptqInferLinear:
    @pytest.fixture
    def fake_autogptq(self):
        importer = MagicMock(return_value="QL")
        import_utils = types.SimpleNamespace(dynamically_import_QuantLinear=importer)
        mods = {
            "auto_gptq": types.ModuleType("auto_gptq"),
            "auto_gptq.utils": types.ModuleType("auto_gptq.utils"),
            "auto_gptq.utils.import_utils": import_utils,
        }
        with patch.dict(sys.modules, mods):
            yield importer

    @pytest.mark.parametrize(
        "backend_name, expected",
        [
            ("gptq:qigen", {"use_qigen": True, "use_triton": False}),
            ("gptq:triton", {"use_triton": True}),
            ("gptq:marlin", {"use_marlin": True, "use_triton": False}),
            ("gptq:exllamav2", {"disable_exllamav2": False, "use_marlin": False}),
            ("gptq:exllamav1", {"disable_exllama": False}),
            ("gptq:cuda", {"disable_exllama": True, "disable_exllamav2": True}),
        ],
    )
    def test_new_autogptq_flags(self, fake_autogptq, backend_name, expected):
        with patch.object(be, "get_library_version", return_value="0.7.2"):
            assert be.get_autogptq_infer_linear(backend_name, bits=4, group_size=32, sym=True) == "QL"
        kwargs = fake_autogptq.call_args.kwargs
        assert kwargs["bits"] == 4 and kwargs["group_size"] == 32 and kwargs["desc_act"] is False
        for k, v in expected.items():
            assert kwargs[k] == v, (k, kwargs)

    def test_legacy_autogptq_uses_disable_marlin(self, fake_autogptq):
        with patch.object(be, "get_library_version", return_value="0.7.1"):
            be.get_autogptq_infer_linear("gptq:marlin")
        kwargs = fake_autogptq.call_args.kwargs
        assert kwargs["disable_marlin"] is False
        assert "use_marlin" not in kwargs and "use_tritonv2" not in kwargs


class TestProcessRequirement:
    def test_all_satisfied_is_noop(self):
        with patch.object(be, "logger") as log:
            be.process_requirement(["torch>=1.0"], logger_level="warning")
        log.warning.assert_not_called()
        log.error.assert_not_called()

    def test_missing_requirements_warning_level(self):
        available = MagicMock(return_value=(True, "spec"))
        with patch.object(be, "logger") as log:
            be.process_requirement(
                ["gptqmodel>=999.0", "not_a_real_pkg_xyz>=1.0", available, lambda: (False, "git+https://x/y.git")],
                logger_level="warning",
            )
        messages = [c.args[0] for c in log.warning.call_args_list]
        assert messages[0].startswith("Better backend is found")
        assert 'pip install -v "gptqmodel>=999.0" --no-build-isolation' in messages[-1]
        assert '"not_a_real_pkg_xyz>=1.0"' in messages[-1] and '"git+https://x/y.git"' in messages[-1]
        available.assert_called_once()

    def test_missing_requirements_error_level_exits(self):
        with patch.object(be, "logger") as log, pytest.raises(SystemExit):
            be.process_requirement(["not_a_real_pkg_xyz>=1.0"])
        assert log.error.call_args_list[0].args[0].startswith("Inference requires")


class TestCompatibilityHelpers:
    def test_cute_nvfp4_e5m3_checker_rejects_bad_group_size(self):
        cfg = {"bits": 4, "group_size": 16, "sym": True}
        assert be.cute_nvfp4_e5m3_checker(17, 32, cfg) is False

    def test_cute_nvfp4_e5m3_checker_without_cute(self):
        cfg = {"bits": 4, "group_size": 16, "sym": True}
        with patch.dict(sys.modules, {"auto_round_extension.cuda.cute_nvfp4_e5m3": None}):
            assert be.cute_nvfp4_e5m3_checker(64, 32, cfg) is False

    def test_check_compatible_callable_requirement_and_system_filter(self):
        info = be.BackendInfo(
            device=["cpu"],
            sym=[True],
            packing_format=["fake_pack"],
            bits=[4],
            requirements=[lambda: (False, "spec")],
        )
        with patch.dict(be.BackendInfos, {"unit_test_backend": info}):
            assert be.check_compatible("unit_test_backend", "cpu", {"bits": 4}, "fake_pack", 64, 64) is False
            assert (
                be.check_compatible(
                    "unit_test_backend", "cpu", {"bits": 4}, "fake_pack", 64, 64, check_requirements=False
                )
                is True
            )
            info.systems = ["no_such_os"]
            assert (
                be.check_compatible(
                    "unit_test_backend", "cpu", {"bits": 4}, "fake_pack", 64, 64, check_requirements=False
                )
                is False
            )

    def test_highest_priority_backend(self):
        low = be.BackendInfo(device=["cpu"], sym=[True], packing_format=["fake_pack"], bits=[4], priority=1)
        high = be.BackendInfo(
            device=["cpu"], sym=[True], packing_format=["fake_pack"], bits=[4], group_size=[128], priority=5
        )
        act = be.BackendInfo(
            device=["cpu"], sym=[True], packing_format=["fake_pack"], bits=[4], act_bits=[8], priority=10
        )
        q = types.SimpleNamespace(bits=4, group_size=128, sym=True)
        with patch.object(be, "BackendInfos", {"low": low, "high": high, "act": act}):
            # act fields absent from the scheme are treated as supported
            assert be.get_highest_priority_backend(q, "cpu", "fake_pack") == "act"
            q.act_bits = 4
            assert be.get_highest_priority_backend(q, "cpu", "fake_pack") is None
            del q.act_bits
            act.packing_format = ["other_pack"]
            assert be.get_highest_priority_backend(q, "cpu", "fake_pack") == "high"
            q.group_size = 32
            assert be.get_highest_priority_backend(q, "cpu", "fake_pack") == "low"
            assert be.get_highest_priority_backend(q, "cpu", "unknown_pack") is None
            assert be.get_highest_priority_backend(q, "cuda", "fake_pack") is None
            q.sym = False
            assert be.get_highest_priority_backend(q, "cpu", "fake_pack") is None

    def test_highest_priority_backend_system_filter(self):
        info = be.BackendInfo(device=["cpu"], sym=[True], packing_format=["p"], bits=[4], systems=["no_such_os"])
        q = types.SimpleNamespace(bits=4, group_size=128, sym=True)
        with patch.object(be, "BackendInfos", {"only": info}):
            assert be.get_highest_priority_backend(q, "cpu", "p") is None
            info.systems = None
            assert be.get_highest_priority_backend(q, "cpu", "p") == "only"
            q.bits = 2
            assert be.get_highest_priority_backend(q, "cpu", "p") is None


class TestBackendLookup:
    @pytest.fixture
    def infos(self):
        a = be.BackendInfo(device=["cpu"], sym=[True], packing_format=["p"], bits=[4], priority=1, alias=["al"])
        b = be.BackendInfo(device=["cpu"], sym=[True], packing_format=["p"], bits=[4], priority=3)
        c = be.BackendInfo(
            device=["cuda"], sym=[True], packing_format=["p"], bits=[4], priority=9, requirements=["not_a_pkg>=1"]
        )
        with patch.object(be, "BackendInfos", {"a": a, "b": b, "c": c}):
            yield

    def test_find_backend(self, infos):
        assert be.find_backend("missing") is None
        assert be.find_backend("al") == "a"
        assert be.find_backend("b", orig_backend="a") == "b"

    def test_get_layer_backend(self, infos):
        cfg = {"bits": 4, "sym": True}
        assert be.get_layer_backend("cpu", "auto", "p", cfg, 64, 64) == "b"
        assert be.get_layer_backend("cpu", "al", "p", cfg, 64, 64) == "a"
        assert be.get_layer_backend("cpu", "auto", "other", cfg, 64, 64) == ""

    def test_get_layer_backend_reports_missing_requirement(self, infos):
        with patch.object(be, "process_requirement") as proc:
            assert be.get_layer_backend("cuda", "auto", "p", {"bits": 4}, 64, 64) == ""
        proc.assert_called_once_with(["not_a_pkg>=1"], target_device="cuda")

    def test_get_all_compatible_backend_ignores_requirements(self, infos):
        assert sorted(be.get_all_compatible_backend("cuda", "p", {"bits": 4}, 64, 64)) == ["c"]
