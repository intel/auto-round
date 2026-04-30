from unittest.mock import patch

import torch

import auto_round.utils.device as auto_round_utils
from auto_round.utils.model import mv_module_from_gpu


class TestPackingWithNumba:

    @patch.object(auto_round_utils, "_is_tbb_installed", lambda: False)
    def test_tbb_not_installed(self):
        assert auto_round_utils.is_tbb_available() is False, "`is_tbb_available` should return False."
        assert auto_round_utils.can_pack_with_numba() is False, "`can_pack_with_numba` should return False."

    @patch.object(auto_round_utils, "_is_tbb_installed", lambda: True)
    @patch.object(auto_round_utils, "_is_tbb_configured", lambda: False)
    def test_tbb_installed_but_not_configured_right(self):
        assert auto_round_utils.is_tbb_available() is False, "`is_tbb_available` should return False."
        assert auto_round_utils.can_pack_with_numba() is False, "`can_pack_with_numba` should return False."

    @patch.object(auto_round_utils, "is_numba_available", lambda: False)
    def test_numba_not_installed(self):
        assert auto_round_utils.can_pack_with_numba() is False, "`can_pack_with_numba` should return False."


class _FakeAcceleratorParameter:
    def __init__(self):
        self.device = torch.device("cuda")
        self.requires_grad = False

    def to(self, _device):
        return torch.ones(1, dtype=torch.float32)


class _FakeMetaBuffer:
    def __init__(self):
        self.device = torch.device("meta")


class TestMetaMoveHelpers:
    def test_mv_module_from_gpu_preserves_parameter_type(self):
        module = torch.nn.Module()
        module._parameters["fake_weight"] = _FakeAcceleratorParameter()
        module._buffers["meta_marker"] = _FakeMetaBuffer()

        mv_module_from_gpu(module)

        assert isinstance(module._parameters["fake_weight"], torch.nn.Parameter)
