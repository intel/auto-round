import pytest
import torch

from auto_round.export.export_to_itrex.model_wrapper import WeightOnlyLinear


class TestWeightOnlyLinear:
    @pytest.mark.parametrize(
        "bits, compression_dtype",
        [
            (8, torch.int16),
            (8, torch.int32),
            (8, torch.int64),
            (4, torch.int8),
            (4, torch.int16),
            (4, torch.int32),
            (4, torch.int64),
            (2, torch.int8),
            (2, torch.int16),
            (2, torch.int32),
            (2, torch.int64),
        ],
    )
    def test_pack_with_numba(self, bits, compression_dtype):
        m = torch.nn.Linear(1024, 512)
        dtype = "int"
        weight = m.weight.detach()
        group_size = 32
        origin_shape = weight.shape
        from auto_round.data_type.int import quant_tensor_sym

        origin_shape = weight.shape
        weight = weight.reshape(-1, group_size)
        qdq, scale, zp = quant_tensor_sym(weight, -1)
        if isinstance(zp, int | float):
            zp = torch.full_like(scale, zp)
        int_weight = qdq.div(scale).add(zp).clamp(0, 2 ** (bits) - 1).to(torch.int32).reshape(origin_shape)
        scale = scale.reshape(origin_shape[0], -1)
        if isinstance(zp, torch.Tensor):
            zp = zp.reshape(origin_shape[0], -1).to(torch.int32).clamp(0, 2 ** (bits) - 1)
        module_with_legacy_pack = WeightOnlyLinear(
            in_features=m.in_features,
            out_features=m.out_features,
            dtype=dtype,
            bits=bits,
            groupsize=32,
            zp=zp is not None,
            bias=m.bias is not None,
            use_optimum_format=False,
            compression_dtype=compression_dtype,
            use_legacy_pack=True,
        )
        module_with_legacy_pack.pack(
            int_weight.clone(), scale.clone(), zp.clone() if isinstance(zp, torch.Tensor) else zp, m.bias
        )
        module_with_new_pack = WeightOnlyLinear(
            in_features=m.in_features,
            out_features=m.out_features,
            dtype=dtype,
            bits=bits,
            groupsize=32,
            zp=zp is not None,
            bias=m.bias is not None,
            use_optimum_format=False,
            compression_dtype=compression_dtype,
            use_legacy_pack=False,
        )
        module_with_new_pack.pack(
            int_weight.clone(), scale.clone(), zp.clone() if isinstance(zp, torch.Tensor) else zp, m.bias
        )

        assert torch.equal(module_with_new_pack.qweight, module_with_legacy_pack.qweight)

        assert torch.equal(module_with_new_pack.qzeros, module_with_legacy_pack.qzeros)
        assert torch.equal(module_with_new_pack.scales, module_with_legacy_pack.scales)
        unpacked_int_weight = module_with_new_pack.unpack_tensor(module_with_legacy_pack.qweight)
        assert torch.equal(unpacked_int_weight, int_weight)
