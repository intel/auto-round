import timeit

import numpy as np
import pytest
import torch

import auto_round.utils as auto_round_utils
import os
ref_fn = auto_round_utils.Packer.pack_2d_tensor
fn1 = auto_round_utils.Packer.pack_tensor_with_numpy_opt_np_numba
fn2 = auto_round_utils.Packer.pack_tensor_with_numpy_opt_np_numba_v2


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("out_features", [128, 1024, 5120, 13824])
@pytest.mark.parametrize("in_features", [1024, 13824])
def test_correctness(bits, in_features, out_features):
    _max = 2 ** (bits - 1) - 1
    raw_tensor = torch.randint(0, _max, (out_features, in_features), dtype=torch.int8)
    n_pack = 32 // bits
    raw_np = raw_tensor.numpy()
    ref = ref_fn(raw_tensor, n_pack, bits)
    res = fn2(raw_np, n_pack, bits)
    assert np.array_equal(ref.numpy(), res), f"ref:{ref}, res:{res}"


PROFILE_PACK = os.environ.get("PROFILE_PACK", "0") == "1"
@pytest.mark.skipif(not PROFILE_PACK, reason="skip profiling, set `PROFILE_PACK=1` to enable it.")
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("out_features", [128, 1024, 5120, 13824])
@pytest.mark.parametrize("in_features", [1024, 13824])
def test_pack(bits, in_features, out_features):
    _max = 2 ** (bits - 1) - 1
    raw_tensor = torch.randint(0, _max, (out_features, in_features), dtype=torch.int8)
    n_pack = 32 // bits
    iters = 20
    raw_np = raw_tensor.numpy()
    time_ref = timeit.timeit(lambda: ref_fn(raw_tensor, n_pack, bits), number=iters)
    time_res = timeit.timeit(lambda: fn2(raw_np, n_pack, bits), number=iters)
    print(f"ref : {time_ref},  res: {time_res}, speed up: {time_ref / time_res}")
    # numactl -l -C 0-24  pytest -sv ./test/test_utility.py 