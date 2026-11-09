"""Exported ``quantization_config`` must only carry fields owned by the algorithm that ran.

Regression for https://github.com/intel/auto-round/issues/2224: an RTN run (``iters=0``)
exported ``enable_quanted_input: false`` because the field was collected from a
shared, algorithm-agnostic allowlist instead of from the algorithm that owns it.
"""

import json
import os

import pytest

from auto_round import AutoRound
from auto_round.algorithms.composer import AlgorithmComposer
from auto_round.algorithms.config import AlgorithmConfig, all_algorithm_serialized_fields
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.compressors.base import SERIALIZATION_KEYS

SIGNROUND_ONLY_FIELDS = {
    "iters",
    "lr",
    "minmax_lr",
    "enable_minmax_tuning",
    "enable_norm_bias_tuning",
    "enable_quanted_input",
    "gradient_accumulate_steps",
}


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def test_shared_keys_exclude_algorithm_specific_fields():
    assert not SIGNROUND_ONLY_FIELDS & set(SERIALIZATION_KEYS)
    assert SIGNROUND_ONLY_FIELDS <= all_algorithm_serialized_fields()


def test_rtn_declares_no_signround_fields():
    assert RTNConfig().to_serialized_dict() == {}
    fields = AlgorithmComposer([RTNConfig()]).serialized_algorithm_fields()
    assert not SIGNROUND_ONLY_FIELDS & set(fields)


def test_signround_exports_its_own_values():
    cfg = SignRoundConfig(iters=3, lr=0.01, enable_quanted_input=False)
    out = AlgorithmComposer([cfg]).serialized_algorithm_fields()
    assert out["iters"] == 3
    assert out["lr"] == 0.01
    assert out["enable_quanted_input"] is False


def test_every_declared_field_exists_on_its_config():
    """Contract: a config must own every field it declares for export."""
    checked = 0
    for cls in _all_subclasses(AlgorithmConfig):
        if not cls.serialized_fields:
            continue
        try:
            cfg = cls()
        except Exception:  # configs that need required args are covered elsewhere
            continue
        assert set(cfg.to_serialized_dict()) == set(cls.serialized_fields), cls.__name__
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("iters", [0, 1])
def test_exported_quantization_config_fields(iters, tiny_opt_model_path, dataloader, tmp_path):
    ar = AutoRound(tiny_opt_model_path, scheme="W4A16", iters=iters, nsamples=1, seqlen=8, dataset=dataloader)
    _, folder = ar.quantize_and_save(output_dir=str(tmp_path), format="auto_round")
    with open(os.path.join(folder, "config.json")) as f:
        qconfig = json.load(f)["quantization_config"]

    if iters == 0:
        assert not SIGNROUND_ONLY_FIELDS & set(qconfig), qconfig
    else:
        # defaults are filtered out; non-default SignRound values must survive
        assert qconfig["iters"] == 1
