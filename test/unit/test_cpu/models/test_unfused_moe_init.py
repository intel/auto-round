# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.modeling.unfused_moe``.

The package's ``__init__.py`` is the *registration* entry point: it
holds the per-model-type config (which block to patch on what
transformers class) and the two public hooks
``apply_model_monkey_patches`` and ``apply_modeling_patch``.

The tests cover:

* ``MODEL_CONFIG`` shape - regression guard against typos in the
  ``block_patch`` entries.
* ``get_checkpoint_conversion_mapping_ar`` - both the "registered"
  and "passthrough" paths.
* ``get_file_path_via_model_name`` - local dir, model name, and
  env-disabled paths.
* ``pre_check_config`` - the "ok", "unknown model_type", "wrong
  transformers version" branches.
* ``apply_model_monkey_patches`` / ``apply_modeling_patch`` - happy
  path on a real module, plus error handling.
"""

import os
import sys
import types
from unittest import mock

import pytest
import torch
import torch.nn as nn

from auto_round.modeling import unfused_moe
from auto_round.modeling.unfused_moe import (
    MODEL_CONFIG,
    apply_model_monkey_patches,
    apply_modeling_patch,
    get_checkpoint_conversion_mapping_ar,
    get_file_path_via_model_name,
    pre_check_config,
)

# ---------------------------------------------------------------------------
# MODEL_CONFIG
# ---------------------------------------------------------------------------


def test_model_config_has_required_keys():
    """Every entry must declare a non-empty ``block_patch`` list and the
    standard version-gate keys."""
    for model_type, cfg in MODEL_CONFIG.items():
        assert isinstance(cfg, dict), f"{model_type} config is not a dict"
        assert cfg.get("block_patch"), f"{model_type} has no block_patch"
        # The "min/max transformers version" key is required.
        assert (
            "min_transformers_version" in cfg or "max_transformers_version" in cfg
        ), f"{model_type} missing transformers-version gate"


def test_model_config_known_architectures():
    """Regression guard for the small set of MoE architectures we explicitly
    support.  Adding a new architecture requires updating this test
    *and* the e2e matrix.
    """
    required = {"qwen3_moe", "glm4_moe_lite", "glm4_moe", "deepseek_v3", "ernie4_5_moe"}
    assert required.issubset(MODEL_CONFIG.keys()), f"Missing architectures: {required - MODEL_CONFIG.keys()}"


def test_model_config_block_patch_paths_are_valid_python():
    """``block_patch`` is a list of (orig_path, custom_path) tuples.  Each
    path should be importable.  We use ``importlib.util.find_spec`` for
    the module part and ``getattr`` for the class.
    """
    import importlib

    for model_type, cfg in MODEL_CONFIG.items():
        for orig_path, custom_path in cfg.get("block_patch", []):
            for full in (orig_path, custom_path):
                module_path, _, class_name = full.rpartition(".")
                spec = importlib.util.find_spec(module_path)
                assert spec is not None, f"{model_type}: cannot find module {module_path}"
                mod = importlib.import_module(module_path)
                assert hasattr(mod, class_name), f"{model_type}: {module_path} does not define {class_name}"


# ---------------------------------------------------------------------------
# get_checkpoint_conversion_mapping_ar
# ---------------------------------------------------------------------------


def test_get_checkpoint_conversion_mapping_ar_registered():
    """For a model_type present in MODEL_CONFIG, return its ``checkpoint_mapping``."""
    for model_type, cfg in MODEL_CONFIG.items():
        if "checkpoint_mapping" in cfg:
            result = get_checkpoint_conversion_mapping_ar(model_type)
            assert result == cfg["checkpoint_mapping"]


def test_get_checkpoint_conversion_mapping_ar_passthrough():
    """For an unknown model_type the helper delegates to the original
    transformers mapping function.  We mock that to verify the call.
    """
    sentinel = ["x", "y"]
    with mock.patch(
        "transformers.conversion_mapping.orig_get_checkpoint_conversion_mapping",
        create=True,
        return_value=sentinel,
        new_callable=mock.MagicMock,
    ) as fake:
        # Some transformers versions don't expose ``orig_*`` yet; guard
        # against that by also patching the public name.
        with mock.patch(
            "transformers.conversion_mapping.get_checkpoint_conversion_mapping",
            side_effect=lambda mt: sentinel,
        ):
            result = get_checkpoint_conversion_mapping_ar("not_in_our_list")
    assert result == sentinel


# ---------------------------------------------------------------------------
# get_file_path_via_model_name
# ---------------------------------------------------------------------------


def test_get_file_path_via_model_name_local_dir(tmp_path):
    """A local directory containing the requested file is returned verbatim."""
    f = tmp_path / "weights.index.json"
    f.write_text("{}")
    result = get_file_path_via_model_name(str(tmp_path), "weights.index.json")
    assert result == str(f)


def test_get_file_path_via_model_name_missing_local_dir(tmp_path):
    """A local directory *without* the requested file returns the canonical
    path even if the file does not exist; the caller decides what to do
    with it.
    """
    result = get_file_path_via_model_name(str(tmp_path), "missing.json")
    assert result == os.path.join(str(tmp_path), "missing.json")


def test_get_file_path_via_model_name_uses_hf_hub(monkeypatch):
    """When the path is a model name (not a directory) the helper falls
    through to ``huggingface_hub.hf_hub_download``.
    """
    fake_path = "/tmp/fake_download/weights.index.json"

    def fake_hf_hub_download(repo_id, filename, repo_type):
        assert repo_id == "Qwen/Qwen3-0.6B"
        assert filename == "weights.index.json"
        assert repo_type == "model"
        return fake_path

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        fake_hf_hub_download,
    )
    # Make sure AR_USE_MODELSCOPE is unset (the default).
    monkeypatch.setattr("auto_round.envs.AR_USE_MODELSCOPE", False, raising=False)

    result = get_file_path_via_model_name("Qwen/Qwen3-0.6B", "weights.index.json")
    assert result == fake_path


def test_get_file_path_via_model_name_uses_modelscope(monkeypatch):
    """When ``AR_USE_MODELSCOPE`` is truthy the helper takes the ModelSCOPE
    path and joins the downloaded folder + filename.
    """
    monkeypatch.setattr("auto_round.envs.AR_USE_MODELSCOPE", True, raising=False)

    fake_folder = "/tmp/ms_snapshot"
    calls = {"n": 0}

    def fake_snapshot_download(repo_id, allow_patterns):
        calls["n"] += 1
        assert repo_id == "Qwen/Qwen3-0.6B"
        assert "weights.index.json" in allow_patterns
        return fake_folder

    # The function does ``from modelscope import snapshot_download``, so
    # we must attach the function to the *modelscope* module itself.
    fake_modelscope = types.ModuleType("modelscope")
    fake_modelscope.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)

    result = get_file_path_via_model_name("Qwen/Qwen3-0.6B", "weights.index.json")
    assert calls["n"] == 1
    assert result == os.path.join(fake_folder, "weights.index.json")


# ---------------------------------------------------------------------------
# pre_check_config
# ---------------------------------------------------------------------------


def test_pre_check_config_rejects_unknown_model_type(monkeypatch):
    """A bare nn.Module without a model_type attribute is rejected."""

    class _Bare(nn.Module):
        pass

    assert pre_check_config(_Bare()) is False


def test_pre_check_config_rejects_string_path_that_does_not_exist():
    """A non-existent HF model id should be tolerated and rejected."""
    assert pre_check_config("this-model-does-not-exist-12345") is False


def test_pre_check_config_accepts_known_model_type(monkeypatch):
    """A module whose ``config.model_type`` is in MODEL_CONFIG and whose
    transformers version is in range should be accepted - but only if
    the gate_up_proj heuristic at the bottom of the function agrees.
    """
    cfg = mock.MagicMock()
    cfg.model_type = "qwen3_moe"

    block = nn.Linear(2, 2)
    block.config = cfg
    # Provide a fake index file with no ``gate_up_proj`` keys so the
    # heuristic returns True at the bottom of pre_check_config.
    monkeypatch.setattr(
        unfused_moe,
        "get_file_path_via_model_name",
        lambda *a, **kw: "/dev/null/no-such-file",
    )

    # The function calls ``os.path.exists`` and tries to read the file;
    # we make ``open`` raise so the heuristic falls into the ``except:``
    # branch and returns True.
    def fake_open(*a, **kw):
        raise OSError("simulated missing file")

    monkeypatch.setattr("builtins.open", fake_open)
    assert pre_check_config(block) is True


def test_pre_check_config_rejects_gate_up_proj_present(monkeypatch, tmp_path):
    """If the checkpoint index contains ``gate_up_proj`` keys the function
    must return False (the model is "fused MoE" and does not need
    unfusing).
    """
    import json

    cfg = mock.MagicMock()
    cfg.model_type = "qwen3_moe"

    block = nn.Linear(2, 2)
    block.config = cfg

    # Write a fake index file that *does* contain a ``gate_up_proj`` key.
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {"layer.0.gate_up_proj.weight": "x"}}))

    monkeypatch.setattr(
        unfused_moe,
        "get_file_path_via_model_name",
        lambda *a, **kw: str(index_path),
    )
    assert pre_check_config(block) is False


def test_pre_check_config_rejects_too_old_transformers(monkeypatch):
    """A model_type whose min_transformers_version is above the
    installed version is rejected.
    """
    # Pick any model_type in MODEL_CONFIG and force its min version to
    # something absurdly high.
    some_type, some_cfg = next(iter(MODEL_CONFIG.items()))
    monkeypatch.setitem(some_cfg, "min_transformers_version", "999.0.0")

    block = nn.Linear(2, 2)
    block.config = mock.MagicMock()
    block.config.model_type = some_type
    assert pre_check_config(block) is False


def test_pre_check_config_rejects_too_new_transformers(monkeypatch):
    """A model_type whose max_transformers_version is below the
    installed version is rejected.
    """
    some_type, some_cfg = next(iter(MODEL_CONFIG.items()))
    monkeypatch.setitem(some_cfg, "max_transformers_version", "0.0.1")

    block = nn.Linear(2, 2)
    block.config = mock.MagicMock()
    block.config.model_type = some_type
    assert pre_check_config(block) is False


# ---------------------------------------------------------------------------
# apply_model_monkey_patches
# ---------------------------------------------------------------------------


def test_apply_model_monkey_patches_returns_false_for_unknown_model(monkeypatch):
    """A non-existent model id returns False without raising."""
    monkeypatch.setattr(
        "auto_round.modeling.unfused_moe.pre_check_config",
        lambda *a, **kw: False,
    )
    assert apply_model_monkey_patches("nope/never") is False


def test_apply_model_monkey_patches_happy_path(monkeypatch):
    """When ``pre_check_config`` agrees and the patch target exists, the
    upstream class is replaced in the upstream module and ``True`` is
    returned.
    """
    import importlib

    # Use a real, present architecture so the patching loop has a real
    # module to import.
    arch = "qwen3_moe" if "qwen3_moe" in MODEL_CONFIG else next(iter(MODEL_CONFIG))
    cfg_entry = MODEL_CONFIG[arch]
    orig_path, custom_path = cfg_entry["block_patch"][0]
    orig_module_path, orig_class_name = orig_path.rsplit(".", 1)

    monkeypatch.setattr(
        "auto_round.modeling.unfused_moe.pre_check_config",
        lambda *a, **kw: True,
    )
    monkeypatch.setattr(
        "auto_round.modeling.unfused_moe.AutoConfig.from_pretrained",
        classmethod(lambda cls, *a, **kw: mock.MagicMock(model_type=arch)),
    )

    # Pre-import the original module so we can verify it is mutated.
    orig_mod = importlib.import_module(orig_module_path)
    orig_class = getattr(orig_mod, orig_class_name)
    custom_mod = importlib.import_module(custom_path.rsplit(".", 1)[0])
    custom_class = getattr(custom_mod, custom_path.rsplit(".", 1)[1])

    # Pretend transformers is < 5 so the v5-only branch in the patcher
    # is skipped.  ``version.parse(...)`` is called twice, and the
    # resulting object's ``>=``/``<`` operators must return False for
    # both so we stay in the simple ``setattr`` branch.
    class _V:
        def __ge__(self, other):
            return False

        def __lt__(self, other):
            return True

    monkeypatch.setattr("auto_round.modeling.unfused_moe.version.parse", lambda v: _V())

    result = apply_model_monkey_patches("fake/model")
    assert result is True
    # The upstream class has been replaced.
    assert getattr(orig_mod, orig_class_name) is custom_class
    # Restore for hygiene.
    setattr(orig_mod, orig_class_name, orig_class)


def test_apply_model_monkey_patches_swallows_import_errors(monkeypatch):
    """If the upstream module cannot be imported, the helper must log a
    warning and return False (it does not raise).
    """
    import importlib

    arch = next(iter(MODEL_CONFIG))
    cfg_entry = MODEL_CONFIG[arch]
    orig_path, _ = cfg_entry["block_patch"][0]
    orig_module_path, _ = orig_path.rsplit(".", 1)

    monkeypatch.setattr(
        "auto_round.modeling.unfused_moe.pre_check_config",
        lambda *a, **kw: True,
    )
    monkeypatch.setattr(
        "auto_round.modeling.unfused_moe.AutoConfig.from_pretrained",
        classmethod(lambda cls, *a, **kw: mock.MagicMock(model_type=arch)),
    )

    def fake_import_module(name, package=None):
        if name == orig_module_path:
            raise ImportError("simulated import failure")
        return importlib.import_module(name, package)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    assert apply_model_monkey_patches("fake/model") is False


# ---------------------------------------------------------------------------
# apply_modeling_patch
# ---------------------------------------------------------------------------


def test_apply_modeling_patch_returns_false_when_pre_check_fails(monkeypatch):
    """A model that fails pre-check is not patched and returns False."""
    block = nn.Linear(2, 2)
    block.config = mock.MagicMock()

    monkeypatch.setattr(
        "auto_round.modeling.unfused_moe.pre_check_config",
        lambda *a, **kw: False,
    )
    assert apply_modeling_patch(block) is False


def test_apply_modeling_patch_replaces_modules_in_place(monkeypatch):
    """``apply_modeling_patch`` walks the model, finds matching modules
    and replaces them via ``model.set_submodule``.
    """
    # Find a real architecture that ships with the test environment.
    arch = "qwen3_moe" if "qwen3_moe" in MODEL_CONFIG else next(iter(MODEL_CONFIG))
    cfg_entry = MODEL_CONFIG[arch]
    orig_path, _ = cfg_entry["block_patch"][0]
    orig_module_path, orig_class_name = orig_path.rsplit(".", 1)

    import importlib

    orig_mod = importlib.import_module(orig_module_path)
    orig_class = getattr(orig_mod, orig_class_name)

    # Bypass the real ``__init__`` to get a bare instance - we only
    # need an object whose ``__class__`` is ``orig_class`` so the
    # ``isinstance(m, orig_class)`` check in apply_modeling_patch
    # returns True.
    bare = orig_class.__new__(orig_class)
    nn.Module.__init__(bare)

    parent = nn.Module()
    parent.the_block = bare  # type: ignore[attr-defined]
    parent.config = mock.MagicMock(model_type=arch)
    # ``custom_class(model.config)`` is invoked inside
    # ``apply_modeling_patch``; the real class' ``__init__`` reads
    # several attributes off the config, so we set the bare minimum
    # to avoid raising inside the constructor.  Values mirror the
    # tiny fake used in the unfused_moe_blocks tests.
    cfg = parent.config
    cfg.num_experts = 1
    cfg.num_experts_per_tok = 1
    cfg.norm_topk_prob = False
    cfg.moe_intermediate_size = 4
    cfg.hidden_size = 4
    cfg.hidden_act = "silu"

    # Make pre_check_config agree.
    monkeypatch.setattr(
        "auto_round.modeling.unfused_moe.pre_check_config",
        lambda *a, **kw: True,
    )

    # ``set_submodule`` is only present on PreTrainedModel; the helper
    # we test against is called on a bare ``nn.Module``, so we add a
    # tiny stub via ``types.MethodType`` rather than monkey-patching
    # the class.
    def _set_submodule(self, name, module, *args, **kwargs):  # noqa: ARG001
        # Match PyTorch's API: rebuild ``_modules`` and notify.
        self._modules[name] = module

    parent.set_submodule = _set_submodule.__get__(parent)  # type: ignore[attr-defined]

    result = apply_modeling_patch(parent)
    assert result is True


def test_apply_modeling_patch_returns_false_on_import_error(monkeypatch):
    """If the replacement module cannot be imported, the helper returns
    False and does not raise.
    """
    import importlib

    arch = next(iter(MODEL_CONFIG))
    cfg_entry = MODEL_CONFIG[arch]
    _, custom_path = cfg_entry["block_patch"][0]
    custom_module_path = custom_path.rsplit(".", 1)[0]

    parent = nn.Module()
    parent.config = mock.MagicMock(model_type=arch)

    monkeypatch.setattr(
        "auto_round.modeling.unfused_moe.pre_check_config",
        lambda *a, **kw: True,
    )

    def fake_import_module(name, package=None):
        if name == custom_module_path:
            raise ImportError("simulated custom-module import failure")
        return importlib.import_module(name, package)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    assert apply_modeling_patch(parent) is False
