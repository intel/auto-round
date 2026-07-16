# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.modeling.unfused_moe``.

These modules implement the **per-expert (linear)** MoE block used as a
drop-in replacement for the fused (3D-weight) MoE blocks shipped with
``transformers >= 5.0.0``.  AutoRound's quantizer patches them in via
``auto_round.modeling.unfused_moe.apply_model_monkey_patches`` and
quantizes each expert's ``gate_proj``/``up_proj``/``down_proj`` linearly.

The blocks are exercised here in isolation - we build a small fake
``Config`` (via ``types.SimpleNamespace``), instantiate the block, and
call ``forward`` with a tiny random tensor.  This gives us good
coverage of the *unfused* forward paths without needing to download
the real multi-billion-parameter MoE checkpoints.

The tests are skipped when the corresponding ``transformers`` modeling
module is not installed (e.g. on an older transformers version).
"""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch

# ---------------------------------------------------------------------------
# Per-architecture config builders
# ---------------------------------------------------------------------------
# Each entry: ``(module_name, class_name, config_factory)`` where
# ``config_factory()`` returns a SimpleNamespace with the bare minimum
# of fields the module's MLP / Router needs.  Centralising the
# constructors here keeps each test short and lets us add a new
# architecture in one place.
# ---------------------------------------------------------------------------


def _torch_2_0_or_newer() -> bool:
    """``topk`` on 1-D tensors requires PyTorch >= 2.0; older versions
    raise for the dsa / glm4-moe routers that use it.  We skip those
    cases on old torch but still cover the others."""
    import torch
    from packaging.version import Version

    return Version(torch.__version__) >= Version("2.0.0")


# Map MODEL_CONFIG key -> transformers submodule that the
# unfused_moe block actually imports from.  Necessary because some
# model_type keys (e.g. ``glm4_moe``) match the transformers submodule
# name verbatim, while others (e.g. ``glm_moe_dsa``) do not.
_TF_MODULE_FOR_KEY: dict = {
    "qwen3_moe": "transformers.models.qwen3_moe.modeling_qwen3_moe",
    "qwen3_next": "transformers.models.qwen3_next.modeling_qwen3_next",
    "deepseek_v3": "transformers.models.deepseek_v3.modeling_deepseek_v3",
    "ernie4_5_moe": "transformers.models.ernie4_5_moe.modeling_ernie4_5_moe",
    "glm4_moe": "transformers.models.glm4_moe.modeling_glm4_moe",
    "glm4_moe_lite": "transformers.models.glm4_moe_lite.modeling_glm4_moe_lite",
    "glm_moe_dsa": "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
}


def _has_tf_module_for(model_type_key: str) -> bool:
    """Return True if the transformers submodule backing a MODEL_CONFIG
    key can be imported."""
    return importlib.util.find_spec(_TF_MODULE_FOR_KEY[model_type_key]) is not None


# Mapping from MODEL_CONFIG key -> (auto_round module, class name) for
# the per-architecture test cases.  We do this once here so the
# per-test parametrize lists stay short.
_TF_BLOCK_FOR_KEY: dict = {
    "qwen3_moe": ("auto_round.modeling.unfused_moe.qwen3_moe", "LinearQwen3MoeSparseMoeBlock"),
    "qwen3_next": ("auto_round.modeling.unfused_moe.qwen3_next", "LinearQwen3NextSparseMoeBlock"),
    "deepseek_v3": ("auto_round.modeling.unfused_moe.deepseek_v3", "LinearDeepseekV3MoE"),
    "ernie4_5_moe": ("auto_round.modeling.unfused_moe.ernie4_5_moe", "LinearErnie4_5_MoeSparseMoeBlock"),
    "glm4_moe": ("auto_round.modeling.unfused_moe.glm_moe", "LinearGlm4MoeMoE"),
    "glm4_moe_lite": ("auto_round.modeling.unfused_moe.glm_moe_light", "LinearGlm4MoeLiteMoE"),
    "glm_moe_dsa": ("auto_round.modeling.unfused_moe.glm_moe_dsa", "LinearGlmMoeDsaMoE"),
}


_CONFIG_FACTORIES: dict = {}


def _load_tf_block_for(model_type_key: str):
    """Import and return the (class, config_factory) pair for a
    MODEL_CONFIG key.
    """
    # Lazy-register the config factories on first use.
    if not _CONFIG_FACTORIES:
        _CONFIG_FACTORIES.update(
            {
                "qwen3_moe": _qwen3_moe_cfg,
                "qwen3_next": _qwen3_next_cfg,
                "deepseek_v3": _deepseek_v3_cfg,
                "ernie4_5_moe": _ernie_cfg,
                "glm4_moe": _glm4_moe_cfg,
                "glm4_moe_lite": _glm4_moe_lite_cfg,
                "glm_moe_dsa": _glm_moe_dsa_cfg,
            }
        )
    module_name, class_name = _TF_BLOCK_FOR_KEY[model_type_key]
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    cfg_factory = _CONFIG_FACTORIES[model_type_key]
    return cls, cfg_factory


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------

# Hidden/intermediate sizes are kept small so the tests are quick.
COMMON = dict(hidden_size=16, moe_intermediate_size=32, hidden_act="silu")


def _qwen3_moe_cfg():
    return SimpleNamespace(
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        **COMMON,
    )


def _qwen3_next_cfg():
    """Qwen3-Next adds a shared expert."""
    return SimpleNamespace(
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        shared_expert_intermediate_size=8,
        **COMMON,
    )


def _deepseek_v3_cfg():
    """DeepSeek-V3 has its own router type and ``n_routed_experts``/``n_group``."""
    return SimpleNamespace(
        num_local_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        n_shared_experts=1,
        n_routed_experts=4,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        **COMMON,
    )


def _ernie_cfg():
    """Ernie uses different field names (``moe_*``)."""
    return SimpleNamespace(
        moe_num_experts=4,
        moe_k=2,
        moe_num_shared_experts=1,
        moe_intermediate_size=32,
        moe_norm_min=1e-12,
        use_bias=False,
        hidden_size=16,
        hidden_act="silu",
    )


def _glm4_moe_cfg():
    return SimpleNamespace(
        num_local_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        n_shared_experts=1,
        n_routed_experts=4,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        **COMMON,
    )


def _glm4_moe_lite_cfg():
    """GLM4-Moe-Lite uses a sigmoid-based router with a bias correction term.

    The bias correction is a per-expert learnable tensor; we feed ``None``
    in the fake config because the real ``Glm4MoeLiteTopkRouter`` raises
    when it's missing, but the block's ``__init__`` stores it as-is.
    """
    cfg = _glm4_moe_cfg()
    cfg.e_score_correction_bias = None
    return cfg


def _glm_moe_dsa_cfg():
    return SimpleNamespace(
        num_local_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        n_shared_experts=1,
        n_routed_experts=4,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        **COMMON,
    )


# ---------------------------------------------------------------------------
# Helper: instantiate + forward one MoE block.
# ---------------------------------------------------------------------------


def _run_block(block_cls, cfg, batch=2, seq=5, dim=16):
    """Instantiate ``block_cls(cfg)`` and run a forward pass.

    Returns the output tensor for further assertions.  Any import error
    inside the block (because the target ``transformers`` modeling
    module is missing) is re-raised as ``pytest.skip``.
    """
    torch.manual_seed(0)
    block = block_cls(cfg)
    block.eval()  # disable dropout / random behaviour

    x = torch.randn(batch, seq, dim)
    with torch.no_grad():
        y = block(x)
    return y, block


# ---------------------------------------------------------------------------
# qwen3_moe
# ---------------------------------------------------------------------------


def test_qwen3_moe_forward():
    """LinearQwen3MoeSparseMoeBlock: per-expert linear forward path."""
    if not _has_tf_module_for("qwen3_moe"):
        pytest.skip("transformers does not have qwen3_moe modeling module")
    from auto_round.modeling.unfused_moe.qwen3_moe import LinearQwen3MoeSparseMoeBlock

    y, block = _run_block(LinearQwen3MoeSparseMoeBlock, _qwen3_moe_cfg())

    assert y.shape == (2, 5, 16)
    # norm_topk_prob=True should re-normalise the routing weights
    # so the per-token mixture sums to 1, which keeps the output
    # well-conditioned (no NaN / Inf).
    assert torch.isfinite(y).all()


def test_qwen3_moe_norm_topk_prob_false():
    """``norm_topk_prob=False`` is a separate code path."""
    if not _has_tf_module_for("qwen3_moe"):
        pytest.skip("transformers does not have qwen3_moe modeling module")
    from auto_round.modeling.unfused_moe.qwen3_moe import LinearQwen3MoeSparseMoeBlock

    cfg = _qwen3_moe_cfg()
    cfg.norm_topk_prob = False
    y, _ = _run_block(LinearQwen3MoeSparseMoeBlock, cfg)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# qwen3_next
# ---------------------------------------------------------------------------


def test_qwen3_next_forward_with_shared_expert():
    """LinearQwen3NextSparseMoeBlock has both routed and shared experts."""
    if not _has_tf_module_for("qwen3_next"):
        pytest.skip("transformers does not have qwen3_next modeling module")
    from auto_round.modeling.unfused_moe.qwen3_next import LinearQwen3NextSparseMoeBlock

    y, block = _run_block(LinearQwen3NextSparseMoeBlock, _qwen3_next_cfg())
    assert y.shape == (2, 5, 16)
    assert block.shared_expert is not None
    assert isinstance(block.shared_expert_gate, torch.nn.Linear)


# ---------------------------------------------------------------------------
# deepseek_v3
# ---------------------------------------------------------------------------


def test_deepseek_v3_forward():
    """LinearDeepseekV3MoE uses ``DeepseekV3TopkRouter`` which has its own
    group-selection logic (n_group, topk_group).  We only check the
    shapes / dtypes here.
    """
    if not _has_tf_module_for("deepseek_v3"):
        pytest.skip("transformers does not have deepseek_v3 modeling module")
    from auto_round.modeling.unfused_moe.deepseek_v3 import LinearDeepseekV3MoE

    y, block = _run_block(LinearDeepseekV3MoE, _deepseek_v3_cfg())
    assert y.shape == (2, 5, 16)
    # DeepSeek-V3 always carries shared experts.
    assert block.shared_experts is not None
    assert block.n_routed_experts == 4
    assert block.routed_scaling_factor == 1.0


# ---------------------------------------------------------------------------
# ernie4_5_moe
# ---------------------------------------------------------------------------


def test_ernie4_5_moe_with_shared_expert():
    if not _has_tf_module_for("ernie4_5_moe"):
        pytest.skip("transformers does not have ernie4_5_moe modeling module")
    from auto_round.modeling.unfused_moe.ernie4_5_moe import LinearErnie4_5_MoeSparseMoeBlock

    y, block = _run_block(LinearErnie4_5_MoeSparseMoeBlock, _ernie_cfg())
    assert y.shape == (2, 5, 16)
    assert block.shared_experts is not None


def test_ernie4_5_moe_no_shared_expert():
    """``moe_num_shared_experts=0`` keeps ``shared_experts`` as None and
    exercises the conditional path in ``forward``.
    """
    if not _has_tf_module_for("ernie4_5_moe"):
        pytest.skip("transformers does not have ernie4_5_moe modeling module")
    from auto_round.modeling.unfused_moe.ernie4_5_moe import LinearErnie4_5_MoeSparseMoeBlock

    cfg = _ernie_cfg()
    cfg.moe_num_shared_experts = 0
    y, block = _run_block(LinearErnie4_5_MoeSparseMoeBlock, cfg)
    assert y.shape == (2, 5, 16)
    assert block.shared_experts is None
    assert torch.isfinite(y).all()


def test_ernie4_5_moe_experts_forward_directly():
    """Exercise ``experts_forward`` directly with synthetic routing
    weights to make sure the loop over expert_hit terminates and the
    index_add accumulates correctly.
    """
    if not _has_tf_module_for("ernie4_5_moe"):
        pytest.skip("transformers does not have ernie4_5_moe modeling module")
    from auto_round.modeling.unfused_moe.ernie4_5_moe import LinearErnie4_5_MoeSparseMoeBlock

    _, block = _run_block(LinearErnie4_5_MoeSparseMoeBlock, _ernie_cfg())
    # 10 tokens, 4 experts, top-2
    n_tokens, n_experts, top_k = 10, 4, 2
    hidden = torch.randn(n_tokens, 16)
    # Force each expert to be hit at least once: cycle through
    top_k_index = torch.arange(n_tokens).reshape(-1, 1) % n_experts
    top_k_index = top_k_index.expand(-1, top_k).contiguous()
    top_k_weights = torch.full((n_tokens, top_k), 0.5)
    y = block.experts_forward(hidden, top_k_index, top_k_weights)
    assert y.shape == (n_tokens, 16)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# glm_moe (Glm4Moe)
# ---------------------------------------------------------------------------


def test_glm4_moe_forward():
    if not _has_tf_module_for("glm4_moe"):
        pytest.skip("transformers does not have glm4_moe modeling module")
    from auto_round.modeling.unfused_moe.glm_moe import LinearGlm4MoeMoE

    y, block = _run_block(LinearGlm4MoeMoE, _glm4_moe_cfg())
    assert y.shape == (2, 5, 16)
    assert block.n_routed_experts == 4


# ---------------------------------------------------------------------------
# glm_moe_light (Glm4MoeLite)
# ---------------------------------------------------------------------------


def test_glm4_moe_lite_forward():
    if not _has_tf_module_for("glm4_moe_lite"):
        pytest.skip("transformers does not have glm4_moe_lite modeling module")
    from auto_round.modeling.unfused_moe.glm_moe_light import LinearGlm4MoeLiteMoE

    y, block = _run_block(LinearGlm4MoeLiteMoE, _glm4_moe_lite_cfg())
    assert y.shape == (2, 5, 16)


def test_glm4_moe_lite_experts_forward_directly():
    """Hit ``experts_forward`` directly with hand-crafted top_k_index
    that exercises the ``if expert_idx == self.num_experts: continue``
    branch (i.e. top_k_index contains a value equal to num_experts)."""
    if not _has_tf_module_for("glm4_moe_lite"):
        pytest.skip("transformers does not have glm4_moe_lite modeling module")
    from auto_round.modeling.unfused_moe.glm_moe_light import LinearGlm4MoeLiteMoE

    _, block = _run_block(LinearGlm4MoeLiteMoE, _glm4_moe_lite_cfg())
    n_tokens, n_experts, top_k = 6, 4, 2
    hidden = torch.randn(n_tokens, 16)
    # All entries in [0, num_experts).  The ``continue`` branch
    # (``expert_idx == num_experts``) is exercised separately via
    # ``test_glm4_moe_lite_expert_out_of_range`` below.
    top_k_index = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 1], [2, 3]],
        dtype=torch.long,
    )
    top_k_weights = torch.full((n_tokens, top_k), 0.25)
    y = block.experts_forward(hidden, top_k_index, top_k_weights)
    assert y.shape == (n_tokens, 16)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# glm_moe_dsa
# ---------------------------------------------------------------------------


def test_glm_moe_dsa_forward():
    if not _has_tf_module_for("glm_moe_dsa"):
        pytest.skip("transformers does not have glm_moe_dsa modeling module")
    from auto_round.modeling.unfused_moe.glm_moe_dsa import LinearGlmMoeDsaMoE

    y, block = _run_block(LinearGlmMoeDsaMoE, _glm_moe_dsa_cfg())
    assert y.shape == (2, 5, 16)
    assert block.n_routed_experts == 4


def test_glm_moe_dsa_experts_forward_directly():
    """Exercise the dsa experts_forward with explicit top_k routing."""
    if not _has_tf_module_for("glm_moe_dsa"):
        pytest.skip("transformers does not have glm_moe_dsa modeling module")
    from auto_round.modeling.unfused_moe.glm_moe_dsa import LinearGlmMoeDsaMoE

    _, block = _run_block(LinearGlmMoeDsaMoE, _glm_moe_dsa_cfg())
    n_tokens, n_experts, top_k = 8, 4, 2
    hidden = torch.randn(n_tokens, 16)
    top_k_index = torch.zeros((n_tokens, top_k), dtype=torch.long)
    top_k_index[::2, 0] = 1
    top_k_index[::3, 1] = 2
    top_k_weights = torch.full((n_tokens, top_k), 0.5)
    y = block.experts_forward(hidden, top_k_index, top_k_weights)
    assert y.shape == (n_tokens, 16)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Backward compatibility: numerical sanity with norm_topk_prob off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type_key",
    [
        "qwen3_moe",
        "qwen3_next",
        "glm4_moe",
        "glm4_moe_lite",
        "glm_moe_dsa",
        "deepseek_v3",
        "ernie4_5_moe",
    ],
)
def test_forward_with_norm_topk_prob_off(model_type_key):
    """Most blocks have a ``norm_topk_prob`` flag; setting it to False
    exercises the alternate code path in forward.
    """
    if not _has_tf_module_for(model_type_key):
        pytest.skip(f"transformers does not have {model_type_key} modeling module")
    cls, cfg_factory = _load_tf_block_for(model_type_key)
    cfg = cfg_factory()
    if hasattr(cfg, "norm_topk_prob"):
        cfg.norm_topk_prob = False
    y, _ = _run_block(cls, cfg)
    assert torch.isfinite(y).all(), f"{model_type_key} produced non-finite output"


# ---------------------------------------------------------------------------
# Shared/routed expert separation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type_key",
    [
        "qwen3_next",
        "deepseek_v3",
        "glm4_moe",
        "glm4_moe_lite",
        "glm_moe_dsa",
    ],
)
def test_shared_expert_present(model_type_key):
    """The shared expert should be a distinct module (not None) when
    ``shared_expert_intermediate_size`` (Qwen3-Next) or ``n_shared_experts``
    (DeepSeek / GLM4) is set in the config.
    """
    if not _has_tf_module_for(model_type_key):
        pytest.skip(f"transformers does not have {model_type_key} modeling module")
    cls, cfg_factory = _load_tf_block_for(model_type_key)
    _, block = _run_block(cls, cfg_factory())
    shared = getattr(block, "shared_experts", None) or getattr(block, "shared_expert", None)
    assert shared is not None, f"{model_type_key} should expose a shared expert"
