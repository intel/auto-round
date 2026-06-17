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
"""Unit tests for DiffusionGemma model changes."""

import types
from typing import Iterable
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from transformers import PretrainedConfig, PreTrainedModel


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_pretrained_submodule(
    name: str,
    params: Iterable[str],
    tied: dict | None = None,
    all_tied: dict | None = None,
    tie_word_embeddings: bool = True,
    parent: nn.Module | None = None,
):
    """Build a minimal ``PreTrainedModel`` submodule with controllable tie state.

    Args:
        name: attribute name on ``parent`` (or local var if ``parent`` is None).
        params: iterable of leaf names — each becomes an ``nn.Linear`` of
            shape ``(out=2, in=2)`` attached directly to the fake model.
            ``named_parameters()`` therefore surfaces them as just the leaf
            name (e.g. ``"weight"``) which matches ``common_case`` regexes.
        tied: value to assign to ``_tied_weights_keys``.
        all_tied: value to assign to ``all_tied_weights_keys``.
        tie_word_embeddings: value for ``config.tie_word_embeddings``.
        parent: if provided, register the module as ``parent.name``.
    """

    class _FakePT(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, cfg):
            super().__init__(cfg)
            self.dummy = nn.Parameter(torch.zeros(1))

        def _init_weights(self, module):
            pass

    cfg = PretrainedConfig(tie_word_embeddings=tie_word_embeddings)
    module = _FakePT(cfg)
    for pname in params:
        # Use bare ``nn.Parameter`` so ``named_parameters`` surfaces the
        # leaf name directly (matches what the production model looks like
        # after MoE unfuse: a ``gate_proj`` / ``up_proj`` weight).
        # ``nn.Module.__setattr__`` registers ``nn.Parameter`` as a parameter
        # when the name contains no "."; nested names would need explicit
        # ``register_parameter`` on the sub-module.
        assert "." not in pname, "test fixture only supports leaf parameter names"
        setattr(module, pname, nn.Parameter(torch.zeros(2, 2)))

    if tied is not None:
        module._tied_weights_keys = tied
    if all_tied is not None:
        module.all_tied_weights_keys = all_tied

    if parent is not None:
        setattr(parent, name, module)
        return parent
    return module


# ---------------------------------------------------------------------------
# 1. Tests for prune_stale_tied_weights_keys
# ---------------------------------------------------------------------------


class TestPruneStaleTiedWeightsKeys:
    """Unit tests for ``prune_stale_tied_weights_keys``."""

    def test_drops_pattern_with_no_matching_target_or_source(self):
        """Tie pattern referencing a name that no longer exists is dropped."""
        from auto_round.utils.model import prune_stale_tied_weights_keys

        wrapper = nn.Module()
        # Only ``gate`` exists; ``gate_up_proj`` was unfused away.
        _make_pretrained_submodule(
            "text",
            params=["gate"],
            tied={"encoder.gate_up_proj": "decoder.gate_up_proj"},
            parent=wrapper,
        )

        removed = prune_stale_tied_weights_keys(wrapper)

        assert removed == 1
        assert wrapper.text._tied_weights_keys == {}

    def test_keeps_pattern_that_resolves_to_real_params(self):
        """Tie patterns whose target/source both resolve to existing parameters are kept."""
        from auto_round.utils.model import prune_stale_tied_weights_keys

        wrapper = nn.Module()
        _make_pretrained_submodule(
            "text",
            params=["src", "dst"],
            tied={"dst": "src"},
            parent=wrapper,
        )

        removed = prune_stale_tied_weights_keys(wrapper)

        assert removed == 0
        assert wrapper.text._tied_weights_keys == {"dst": "src"}

    def test_keeps_common_case_patterns_even_if_unused(self):
        """``*.weight`` / ``*.bias`` style patterns are kept untouched even if no params match."""
        from auto_round.utils.model import prune_stale_tied_weights_keys

        wrapper = nn.Module()
        _make_pretrained_submodule(
            "text",
            params=[],
            tied={"model.embed_tokens.weight": "lm_head.weight"},
            parent=wrapper,
        )

        removed = prune_stale_tied_weights_keys(wrapper)

        assert removed == 0
        assert wrapper.text._tied_weights_keys == {
            "model.embed_tokens.weight": "lm_head.weight"
        }

    def test_keeps_pattern_when_target_source_count_is_multiple_of_source(self):
        """Pattern is kept when target param count is a non-zero multiple of source param count."""
        from auto_round.utils.model import prune_stale_tied_weights_keys

        wrapper = nn.Module()
        # 1 source, 2 targets -> 2 % 1 == 0 -> keep.
        _make_pretrained_submodule(
            "text",
            params=["src", "t0", "t1"],
            tied={r"^t[0-9]$": r"^src$"},
            parent=wrapper,
        )

        removed = prune_stale_tied_weights_keys(wrapper)

        assert removed == 0
        assert r"^t[0-9]$" in wrapper.text._tied_weights_keys

    def test_drops_pattern_when_target_count_not_multiple_of_source(self):
        """Pattern is dropped when target param count is not a multiple of source."""
        from auto_round.utils.model import prune_stale_tied_weights_keys

        wrapper = nn.Module()
        # 2 sources, 3 targets -> 3 % 2 != 0 -> drop.
        _make_pretrained_submodule(
            "text",
            params=["src0", "src1", "t0", "t1", "t2"],
            tied={r"^t[0-9]$": r"^src[0-9]$"},
            parent=wrapper,
        )

        removed = prune_stale_tied_weights_keys(wrapper)

        assert removed == 1
        assert r"^t[0-9]$" not in wrapper.text._tied_weights_keys

    def test_skips_submodule_without_tied_weights_keys(self):
        """Submodules without ``_tied_weights_keys`` are skipped."""
        from auto_round.utils.model import prune_stale_tied_weights_keys

        wrapper = nn.Module()
        _make_pretrained_submodule("text", params=["w"], tied=None, parent=wrapper)
        # PreTrainedModel base class does not define ``_tied_weights_keys`` by default.
        assert getattr(wrapper.text, "_tied_weights_keys", None) is None

        removed = prune_stale_tied_weights_keys(wrapper)

        assert removed == 0

    def test_skips_submodule_when_tie_word_embeddings_false(self):
        """Submodules with ``tie_word_embeddings=False`` are skipped entirely."""
        from auto_round.utils.model import prune_stale_tied_weights_keys

        wrapper = nn.Module()
        _make_pretrained_submodule(
            "text",
            params=[],
            tied={"a": "b"},
            tie_word_embeddings=False,
            parent=wrapper,
        )

        removed = prune_stale_tied_weights_keys(wrapper)

        assert removed == 0
        # Untouched.
        assert wrapper.text._tied_weights_keys == {"a": "b"}

    def test_skips_non_pretrained_submodules(self):
        """Plain ``nn.Module`` children are never inspected."""
        from auto_round.utils.model import prune_stale_tied_weights_keys

        wrapper = nn.Module()
        wrapper.plain = nn.Linear(2, 2)
        wrapper.plain._tied_weights_keys = {"a": "b"}  # would never appear on nn.Module

        removed = prune_stale_tied_weights_keys(wrapper)

        assert removed == 0
        assert wrapper.plain._tied_weights_keys == {"a": "b"}

    def test_prunes_all_tied_weights_keys_cache(self):
        """Stale entries in ``all_tied_weights_keys`` are pruned (target/source both must exist)."""
        from auto_round.utils.model import prune_stale_tied_weights_keys

        wrapper = nn.Module()
        _make_pretrained_submodule(
            "text",
            params=["real"],
            tied=None,
            all_tied={
                "real": "real",
                "stale": "missing",
            },
            parent=wrapper,
        )

        removed = prune_stale_tied_weights_keys(wrapper)

        assert removed == 1
        assert wrapper.text.all_tied_weights_keys == {"real": "real"}

    def test_prunes_stale_entry_even_if_only_target_missing(self):
        """Cached entry whose target no longer exists is dropped."""
        from auto_round.utils.model import prune_stale_tied_weights_keys

        wrapper = nn.Module()
        _make_pretrained_submodule(
            "text",
            params=["keep"],
            tied=None,
            all_tied={
                "missing": "keep",
                "keep": "keep",
            },
            parent=wrapper,
        )

        removed = prune_stale_tied_weights_keys(wrapper)

        assert removed == 1
        assert wrapper.text.all_tied_weights_keys == {"keep": "keep"}

    def test_returns_zero_when_nothing_to_prune(self):
        """No stale entries -> returns 0."""
        from auto_round.utils.model import prune_stale_tied_weights_keys

        wrapper = nn.Module()
        _make_pretrained_submodule("text", params=[], tied=None, parent=wrapper)

        removed = prune_stale_tied_weights_keys(wrapper)

        assert removed == 0

    def test_handles_multiple_pretrained_submodules(self):
        """Multiple ``PreTrainedModel`` submodules are pruned independently."""
        from auto_round.utils.model import prune_stale_tied_weights_keys

        wrapper = nn.Module()
        _make_pretrained_submodule(
            "a",
            params=[],
            tied={"stale_a": "stale_a"},
            parent=wrapper,
        )
        _make_pretrained_submodule(
            "b",
            params=["real"],
            tied={"real": "real"},
            parent=wrapper,
        )

        removed = prune_stale_tied_weights_keys(wrapper)

        assert removed == 1
        assert wrapper.a._tied_weights_keys == {}
        assert wrapper.b._tied_weights_keys == {"real": "real"}


# ---------------------------------------------------------------------------
# 2. Tests for safe_tie_weights
# ---------------------------------------------------------------------------


class TestSafeTieWeights:
    """Unit tests for ``safe_tie_weights``."""

    def test_calls_tie_weights_when_supported(self):
        """Calls ``model.tie_weights()`` when present."""
        from auto_round.utils.model import safe_tie_weights

        called = []

        class _M(nn.Module):
            def tie_weights(self):
                called.append(True)

        m = _M()
        safe_tie_weights(m)
        assert called == [True]

    def test_noop_when_no_tie_weights_method(self):
        """Models without ``tie_weights`` are silently ignored."""
        from auto_round.utils.model import safe_tie_weights

        m = nn.Module()  # no tie_weights attribute
        safe_tie_weights(m)  # must not raise

    def test_drops_stale_entries_before_calling_tie(self):
        """Stale tie patterns are pruned before ``tie_weights`` runs."""
        from auto_round.utils.model import safe_tie_weights

        wrapper = nn.Module()
        _make_pretrained_submodule(
            "text",
            params=["real"],
            tied={"stale": "real"},
            parent=wrapper,
        )
        # tie_weights must be called on the wrapper, not on `text` directly.
        wrapper.tie_weights = lambda: None

        safe_tie_weights(wrapper)

        assert wrapper.text._tied_weights_keys == {}

    def test_swallows_value_error_from_tie_weights(self):
        """``ValueError`` raised by ``tie_weights`` is caught and warned, not raised."""
        from auto_round.utils.model import safe_tie_weights

        class _M(nn.Module):
            def tie_weights(self):
                raise ValueError("simulated tie failure")

        m = _M()
        # Must not raise.
        safe_tie_weights(m)

    def test_non_value_error_is_not_swallowed(self):
        """Other exception types propagate to the caller."""
        from auto_round.utils.model import safe_tie_weights

        class _M(nn.Module):
            def tie_weights(self):
                raise RuntimeError("boom")

        m = _M()
        with pytest.raises(RuntimeError, match="boom"):
            safe_tie_weights(m)


# ---------------------------------------------------------------------------
# 3. Tests for _patch_diffusion_gemma_tied_weights
# ---------------------------------------------------------------------------


class TestPatchDiffusionGemmaTiedWeights:
    """Unit tests for ``_patch_diffusion_gemma_tied_weights``."""

    def test_idempotent_when_called_twice(self):
        """Calling the patcher twice does not re-wrap ``__init__``."""
        from auto_round.utils import common as ar_common

        # Reset any prior patching state.
        ar_common.DiffusionGemmaModel._ar_tied_prune_patched = False
        original = ar_common.DiffusionGemmaModel.__init__

        ar_common._patch_diffusion_gemma_tied_weights()
        patched = ar_common.DiffusionGemmaModel.__init__
        ar_common._patch_diffusion_gemma_tied_weights()
        second = ar_common.DiffusionGemmaModel.__init__

        assert patched is second, "second call should be a no-op"
        # Restore the original to avoid side effects on other tests.
        ar_common.DiffusionGemmaModel.__init__ = original
        ar_common.DiffusionGemmaModel._ar_tied_prune_patched = False

    def test_patched_init_calls_prune(self):
        """Patched ``__init__`` invokes ``prune_stale_tied_weights_keys`` after the original."""
        from auto_round.utils import common as ar_common

        # Ensure a clean slate.
        ar_common.DiffusionGemmaModel._ar_tied_prune_patched = False
        original = ar_common.DiffusionGemmaModel.__init__

        prune_called = []

        def _fake_prune(model):
            prune_called.append(model)

        # The patcher imports ``prune_stale_tied_weights_keys`` lazily from
        # ``auto_round.utils.model``; patch it there.
        import auto_round.utils.model as ar_model

        with patch.object(ar_model, "prune_stale_tied_weights_keys", _fake_prune):
            def _original_init(self, *args, **kwargs):
                pass

            ar_common.DiffusionGemmaModel.__init__ = _original_init
            ar_common._patch_diffusion_gemma_tied_weights()

            sentinel = types.SimpleNamespace()
            ar_common.DiffusionGemmaModel.__init__(sentinel)

        assert len(prune_called) == 1
        assert prune_called[0] is sentinel

        # Restore the original __init__ and clear the patched flag.
        ar_common.DiffusionGemmaModel.__init__ = original
        ar_common.DiffusionGemmaModel._ar_tied_prune_patched = False

    def test_patched_init_logs_warning_on_prune_failure(self, caplog):
        """If ``prune_stale_tied_weights_keys`` raises, the patched init logs a warning."""
        from auto_round.utils import common as ar_common
        from auto_round.logger import logger

        ar_common.DiffusionGemmaModel._ar_tied_prune_patched = False
        original = ar_common.DiffusionGemmaModel.__init__

        def _boom(_model):
            raise RuntimeError("prune exploded")

        def _original_init(self, *args, **kwargs):
            pass

        ar_common.DiffusionGemmaModel.__init__ = _original_init

        # Patch the symbol as imported in ``auto_round.utils.common`` — the
        # patcher references it via module attribute, so we monkey-patch at
        # that location.
        with patch.object(ar_common, "prune_stale_tied_weights_keys", _boom, create=True):
            ar_common._patch_diffusion_gemma_tied_weights()

            caplog.set_level("WARNING", logger=logger.name)
            # Make sure propagation is enabled so caplog can capture records.
            prev_propagate = logger.propagate
            logger.propagate = True
            try:
                sentinel = types.SimpleNamespace()
                # Must not raise.
                ar_common.DiffusionGemmaModel.__init__(sentinel)
            finally:
                logger.propagate = prev_propagate

        assert any("prune_stale_tied_weights_keys" in rec.message for rec in caplog.records)

        # Restore.
        ar_common.DiffusionGemmaModel.__init__ = original
        ar_common.DiffusionGemmaModel._ar_tied_prune_patched = False


# ---------------------------------------------------------------------------
# 4. Tests for skip_not_convert_modules default keyword filter
# ---------------------------------------------------------------------------


class TestSkipNotConvertModulesDefaultKeywords:
    """When the user did not specify ``modules_to_not_convert``, the default
    skip set is filtered to the standard keywords: embed / embed_tokens /
    lm_head / output_embed / norm."""

    @staticmethod
    def _call(model, user_modules, mock_default):
        """Invoke ``skip_not_convert_modules`` with a mocked transformers default-skip helper."""
        from auto_round.inference import convert_model

        cfg = types.SimpleNamespace(modules_to_not_convert=user_modules)
        layer_names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        layer_configs = {n: {"bits": 4} for n in layer_names}
        with patch.object(convert_model, "get_modules_to_not_convert", return_value=mock_default):
            return convert_model.skip_not_convert_modules(model, cfg, layer_names, layer_configs)

    def test_user_unspecified_keeps_only_default_keywords(self):
        """Default-skip keywords are kept; everything else is filtered out."""
        from auto_round.inference import convert_model

        model = nn.Module()
        result = self._call(
            model,
            user_modules=None,
            mock_default=["model.embed_tokens", "lm_head", "model.norm"],
        )
        # Only default-keyword layers were marked as 16-bit; MLP layers unchanged.
        assert result["model.embed_tokens.weight"]["bits"] == 16
        assert result["lm_head.weight"]["bits"] == 16
        assert result["model.norm.weight"]["bits"] == 16
        assert result["model.layers.0.self_attn.q_proj.weight"]["bits"] == 4
        assert result["model.layers.0.mlp.gate_up_proj.weight"]["bits"] == 4

    def test_user_specified_passes_full_list_through(self):
        """When the user explicitly listed modules, no default filtering is applied."""
        from auto_round.inference import convert_model

        model = nn.Module()
        result = self._call(
            model,
            user_modules=["model.layers.0.mlp"],
            mock_default=["model.embed_tokens", "model.layers.0.mlp"],
        )
        # Both default and user-supplied keywords reach the matcher.
        assert result["model.embed_tokens.weight"]["bits"] == 16
        assert result["model.layers.0.self_attn.q_proj.weight"]["bits"] == 4
        # `model.layers.0.mlp` matches `model.layers.0.mlp.gate_up_proj.weight`.
        assert result["model.layers.0.mlp.gate_up_proj.weight"]["bits"] == 16

    def test_empty_everything_no_op(self):
        """No skip patterns -> no layer is touched."""
        from auto_round.inference import convert_model

        model = nn.Module()
        result = self._call(model, user_modules=None, mock_default=[])
        for name, cfg in result.items():
            assert cfg["bits"] == 4, f"{name} unexpectedly set to 16-bit"

    def test_filter_keeps_keywords_embeds_norm_lm_head_output_embed(self):
        """Each of the five default keywords is preserved by the filter."""
        from auto_round.inference import convert_model

        model = nn.Module()
        result = self._call(
            model,
            user_modules=None,
            mock_default=[
                "model.embed_tokens",
                "model.embed",
                "lm_head",
                "output_embed",
                "model.norm",
                "model.layers.0.self_attn",  # should be filtered out
                "model.layers.0.mlp",  # should be filtered out
            ],
        )
        # The two "model.layers.*" entries should have been filtered out.
        assert result["model.layers.0.self_attn.q_proj.weight"]["bits"] == 4
        assert result["model.layers.0.mlp.gate_up_proj.weight"]["bits"] == 4
        # The five default keywords should still apply.
        assert result["model.embed_tokens.weight"]["bits"] == 16
        assert result["model.norm.weight"]["bits"] == 16
        assert result["lm_head.weight"]["bits"] == 16


# ---------------------------------------------------------------------------
# 5. Tests for _PRE_DEFINED_FIXED_ATTR registration
# ---------------------------------------------------------------------------


class TestPredefinedFixedAttrDiffusionGemma:
    """``_PRE_DEFINED_FIXED_ATTR`` exposes DiffusionGemma-specific fixed attrs."""

    def test_diffusion_gemma_registered(self):
        from auto_round.special_model_handler import (
            _PRE_DEFINED_FIXED_ATTR,
            get_predefined_fixed_attr,
        )

        assert "diffusion_gemma" in _PRE_DEFINED_FIXED_ATTR
        assert _PRE_DEFINED_FIXED_ATTR["diffusion_gemma"] == {"has_variable_block_shape": True}

    def test_get_predefined_fixed_attr_returns_diffusion_gemma_attrs(self):
        from auto_round.special_model_handler import get_predefined_fixed_attr

        model = nn.Module()
        model.config = types.SimpleNamespace(model_type="diffusion_gemma")
        assert get_predefined_fixed_attr(model) == {"has_variable_block_shape": True}

    def test_get_predefined_fixed_attr_returns_none_for_unknown_model(self):
        from auto_round.special_model_handler import get_predefined_fixed_attr

        model = nn.Module()
        model.config = types.SimpleNamespace(model_type="llama")
        assert get_predefined_fixed_attr(model) is None

    def test_get_predefined_fixed_attr_returns_none_when_no_config(self):
        from auto_round.special_model_handler import get_predefined_fixed_attr

        assert get_predefined_fixed_attr(nn.Module()) is None
