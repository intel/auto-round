"""Smoke tests for the in-tree Nemotron-H MoE handler.

These tests do not download the full Nemotron-H checkpoint — they only
verify that the in-tree handler is registered correctly and that
``apply_model_monkey_patches`` rewires both the class symbol and the
``MIXER_TYPES`` dispatch dict.

If ``transformers.models.nemotron_h`` is unavailable in the runtime
environment (older transformers without Nemotron-H support), the tests
are skipped automatically.
"""

from __future__ import annotations

import pytest


def test_nemotron_h_registered_in_model_config():
    """The handler must appear under model_type ``nemotron_h`` and ship
    both a ``block_patch`` (for setattr-style replacement) and a
    ``dispatch_dict_patch`` entry for ``MIXER_TYPES``."""

    from auto_round.modeling.unfused_moe import MODEL_CONFIG

    assert "nemotron_h" in MODEL_CONFIG, "nemotron_h not registered in MODEL_CONFIG"
    cfg = MODEL_CONFIG["nemotron_h"]
    assert cfg.get("block_patch"), "nemotron_h missing block_patch"
    assert cfg.get("dispatch_dict_patch"), (
        "nemotron_h missing dispatch_dict_patch — without it the MIXER_TYPES dict "
        "stays pointing at the original class and quantization sees zero experts."
    )


def test_nemotron_h_class_is_importable():
    """Smoke test: the class definition must import even if transformers
    has not been initialised with Nemotron-H yet (the handler module
    must not have a top-level dependency on ``transformers.models.nemotron_h``)."""

    pytest.importorskip("transformers")
    from auto_round.modeling.unfused_moe.nemotron_h import LinearNemotronHMoE

    assert LinearNemotronHMoE is not None


def test_nemotron_h_preserves_prefix_rename_drops_expert_bundles_and_embedding_alias():
    """The conversion-mapping override MUST keep the upstream
    ``backbone.→model.`` rename (load without it leaves the backbone at
    random init) but MUST drop both the expert-bundling
    ``WeightConverter`` entries and the legacy
    ``embedding.weight→embeddings.weight`` rename.

    Dropping the embedding rename is required because the current on-disk
    checkpoint already stores the tensor as ``backbone.embeddings.weight``
    (plural): the rename is a no-op on load but is applied in reverse at
    save time — producing ``backbone.embedding.weight`` (singular) on disk.
    That breaks name matching in
    ``copy_missing_tensors_from_source`` which then RTN-quantizes the
    source embedding and writes a phantom
    ``backbone.embeddings.qweight`` tensor alongside the unquantized
    BF16 copy. Keeping load and save symmetrically plural prevents the
    double-store.

    Regression: returning an empty mapping silently dropped every
    upstream rename, leaving the entire backbone (router gate,
    e_score_correction_bias, attention, Mamba2, MLP) at random init —
    quantization then RTN-quantized noise.
    """

    pytest.importorskip("transformers.models.nemotron_h")
    from transformers import conversion_mapping

    from auto_round.modeling.unfused_moe import (
        MODEL_CONFIG,
        get_checkpoint_conversion_mapping_ar,
    )

    cfg = MODEL_CONFIG["nemotron_h"]
    assert cfg.get("preserve_upstream_conversion_mapping") is True, (
        "nemotron_h must preserve the upstream conversion mapping " "(backbone.→model.) — see regression note."
    )
    drop_targets = cfg.get("drop_conversion_target_patterns", [])
    assert "mixer.experts.up_proj" in drop_targets
    assert "mixer.experts.down_proj" in drop_targets
    assert "embeddings.weight" in drop_targets, (
        "The legacy embedding.weight→embeddings.weight rename must be dropped "
        "to keep save symmetric with the on-disk plural ``backbone.embeddings.weight``."
    )

    upstream = conversion_mapping.get_checkpoint_conversion_mapping("nemotron_h")
    upstream_targets = {tuple(getattr(rule, "target_patterns", []) or []) for rule in upstream}

    # Sanity: the upstream mapping itself contains the rules we depend on.
    assert ("model.",) in upstream_targets, (
        "Upstream nemotron_h mapping no longer contains the backbone.→model. rename — "
        "the AutoRound filter must be re-validated against the current transformers version."
    )

    ar_mapping = get_checkpoint_conversion_mapping_ar("nemotron_h")
    ar_targets = {tuple(getattr(rule, "target_patterns", []) or []) for rule in ar_mapping}

    # Required rename preserved.
    assert ("model.",) in ar_targets, "AR override dropped the backbone.→model. rename"

    # Expert-bundling converters AND the embedding singular/plural rename
    # must all be filtered out.
    assert ("mixer.experts.up_proj",) not in ar_targets
    assert ("mixer.experts.down_proj",) not in ar_targets
    assert ("embeddings.weight",) not in ar_targets, (
        "AR override must drop the embedding.weight→embeddings.weight rename so "
        "save does not reverse-rename plural in-memory names to singular on disk."
    )


def test_apply_model_monkey_patches_rewires_nemotron_h():
    """Full registration round-trip: applying the patch must replace
    both ``modeling_nemotron_h.NemotronHMoE`` and
    ``modeling_nemotron_h.MIXER_TYPES['moe']``.

    Skipped if the local ``transformers`` does not yet ship Nemotron-H."""

    pytest.importorskip("transformers.models.nemotron_h")

    from transformers.models.nemotron_h import modeling_nemotron_h

    from auto_round.modeling.unfused_moe import apply_model_monkey_patches
    from auto_round.modeling.unfused_moe.nemotron_h import LinearNemotronHMoE

    # Snapshot originals so we can restore — keeps the test idempotent.
    orig_class = modeling_nemotron_h.NemotronHMoE
    orig_mixer = modeling_nemotron_h.MIXER_TYPES.get("moe")
    try:
        applied = apply_model_monkey_patches("nvidia/Nemotron-Cascade-2-30B-A3B", trust_remote_code=False)
        assert applied is True, "apply_model_monkey_patches returned False for nemotron_h"

        assert modeling_nemotron_h.NemotronHMoE is LinearNemotronHMoE
        assert modeling_nemotron_h.MIXER_TYPES["moe"] is LinearNemotronHMoE
    finally:
        modeling_nemotron_h.NemotronHMoE = orig_class
        if orig_mixer is not None:
            modeling_nemotron_h.MIXER_TYPES["moe"] = orig_mixer
