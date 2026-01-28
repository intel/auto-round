def disable_concat_experts():
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping

    register_checkpoint_conversion_mapping("deepseek_v3", [], overwrite=True)
    register_checkpoint_conversion_mapping("qwen3_moe", [], overwrite=True)


def apply_all_ds_patches():
    from ds_patch import apply_transformer_patches
    # from qwen_v5_patch import apply_transformer_patches_qwen

    disable_concat_experts()
    apply_transformer_patches()

