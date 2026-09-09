from transformers import AutoConfig, AutoTokenizer


def test_tiny_model_cache_rejects_a_changed_model_config(tmp_path):
    from test import fixtures

    signature = fixtures._tiny_model_signature(("model", str(tmp_path)), {"num_layers": 2})
    config_path = tmp_path / "config.json"
    config_path.write_text('{"hidden_size": 64}')
    fixtures._mark_tiny_model(tmp_path, signature)

    config_path.write_text('{"hidden_size": 256}')

    assert not fixtures._tiny_model_ready(tmp_path, signature)


def test_tiny_model_cache_does_not_manage_huggingface_source_cache():
    """Tiny fixture reuse must not delete source checkpoints between pytest runs."""
    from test import fixtures

    assert not hasattr(fixtures, "_source_model_ids")
    assert not hasattr(fixtures, "_release_source_model_cache")


def test_micro_model_tokenizer_ids_fit_the_model_vocabulary(micro_opt_model_path, micro_qwen_model_path):
    for model_path in (micro_opt_model_path, micro_qwen_model_path):
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer("auto round calibration sample", return_tensors="pt")["input_ids"]
        special_token_ids = [
            token_id
            for token_id in (config.bos_token_id, config.eos_token_id, config.pad_token_id)
            if token_id is not None
        ]

        assert input_ids.max().item() < config.vocab_size
        assert all(token_id < config.vocab_size for token_id in special_token_ids)


def test_micro_qwen_preserves_embedding_dominated_parameter_budget(micro_qwen_model_path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(micro_qwen_model_path)
    weights = {
        name: module.weight.numel()
        for name, module in model.named_modules()
        if getattr(module, "weight", None) is not None and not list(module.children())
    }

    assert (weights["model.embed_tokens"] + weights["lm_head"]) / sum(weights.values()) > 0.9


def test_micro_model_tokenizer_keeps_short_calibration_samples_valid(micro_opt_model_path, micro_qwen_model_path):
    text = "auto round calibration sample keeps each token distinct for scoring"
    for model_path in (micro_opt_model_path, micro_qwen_model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(text)["input_ids"]

        assert len(input_ids) >= 8
        assert input_ids.count(input_ids[-1]) <= 4


def test_micro_qwen_moe_preserves_expert_module_structure(micro_qwen_moe_model_path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(micro_qwen_moe_model_path)
    moe = model.model.layers[0].mlp

    assert model.config.num_experts == 4
    assert hasattr(moe, "experts")
    assert hasattr(moe, "shared_expert")
    assert hasattr(moe.shared_expert, "up_proj")
