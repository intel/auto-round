from auto_round.auto_scheme.delta_loss import _apply_head_trick
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme


def test_apply_head_trick_keeps_lowest_loss_candidate_when_budget_allows():
    schemes = AutoScheme(avg_bits=3, options=("GGUF:Q2_K_S", "GGUF:Q4_K_M")).options
    total_scores = {
        "lm_head": [
            [0, 1244659712, 41.0, ["lm_head"]],
            [1, 2489319424, 11.9, ["lm_head"]],
        ],
        "model.layers.0.mlp.down_proj": [
            [0, 100, 10.0, ["model.layers.0.mlp.down_proj"]],
            [1, 200, 5.0, ["model.layers.0.mlp.down_proj"]],
        ],
    }

    _apply_head_trick(
        head_name="lm_head",
        schemes=schemes,
        sorted_indices=[1, 0],
        target_bits=3,
        target_params_cnt=4891670016,
        total_scores=total_scores,
    )

    assert total_scores["lm_head"] == [[1, 2489319424, 11.9, ["lm_head"]]]


def test_apply_head_trick_relaxes_lowest_loss_candidate_when_budget_is_tight():
    schemes = AutoScheme(avg_bits=3, options=("GGUF:Q2_K_S", "GGUF:Q4_K_M")).options
    total_scores = {
        "lm_head": [
            [0, 1244659712, 41.0, ["lm_head"]],
            [1, 2489319424, 11.9, ["lm_head"]],
        ],
        "model.layers.0.mlp.down_proj": [
            [0, 100, 10.0, ["model.layers.0.mlp.down_proj"]],
            [1, 200, 5.0, ["model.layers.0.mlp.down_proj"]],
        ],
    }

    _apply_head_trick(
        head_name="lm_head",
        schemes=schemes,
        sorted_indices=[1, 0],
        target_bits=3,
        target_params_cnt=1244659812,
        total_scores=total_scores,
    )

    assert total_scores["lm_head"] == [
        [0, 1244659712, 41.0, ["lm_head"]],
        [1, 2489319424, 11.9, ["lm_head"]],
    ]


def test_collect_not_fixed_embedding_layer_names_ignores_quant_layer_filter():
    from auto_round.auto_scheme import delta_loss

    embedding_layers_names = ["model.embed_tokens"]
    fixed_layer_scheme = {}
    quant_layer_names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "lm_head",
    ]

    names = delta_loss._get_not_fixed_embedding_layer_names(
        embedding_layers_names=embedding_layers_names,
        fixed_layer_scheme=fixed_layer_scheme,
        quant_layer_names=quant_layer_names,
        include_embeddings=True,
    )

    assert names == ["model.embed_tokens"]


def test_non_gguf_bit_budget_layer_names_exclude_embeddings():
    from auto_round.auto_scheme import delta_loss

    names = delta_loss._get_bit_budget_layer_names(
        quant_layer_names=["model.layers.0.mlp.down_proj"],
        embedding_layers_names=["model.embed_tokens"],
        schemes=AutoScheme(avg_bits=3, options=("W2A16", "W4A16")).options,
    )

    assert names == ["model.layers.0.mlp.down_proj"]


def test_gguf_bit_budget_layer_names_include_embeddings():
    from auto_round.auto_scheme import delta_loss

    names = delta_loss._get_bit_budget_layer_names(
        quant_layer_names=["model.layers.0.mlp.down_proj"],
        embedding_layers_names=["model.embed_tokens"],
        schemes=AutoScheme(avg_bits=3, options=("GGUF:Q2_K_S", "GGUF:Q4_K_M")).options,
    )

    assert names == ["model.layers.0.mlp.down_proj", "model.embed_tokens"]


def test_non_gguf_not_fixed_embedding_layer_names_are_empty():
    from auto_round.auto_scheme import delta_loss

    names = delta_loss._get_not_fixed_embedding_layer_names(
        embedding_layers_names=["model.embed_tokens"],
        fixed_layer_scheme={},
        quant_layer_names=["model.layers.0.mlp.down_proj"],
        include_embeddings=False,
    )

    assert names == []
