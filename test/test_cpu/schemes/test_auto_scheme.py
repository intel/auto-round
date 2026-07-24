import shutil

import pytest
from transformers import AutoRoundConfig

from auto_round import AutoRound, AutoScheme


class TestAutoScheme:
    @pytest.fixture(autouse=True)
    def setup_save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    def test_auto_scheme_export(self, tiny_opt_model_path):
        model_name = tiny_opt_model_path
        scheme = AutoScheme(avg_bits=2, options=("W2A16"), nsamples=1, ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        ar.quantize_and_save(self.save_dir)

        scheme = AutoScheme(avg_bits=4, options=("mxfp4"), nsamples=1, ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        ar.quantize_and_save(self.save_dir)

    def test_gguf_user_fixed_embedding_budget(self, tiny_qwen_model_path):
        """Regression test: a user-fixed embedding must be budget-priced at its fixed bits.

        apply_quant_scheme only covers quant_layer_names (embeddings are carved out),
        so the fixed-layer budget subtraction read bits=16 off the bare module and
        priced the embedding at 16 bits, making low targets infeasible (DP returned
        None -> TypeError, or 'Avg bits is too small').
        """
        target_bits = 3.0
        scheme = AutoScheme(
            avg_bits=target_bits,
            options=("gguf:q2_k_s", "gguf:q4_k_s", "gguf:q6_k"),
            nsamples=1,
            ignore_scale_zp_bits=True,
        )
        user_layer_config = {
            "model.embed_tokens": {
                "bits": 3,
                "super_bits": 6,
                "super_group_size": 16,
                "group_size": 16,
                "sym": True,
                "data_type": "int_sym_dq",
            }
        }
        ar = AutoRound(
            model=tiny_qwen_model_path,
            scheme=scheme,
            format="gguf:q2_k_s",
            iters=0,
            nsamples=1,
            seqlen=32,
            layer_config=user_layer_config,
        )
        weight_numels = {
            n: m.weight.numel()
            for n, m in ar.model.named_modules()
            if getattr(m, "weight", None) is not None and len(list(m.children())) == 0
        }
        model, layer_config = ar.quantize()
        assert layer_config["model.embed_tokens"]["bits"] == 3
        quant_layers = [n for n in layer_config if n in weight_numels]
        total_params = sum(weight_numels[n] for n in quant_layers)
        total_bits = sum(layer_config[n].get("bits", 16) * weight_numels[n] for n in quant_layers)
        avg_bits = total_bits / total_params
        assert avg_bits <= target_bits + 0.05

    def test_gguf_embedding_in_budget(self, tiny_qwen_model_path):
        """Regression test: the (tied) embedding must be charged against the avg_bits budget.

        On tiny Qwen the embedding holds >90% of the params. Before the fix it was
        silently dropped from the AutoScheme budget (dead `in quant_layer_names`
        check) and later filled with the gguf lm_head default (q6_k for tied
        embeddings), so the effective avg_bits landed near 6 instead of the target.
        """
        target_bits = 3.0
        scheme = AutoScheme(
            avg_bits=target_bits,
            options=("gguf:q2_k_s", "gguf:q4_k_s", "gguf:q6_k"),
            nsamples=1,
            ignore_scale_zp_bits=True,
        )
        ar = AutoRound(model=tiny_qwen_model_path, scheme=scheme, format="gguf:q2_k_s", iters=0, nsamples=1, seqlen=32)
        # Snapshot parameter counts before quantization: gguf packing releases
        # weights (module.weight = None) as blocks are packed, so numel is not
        # available on the model afterwards.
        weight_numels = {
            n: m.weight.numel()
            for n, m in ar.model.named_modules()
            if getattr(m, "weight", None) is not None and len(list(m.children())) == 0
        }
        model, layer_config = ar.quantize()
        # Only q2_k_s fits the budget for the embedding (q4_k_s/q6_k alone would exceed it).
        assert layer_config["model.embed_tokens"]["bits"] == 2
        quant_layers = [n for n in layer_config if n in weight_numels]
        total_params = sum(weight_numels[n] for n in quant_layers)
        total_bits = sum(layer_config[n].get("bits", 16) * weight_numels[n] for n in quant_layers)
        avg_bits = total_bits / total_params
        assert avg_bits <= target_bits + 0.05

    def test_layer_config(self, tiny_opt_model_path):
        from auto_round.auto_scheme.utils import compute_avg_bits_for_model
        from auto_round.utils import get_module

        target_bits = 3.5
        model_name = tiny_opt_model_path
        scheme = AutoScheme(avg_bits=target_bits, options=("W2A16", "W4A16", "BF16"))
        user_layer_config = {"model.decoder.layers.1.fc1": {"bits": 8, "group_size": 32, "sym": False}}
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1, layer_config=user_layer_config)
        model, layer_config = ar.quantize()
        assert layer_config["model.decoder.layers.1.fc1"]["bits"] == 8
        assert layer_config["model.decoder.layers.1.fc1"]["sym"] is False
        assert layer_config["model.decoder.layers.1.fc1"]["group_size"] == 32
        layer = get_module(model, "model.decoder.layers.1.fc1")
        assert layer.bits == 8
        assert layer.sym is False
        assert layer.group_size == 32
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3
