import copy
import re
import shutil

import pytest
import transformers
from transformers import AutoRoundConfig

from auto_round import AutoRound, AutoScheme
from auto_round.auto_scheme.utils import compute_avg_bits_for_model
from auto_round.utils import get_module

from ...envs import multi_card
from ...helpers import evaluate_accuracy, get_model_path, get_tiny_model


class TestAutoScheme:

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        yield
        shutil.rmtree("runs", ignore_errors=True)

    def test_gguf_k_0(self, tiny_qwen_model_path):
        target_bits = 5.5
        scheme = AutoScheme(avg_bits=target_bits, options=("GGUF:Q4_K_M", "GGUF:Q8_0"))
        ar = AutoRound(model=tiny_qwen_model_path, scheme=scheme, iters=1, enable_alg_ext=True)
        ar.quantize_and_save(self.save_dir, format="gguf:q2_k_s")

    @pytest.mark.skip_ci(reason="not necessary to test all options")
    def test_auto_scheme_export_gguf(self, tiny_qwen_model_path):
        scheme = AutoScheme(avg_bits=3, options=("gguf:q2_k_s,gguf:q4_k_s"), nsamples=1, ignore_scale_zp_bits=True)
        ar = AutoRound(model=tiny_qwen_model_path, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1)
        ar.quantize()

    @pytest.mark.skip_ci(reason="not necessary to test all options")
    def test_gguf_k_1(self, tiny_qwen_model_path):
        target_bits = 3.5
        scheme = AutoScheme(avg_bits=target_bits, options=("GGUF:Q2_K_S", "GGUF:Q4_1"))
        ar = AutoRound(model=tiny_qwen_model_path, scheme=scheme, iters=1, enable_alg_ext=True)
        ar.quantize_and_save(self.save_dir, format="gguf:q2_k_s")

    @pytest.mark.skip_ci(reason="not necessary to test all options")
    def test_embedding_fallback(self, tiny_qwen_model_path):
        target_bits = 5.0
        scheme = AutoScheme(avg_bits=target_bits, options=("GGUF:Q4_K_M", "GGUF:Q8_0"))
        ar = AutoRound(model=tiny_qwen_model_path, scheme=scheme, iters=1, enable_alg_ext=True)
        ar.quantize_and_save(self.save_dir, format="gguf:q2_k_s")

    @pytest.mark.skip_ci(reason="not necessary to test all options")
    def test_gguf_export(self, tiny_qwen_model_path):
        target_bits = 3
        scheme = AutoScheme(avg_bits=target_bits, options=("GGUF:Q2_K_S", "GGUF:Q4_K_M"), ignore_scale_zp_bits=True)
        ar = AutoRound(model=tiny_qwen_model_path, scheme=scheme, iters=0)
        ar.quantize_and_save(self.save_dir, format="gguf:q2_k_s")

    @pytest.mark.skip_ci(reason="not necessary to test all options")
    def test_gguf(self):
        model_name = get_model_path("Qwen/Qwen3-8B")
        model = get_tiny_model(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        target_bits = 3
        scheme = AutoScheme(avg_bits=target_bits, options=("GGUF:Q2_K_S", "GGUF:Q4_K_M"), ignore_scale_zp_bits=True)
        ar = AutoRound(model=model, tokenizer=tokenizer, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1)
        model, layer_config = ar.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model, ignore_scale_zp_bits=True)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    def test_shared_layers(self, tiny_opt_model_path):
        model_name = tiny_opt_model_path
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_name)
        shared_layers = [
            ["*.self_attn.k_proj", "v_proj", "q_proj", "out_proj"],
            ("model.decoder.layers.6.fc1", "model.decoder.layers.6.fc2"),
            ("fc1", "fc2"),
        ]
        from auto_round.auto_scheme.utils import parse_shared_layers

        res = parse_shared_layers(model, shared_layers)
        assert len(res) == 4
        assert [
            "model.decoder.layers.1.self_attn.out_proj",
            "model.decoder.layers.1.self_attn.q_proj",
            "model.decoder.layers.1.self_attn.v_proj",
        ] in res
        assert ["model.decoder.layers.0.fc1", "model.decoder.layers.0.fc2"] in res
        assert ["model.decoder.layers.1.fc1", "model.decoder.layers.1.fc2"] in res
        target_bits = 5.0
        scheme = AutoScheme(avg_bits=target_bits, options=("W4A16", "MXFP8"), shared_layers=shared_layers)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1)
        model, layer_config = ar.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model)
        for names in res:
            bits = []
            for name in names:
                module = get_module(model, name)
                if hasattr(module, "orig_layer"):
                    bits.append(module.orig_layer.bits)
                else:
                    bits.append(module.bits)
            bits = set(bits)
            assert len(bits) == 1
        print(avg_bits)
        assert target_bits - 0.2 < avg_bits <= target_bits + 1e-3

    @pytest.mark.skip_ci(reason="multiple card test")
    @multi_card
    def test_multi_card(self, tiny_qwen_model_path):
        target_bits = 4.5
        for device_map in ["auto", "0,1", "0", None]:
            scheme = AutoScheme(avg_bits=target_bits, options=("NVFP4"))
            ar = AutoRound(
                model=tiny_qwen_model_path,
                scheme=scheme,
                iters=0,
                disable_opt_rtn=True,
                nsamples=1,
                device_map=device_map,
            )
            model, layer_config = ar.quantize()
            avg_bits, _ = compute_avg_bits_for_model(model)
            print(avg_bits)
            assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    @pytest.mark.skip_ci(reason="multiple card test")
    @multi_card
    def test_multi_card_1(self, tiny_qwen_model_path):
        target_bits = 4.5
        from transformers import AutoModelForCausalLM, AutoTokenizer

        scheme = AutoScheme(avg_bits=target_bits, options=("NVFP4"))
        ar = AutoRound(model=tiny_qwen_model_path, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1)
        model, layer_config = ar.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    def test_non_low_gpu_mem_usage(self, tiny_qwen_model_path):
        target_bits = 4.5
        # for device_map in ["auto", "0,1", "0", None]:
        scheme = AutoScheme(avg_bits=target_bits, options=("NVFP4"), low_gpu_mem_usage=False, device_map="auto")

        ar = AutoRound(model=tiny_qwen_model_path, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1)
        model, layer_config = ar.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    @pytest.mark.skip_ci(reason="multiple card test")
    @multi_card
    def test_dict_device_map(self, tiny_qwen_model_path):
        target_bits = 8.25
        device_map = {"up_proj": 0, "down_proj": 1}

        scheme = AutoScheme(avg_bits=target_bits, options=("MXFP8"))
        ar = AutoRound(
            model=tiny_qwen_model_path, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1, device_map=device_map
        )
        model, layer_config = ar.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    @pytest.mark.skip_ci(reason="Not necessary to test the corner case in CI")
    def test_min_target_bits(self, tiny_opt_model_path):
        target_bits = 4.644
        scheme = AutoScheme(avg_bits=target_bits, options=("MXFP4", "W8A16"))
        ar = AutoRound(model=tiny_opt_model_path, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1)
        model, layer_config = ar.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    def test_max_target_bits(self):
        target_bits = 8.025
        model_path = get_model_path("facebook/opt-125m")
        scheme = AutoScheme(avg_bits=target_bits, options=("MXFP4", "W8A16"))
        ar = AutoRound(model=model_path, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1)
        model, layer_config = ar.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    @pytest.mark.skip_ci(reason="Not necessary to test the corner case in CI")
    def test_patch_scheme(self, tiny_opt_model_path):
        target_bits = 5
        scheme = AutoScheme(avg_bits=target_bits, options=("MXFP4", "W8A16"))
        ar = AutoRound(
            model=tiny_opt_model_path, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1, group_size=32
        )
        model, layer_config = ar.quantize()
        for n, m in model.named_modules():
            if hasattr(m, "group_size"):
                assert m.group_size == 32
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    def test_layer_config(self):
        target_bits = 3.0
        model_name = get_model_path("facebook/opt-125m")
        scheme = AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "BF16"))
        user_layer_config = {"model.decoder.layers.10.fc1": {"bits": 8, "group_size": 32, "sym": False}}
        ar = AutoRound(
            model=model_name, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1, layer_config=user_layer_config
        )
        model, layer_config = ar.quantize()
        assert layer_config["model.decoder.layers.10.fc1"]["bits"] == 8
        assert layer_config["model.decoder.layers.10.fc1"]["sym"] is False
        assert layer_config["model.decoder.layers.10.fc1"]["group_size"] == 32
        layer = get_module(model, "model.decoder.layers.10.fc1")
        assert layer.bits == 8
        assert layer.sym is False
        assert layer.group_size == 32
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

        target_bits = 5.5
        scheme = AutoScheme(avg_bits=target_bits, options=("mxfp4", "mxfp8"))
        user_layer_config = {"model.decoder.layers.10.fc1": {"bits": 8, "group_size": 32, "sym": False}}
        ar = AutoRound(
            model=model_name, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1, layer_config=user_layer_config
        )
        model, layer_config = ar.quantize()
        assert layer_config["model.decoder.layers.10.fc1"]["bits"] == 8
        assert layer_config["model.decoder.layers.10.fc1"]["sym"] is False
        assert layer_config["model.decoder.layers.10.fc1"]["group_size"] == 32
        layer = get_module(model, "model.decoder.layers.10.fc1")
        assert layer.orig_layer.bits == 8
        assert layer.orig_layer.sym is False
        assert layer.orig_layer.group_size == 32
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    def test_lm_head_and_mix_dtype(self, tiny_untied_qwen_model_path):
        target_bits = 5
        scheme = AutoScheme(avg_bits=target_bits, options=("MXFP4", "MXFP8"))
        ar = AutoRound(
            tiny_untied_qwen_model_path, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1, quant_lm_head=True
        )
        model, layer_config = ar.quantize()
        assert layer_config["lm_head"]["bits"] <= 8
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        # cause using tiny model
        assert 4.8 < avg_bits <= target_bits + 1e-3

    @pytest.mark.skip_ci(reason="The evaluation is time-consuming")
    def test_auto_scheme_export(self):
        model_name = get_model_path("facebook/opt-125m")
        scheme = AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "W8A16", "BF16"))
        ar = AutoRound(model=model_name, scheme=scheme)
        ar.quantize_and_save(output_dir=self.save_dir)
        evaluate_accuracy(self.save_dir, threshold=0.25)

    @pytest.mark.skip_ci(reason="The evaluation is time-consuming")
    def test_enable_torch_compile(self):
        model_name = get_model_path("facebook/opt-125m")
        scheme = AutoScheme(avg_bits=2, options=("W2A16"), ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, enable_torch_compile=True)
        ar.quantize_and_save(output_dir=self.save_dir)
        evaluate_accuracy(self.save_dir, threshold=0.10)

    def test_mixed_bits_get_scoring(self):
        """Verify that AutoScheme scoring with low_gpu_mem_usage=False produces
        comparable accuracy to the low_gpu_mem_usage=True baseline for mixed-bit
        quantization.
        """
        target_bits = 2.5
        common_kwargs = dict(
            iters=0, disable_opt_rtn=True, nsamples=2, seqlen=16,
        )
        model_name = get_model_path("facebook/opt-125m")
        scheme_baseline = AutoScheme(
            avg_bits=target_bits, options="W2A16,W3A16",
            ignore_scale_zp_bits=True, low_gpu_mem_usage=True, # default setting
        )
        ar_baseline = AutoRound(
            model=model_name, scheme=scheme_baseline, **common_kwargs,
        )
        model_baseline, _ = ar_baseline.quantize()
        acc_baseline = evaluate_accuracy(
            model_baseline, ar_baseline.tokenizer, task="piqa", limit=200
        )

        # Run with low_gpu_mem_usage=False
        scheme_test = AutoScheme(
            avg_bits=target_bits, options="W2A16,W3A16",
            ignore_scale_zp_bits=True, low_gpu_mem_usage=False,
        )
        ar_test = AutoRound(
            model=model_name, scheme=scheme_test, **common_kwargs,
        )
        model_test, _ = ar_test.quantize()
        acc_test = evaluate_accuracy(
            model_test, ar_test.tokenizer, task="piqa", limit=200
        )

        # Accuracy gap should be small
        gap = abs(acc_baseline - acc_test)
        print(f"acc_baseline={acc_baseline:.4f}, acc_test={acc_test:.4f}, gap={gap:.4f}")
        assert gap < 0.01, (
            f"Accuracy gap {gap:.4f} between low_gpu_mem modes is too large "
            f"(baseline={acc_baseline:.4f}, test={acc_test:.4f})"
        )
