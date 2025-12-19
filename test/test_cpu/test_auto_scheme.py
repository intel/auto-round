import shutil

import pytest

from auto_round import AutoRound, AutoRoundConfig, AutoScheme


class TestAutoScheme:
    save_dir = "./saved"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_auto_scheme_export(self, tiny_opt_model_path):
        model_name = tiny_opt_model_path
        scheme = AutoScheme(avg_bits=2, options=("W2A16"), nsamples=1, ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        ar.quantize_and_save(self.save_dir)
        shutil.rmtree(self.save_dir, ignore_errors=True)

        scheme = AutoScheme(avg_bits=4, options=("mxfp4"), nsamples=1, ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        ar.quantize_and_save(self.save_dir)
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_layer_config(self, tiny_opt_model_path):
        from auto_round.auto_scheme.utils import compute_avg_bits_for_model
        from auto_round.utils import get_module

        target_bits = 3.0
        model_name = tiny_opt_model_path
        scheme = AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "BF16"))
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
