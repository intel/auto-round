import shutil

from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ....helpers import get_model_path


class TestGGUFQ4KM:

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_q4_k_m(self, dataloader):
        model_name = get_model_path("Qwen/Qwen2.5-1.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        layer_config = {
            "lm_head": {
                "bits": 4,
                "group_size": 32,
                "sym": False,
                "data_type": "int_asym_dq",
                "super_bits": 6,
                "super_group_size": 8,
            },
            "model.embed_tokens": {"bits": 6, "group_size": 32, "super_bits": 6, "super_group_size": 8},
            "model.layers.12.mlp.gate_proj": {"bits": 3},
            "model.layers.10.mlp.gate_proj": {"bits": 8},
        }
        autoround = AutoRound(
            model,
            tokenizer,
            layer_config=layer_config,
            iters=0,
            seqlen=1,
            nsamples=8,
            dataset=dataloader,
            disable_opt_rtn=True,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_k_m,fake")
        assert autoround.layer_config["model.layers.11.self_attn.v_proj"]["super_group_size"] == 16
        assert autoround.layer_config["model.layers.11.self_attn.v_proj"]["data_type"] == "int_sym_dq"
        assert autoround.layer_config["model.layers.7.self_attn.v_proj"]["data_type"] == "int_asym_dq"
        assert autoround.model.model.layers[0].self_attn.v_proj.bits == 6
        assert autoround.model.model.layers[12].self_attn.v_proj.bits == 4
        assert autoround.model.model.embed_tokens.bits == 6
        assert autoround.model.model.embed_tokens.group_size == 16
        assert autoround.model.model.layers[12].mlp.gate_proj.bits == 3
        assert autoround.model.model.layers[10].mlp.gate_proj.bits == 8
        assert autoround.layer_config["model.layers.10.mlp.gate_proj"]["mostly"] == "gguf:q8_0"
        shutil.rmtree("./saved", ignore_errors=True)

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(model, tokenizer, iters=0, nsamples=1, seqlen=128, disable_opt_rtn=False)
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_k_m,fake")
        shutil.rmtree("./saved", ignore_errors=True)
