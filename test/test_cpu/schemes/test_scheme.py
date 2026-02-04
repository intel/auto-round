import shutil

import transformers

from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme

from ...helpers import get_model_path, get_tiny_model, opt_name_or_path, qwen_name_or_path


class TestAutoRound:
    @classmethod
    def setup_class(self):
        self.save_folder = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_gguf(self, tiny_qwen_model_path, dataloader):
        ar = AutoRound(
            tiny_qwen_model_path,
            scheme="W2A16",
            nsamples=1,
            iters=1,
            seqlen=2,
            dataset=dataloader,
        )
        ar.quantize_and_save(self.save_folder, format="gguf:q4_k_m")
        assert ar.bits == 4
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_w4a16(self, tiny_opt_model_path, dataloader):
        ar = AutoRound(tiny_opt_model_path, scheme="W4A16", nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        assert ar.bits == 4
        ar.quantize()

    def test_w2a16_rtn(self, tiny_opt_model_path, dataloader):
        ar = AutoRound(tiny_opt_model_path, scheme="W2A16", nsamples=1, iters=0, seqlen=2, dataset=dataloader)
        assert ar.bits == 2
        ar.quantize()

    def test_w4a16_mixed(self, tiny_qwen_moe_model_path, dataloader):

        layer_config = {
            "model.layers.0.self_attn.k_proj": {"bits": 16},
        }
        ar = AutoRound(
            tiny_qwen_moe_model_path,
            scheme="W4A16_MIXED",
            nsamples=1,
            iters=0,
            seqlen=2,
            dataset=dataloader,
            low_cpu_mem_usage=False,
            layer_config=layer_config,
        )
        ar.quantize()
        assert ar.bits == 4
        assert ar.model.model.layers[0].self_attn.q_proj.bits == 8
        assert ar.model.model.layers[0].self_attn.k_proj.bits == 16
        assert ar.model.model.layers[0].mlp.experts[0].up_proj.bits == 4
        assert ar.model.model.layers[0].mlp.shared_expert.gate_proj.bits == 8

        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_w4a16_mixed_mllm(self, tiny_qwen_2_5_vl_model_path, dataloader):

        ar = AutoRound(
            tiny_qwen_2_5_vl_model_path,
            scheme="W4A16_MIXED",
            nsamples=1,
            batch_size=1,
            iters=0,
            seqlen=2,
            dataset=dataloader,
            low_cpu_mem_usage=False,
        )
        ar.quantize()
        assert ar.bits == 4
        assert ar.model.language_model.layers[0].self_attn.q_proj.bits == 16
        assert ar.model.visual.blocks[0].attn.qkv.bits == 16
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_mxfp4(self, tiny_opt_model_path, dataloader):
        ar = AutoRound(tiny_opt_model_path, scheme="MXFP4", nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "mx_fp"
        assert ar.act_data_type == "mx_fp"
        ar.quantize()

    def test_vllm(self, tiny_qwen_vl_model_path):
        from auto_round import AutoRoundMLLM

        ar = AutoRoundMLLM(tiny_qwen_vl_model_path, scheme="W2A16", nsamples=1, iters=1, seqlen=2)
        assert ar.bits == 2
        assert ar.act_bits == 16

    def test_nvfp4(self, tiny_opt_model_path, dataloader):
        ar = AutoRound(tiny_opt_model_path, scheme="NVFP4", nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "nv_fp"
        assert ar.act_data_type == "nv_fp4_with_static_gs"
        ar.quantize()

    def test_all_scheme(self, tiny_opt_model_path, tiny_qwen_model_path, dataloader):
        import copy

        preset_schemes = ["W8A16", "MXFP8", "FPW8A16", "FP8_STATIC", "GGUF:Q2_K_S", "GGUF:Q4_K_M"]
        for scheme in preset_schemes:
            model_name = tiny_opt_model_path
            if "gguf" in scheme.lower():
                model_name = tiny_qwen_model_path
            print(f"scheme={scheme}")
            ar = AutoRound(model_name, scheme=scheme, nsamples=1, iters=1, seqlen=2, dataset=dataloader)
            ar.quantize_and_save(self.save_folder)
            shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_scheme_in_layer_config(self, dataloader):
        model = get_tiny_model(opt_name_or_path, num_layers=5)
        tokenizer = transformers.AutoTokenizer.from_pretrained(opt_name_or_path, trust_remote_code=True)
        layer_config = {
            "model.decoder.layers.2.self_attn": {"bits": 2},
            "model.decoder.layers.3.self_attn.v_proj": "W8A16",
            "model.decoder.layers.4.self_attn.k_proj": QuantizationScheme.from_dict({"group_size": 64}),
        }
        ar = AutoRound(
            model,
            tokenizer,
            scheme="W3A16",
            nsamples=1,
            iters=1,
            layer_config=layer_config,
            seqlen=2,
            dataset=dataloader,
        )

        ar.quantize()
        for n, m in ar.model.named_modules():
            if n == "model.decoder.layers.2.self_attn.q_proj":
                assert m.bits == 2
            if n == "model.decoder.layers.2.self_attn.k_proj":
                assert m.bits == 2
            if n == "model.decoder.layers.3.self_attn.v_proj":
                assert m.bits == 8
            if n == "model.decoder.layers.4.self_attn.k_proj":
                assert m.group_size == 64

    def test_parse_available_devices(self):
        from auto_round.utils.device import parse_available_devices

        device_list = parse_available_devices("auto")
        assert len(device_list) == 1 and "cpu" in device_list
        device_list = parse_available_devices("a:cuda:0,b:cuda:1,c:cpu")
        assert len(device_list) == 3
        assert device_list == ["cuda:0", "cuda:1", "cpu"]
        device_list = parse_available_devices("0,1")
        assert len(device_list) == 1 and "cpu" in device_list

    def test_set_scheme(self, tiny_qwen_model_path):
        ar = AutoRound(
            tiny_qwen_model_path,
            scheme="gguf:q2_k_s",
            data_type="fp",
            nsamples=1,
            disable_opt_rtn=True,
            iters=0,
            seqlen=2,
        )
        ar.quantize()

        from auto_round.schemes import QuantizationScheme

        qs = QuantizationScheme.from_dict({"bits": 4, "group_size": 64})
        ar = AutoRound(
            tiny_qwen_model_path,
            scheme=qs,
            bits=2,
            data_type="int_asym_dq",
            nsamples=1,
            iters=0,
            disable_opt_rtn=True,
            seqlen=2,
        )
        ar.quantize()
