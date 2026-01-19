import os
import shutil

from auto_round import AutoRound

from ....helpers import get_model_path, save_tiny_model


class TestGGUFQ2KMixed:

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_q2k_mixed(self):
        model_name = get_model_path("Qwen/Qwen1.5-MoE-A2.7B")
        saved_tiny_model_path = save_tiny_model(
            model_name,
            "./tmp/tiny_qwen_model_path",
            num_layers=3,
            is_mllm=False,
        )
        autoround = AutoRound(
            saved_tiny_model_path,
            iters=0,
            nsamples=1,
            seqlen=16,
            disable_opt_rtn=True,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q2_k_mixed")
        gguf_file = os.listdir(quantized_model_path)[0]
        file_size = os.path.getsize(os.path.join(quantized_model_path, gguf_file)) / 1024**2
        assert abs(file_size - 1362) < 5.0
        from gguf.gguf_reader import GGUFReader

        gguf_model = GGUFReader(os.path.join(quantized_model_path, gguf_file))
        assert gguf_model.get_tensor(2).name == "blk.0.attn_k.weight"
        assert gguf_model.get_tensor(2).tensor_type.name == "Q4_K"
        assert gguf_model.get_tensor(10).name == "blk.0.ffn_up_exps.weight"
        assert gguf_model.get_tensor(10).tensor_type.name == "Q2_K"

        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree(saved_tiny_model_path, ignore_errors=True)
