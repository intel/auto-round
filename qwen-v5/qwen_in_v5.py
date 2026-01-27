import psutil
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.import_utils import clear_import_cache


# clear cache to reload modified code
clear_import_cache()
model_name = "/storage/yiliu7/deepseek-ai/DeepSeek-R1-0528/"
model_name = "/storage/yiliu7/unsloth/DeepSeek-R1-BF16/"
model_name = "/mnt/disk5/unsloth/DeepSeek-R1-BF16"
model_name = "/mnt/disk8/Qwen/Qwen3-8B-FP8"
model_name = "/mnt/disk6/yiliu4/deepseek-ai/DeepSeek-R1-0528"
model_name = "/mnt/disk8/Qwen/Qwen3-8B"
model_name = "/mnt/disk8/Qwen/Qwen3-8B-FP8"
model_name = "/mnt/disk8/Qwen/Qwen3-30B-A3B"
# model_name = "/mnt/disk8/deepseek-ai/DeepSeek-V2-Lite-Chat"
device = "cpu"
from loguru import logger


def dump_cur_ram(msg: str = ""):
    process = psutil.Process()
    current_ram = process.memory_info().rss / 1024**2  # MB
    logger.warning(f"[Memory] {msg} Current RAM usage: {round(current_ram, 2)}MB")


def fixed_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)


def disable_concat_experts():
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping

    register_checkpoint_conversion_mapping("deepseek_v3", [], overwrite=True)
    register_checkpoint_conversion_mapping("qwen3_moe", [], overwrite=True)


from torch.utils._debug_mode import DebugMode
from fp8_quantizer_patch import *
from transformers.initialization import no_init_weights


def main(args):
    model_name = args.model_name
    fixed_seed(42)

    # from v5_patch import apply_transformer_patches
    from qwen_v5_patch import apply_transformer_patches_qwen

    apply_transformer_patches_qwen()
    disable_concat_experts()
    # apply_transformer_patches()

    with torch.no_grad():
        trust_remote_code = False
        dump_cur_ram("before model load")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        with no_init_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                trust_remote_code=trust_remote_code,
                #   _experts_implementation="eager",
                device_map="cpu",  # device_map="auto",
            )
        msg = "The capital of France is"
        model.eval()
        dump_cur_ram("after model load")

        inputs = tokenizer(msg, return_tensors="pt").to(device)
        if args.debug:
            with (
                DebugMode(
                    record_stack_trace=args.record_stack_trace,
                    record_ids=True,
                ) as dm,
                DebugMode.log_tensor_hashes(
                    hash_inputs=True,
                ),
            ):
                # outputs = model.generate(**inputs, max_new_tokens=32)
                print(f"Inputs: {inputs['input_ids']}")
                res = model(inputs["input_ids"])

            print(dm.debug_string(show_stack_trace=True))
            print(res)
            exit(0)
        inputs = tokenizer(msg, return_tensors="pt").to("cpu")

        outputs = model.generate(**inputs, max_new_tokens=32)
        decode_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decode_output)
        exit(0)

        exit(0)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # output_dir = (
        #     args.output_dir
        #     if args.output_dir is not None
        #     else f"/storage/yiliu7/{model_name.rstrip('/').split('/')[-1]}-fp8-w4a16-4layers"
        # )
        # quant_ar(model, tokenizer, output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # input model path
    parser.add_argument("--model_name", type=str, default=model_name, help="Path to the pretrained model")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save the quantized model")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--record_stack_trace", "--stack", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    main(args)
