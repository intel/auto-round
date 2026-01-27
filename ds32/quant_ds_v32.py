import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.import_utils import clear_import_cache

# clear cache to reload modified code
clear_import_cache()
model_name = "/storage/yiliu7/deepseek-ai/DeepSeek-V3.2/"
model_name = "/storage/yiliu7/DeepSeek-V3.2-4layers/"
model_name = "/mnt/disk8/Qwen/Qwen3-8B-FP8"
device = "cpu"

from ds_v47 import *


def fixed_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)


fixed_seed(42)


def quant_ar(model, tokenizer, output_dir):

    from auto_round import AutoRound

    scheme = "W4A16"
    autoround = AutoRound(
        model,
        tokenizer,
        scheme=scheme,
        enable_torch_compile=False,
        iters=0,
        low_gpu_mem_usage=True,
        disable_opt_rtn=True,
        ignore_layers="indexer",
    )
    model_base_name = model_name.rstrip("/").split("/")[-1]
    print(f"Output dir: {output_dir}")

    model, save_folder = autoround.quantize_and_save(
        output_dir=output_dir,
    )


def check_meta_module(model):
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            if param.device.type == "meta":
                print("Found meta parameter:", pname, "in module:", name, "shape:", param.shape)
                breakpoint()
                raise RuntimeError(
                    f"The model contains some parameters on the meta device (found in module {name}, parameter {name}). "
                )
from torch.utils._debug_mode import DebugMode

def main(args):
    model_name = args.model_name
    with torch.no_grad():
        trust_remote_code = False
        # trust_remote_code = True
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
            #   _experts_implementation="eager",
            device_map="cpu",  # device_map="auto",
        )
        msg = "The capital of France is"
        model.eval()
        print(model)
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
        
        inputs = tokenizer(msg, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=32)

        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        exit(0)
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else f"/storage/yiliu7/{model_name.rstrip('/').split('/')[-1]}-fp8-w4a16-4layers"
        )
        quant_ar(model, tokenizer, output_dir=output_dir)


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
