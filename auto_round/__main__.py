# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import argparse

import torch
import subprocess
import transformers
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from lm_eval.utils import make_table  # pylint: disable=E0401

from auto_round import AutoRoundConfig
from auto_round.eval.evaluation import simple_evaluate
from auto_round.utils import detect_device

def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", default="facebook/opt-125m"
    )

    parser.add_argument('--eval', action='store_true', 
                        help="whether to use eval mode.")

    parser.add_argument("--bits", default=4, type=int,
                        help="number of  bits")

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--batch_size", default=8, type=int,
                        help="train batch size")

    parser.add_argument("--eval_bs", default=1, type=int,
                        help="eval batch size")

    parser.add_argument("--device", default="auto", type=str,
                        help="The device to be used for tuning. The default is set to auto/None,"
                             "allowing for automatic detection. Currently, device settings support CPU, GPU, and HPU.")

    parser.add_argument("--sym", action='store_true',
                        help=" sym quantization")

    parser.add_argument("--iters", default=200, type=int,
                        help=" iters")

    parser.add_argument("--dataset", default="NeelNanda/pile-10k", type=str,
                        help="The dataset for quantization training. It can be a custom one.")

    parser.add_argument("--lr", default=None, type=float,
                        help="learning rate, if None, it will be set to 1.0/iters automatically")

    parser.add_argument("--minmax_lr", default=None, type=float,
                        help="minmax learning rate, if None,it will beset to be the same with lr")

    parser.add_argument("--seed", default=42, type=int,
                        help="seed")

    parser.add_argument("--adam", action='store_true',
                        help="adam")

    parser.add_argument("--seqlen", default=2048, type=int,
                        help="sequence length")

    parser.add_argument("--gradient_accumulate_steps", default=1, type=int, help="gradient accumulate steps")

    parser.add_argument("--nblocks", default=1, type=int, help="num of blocks to tune together")

    parser.add_argument("--nsamples", default=128, type=int,
                        help="number of samples")

    parser.add_argument("--low_gpu_mem_usage", action='store_true',
                        help="lower gpu memory but 50%-100% slower")

    parser.add_argument("--format", default=None, type=str,
                        help="The format in which to save the model. "
                        "The options are 'auto_round', 'auto_gptq', 'auto_awq', 'itrex', 'itrex_xpu' and 'fake'."
                        "default to 'auto_round."
                        )

    parser.add_argument("--data_type", default='int',
                        help="data type for tuning, 'int', 'mx_fp' and etc.")

    parser.add_argument("--scale_dtype", default='fp16',
                        help="which scale data type to use for quantization, 'fp16', 'fp32' or 'bf16'.")

    parser.add_argument("--tasks",
                        default="lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1," \
                                "truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge",
                        help="lm-eval tasks for lm_eval version 0.4")

    parser.add_argument("--output_dir", default="./tmp_autoround", type=str,
                        help="Where to store the final model.")

    parser.add_argument("--disable_eval", action='store_true',
                        help="Whether to do lmeval evaluation.")

    parser.add_argument("--disable_amp", action='store_true',
                        help="disable amp")

    parser.add_argument("--disable_minmax_tuning", action='store_true',
                        help="whether disable enable weight minmax tuning")

    parser.add_argument("--enable_norm_bias_tuning", action='store_true',
                        help="whether enable norm bias tuning")

    parser.add_argument("--disable_trust_remote_code", action='store_true',
                        help="Whether to disable trust_remote_code")

    parser.add_argument("--disable_quanted_input", action='store_true',
                        help="whether to disuse the output of quantized block to tune the next block")

    parser.add_argument("--quant_lm_head", action='store_true',
                        help="quant_lm_head")

    parser.add_argument("--low_cpu_mem_mode", default=0, type=int,
                        help="Choose which low cpu memory mode to use. "
                        "Can significantly reduce cpu memory footprint but cost more time."
                        "1 means choose block-wise mode, load the weights of each block"
                        " from disk when tuning and release the memory of the block after tuning."
                        "2 means choose layer-wise mode, load the weights of each layer from disk when tuning,"
                        " minimum memory consumption and also slowest running speed."
                        "others means not use low cpu memory. Default to 0, not use low cpu memory.")

    parser.add_argument("--low_cpu_mem_tmp_dir", default=None, type=str,
                        help="temp work space to store the temporary files "
                        "when using low cpu memory mode. Will remove after tuning.")

    parser.add_argument("--model_dtype", default=None, type=str,
                        help="force to convert the dtype, some backends supports fp16 dtype better")

    parser.add_argument("--act_bits", default=32, type=int,
                        help="activation bits")
    
    parser.add_argument("--fp_layers_list", default="", type=str,
                        help="List of Layers to maintain original data type")

    args = parser.parse_args()
    return args

def tune(args):
    tasks = args.tasks
    if args.format is None:
        args.format = "auto_round"

    model_name = args.model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    print(model_name, flush=True)


    device_str = detect_device(args.device)
    torch_dtype = "auto"
    if "hpu" in device_str:
        torch_dtype = torch.bfloat16

    is_glm = bool(re.search("chatglm", model_name.lower()))
    low_cpu_mem_usage = False
    model_cls = AutoModel if is_glm else AutoModelForCausalLM
    if args.low_cpu_mem_tmp_dir is None:
        args.low_cpu_mem_tmp_dir = os.path.join(args.output_dir, "low_cpu_mem_tmp")
    if args.low_cpu_mem_mode == 2:
        from auto_round.low_cpu_mem.utils import load_model_with_hooks

        model = load_model_with_hooks(
            model_name,
            model_cls,
            device=device_str,
            clean_weight=True,
            saved_path=args.low_cpu_mem_tmp_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=not args.disable_trust_remote_code
        )
    elif args.low_cpu_mem_mode == 1:
        from auto_round.low_cpu_mem.utils import load_empty_model

        low_cpu_mem_usage = True
        model = load_empty_model(
            model_name,
            model_cls,
            device=device_str,
            saved_path=args.low_cpu_mem_tmp_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=not args.disable_trust_remote_code
        )
    else:
        model = model_cls.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype=torch_dtype,
            trust_remote_code=not args.disable_trust_remote_code
        )

    from auto_round import (AutoRound,
                            AutoAdamRound)

    model = model.eval()
    # align with GPTQ to eval ppl
    if "opt" in model_name:
        seqlen = model.config.max_position_embeddings
        model.seqlen = model.config.max_position_embeddings
    else:
        seqlen = 2048
        model.seqlen = seqlen
    seqlen = args.seqlen

    if args.model_dtype != None:
        if args.model_dtype == "float16" or args.model_dtype == "fp16":
            model = model.to(torch.float16)
        if args.model_dtype == "bfloat16" or args.model_dtype == "bfp16":
            model = model.to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code)

    if hasattr(tokenizer, "model_max_length"):
        if tokenizer.model_max_length < seqlen:
            print(f"change sequence length to {tokenizer.model_max_length} due to the limitation of model_max_length",
                  flush=True)
            seqlen = min(seqlen, tokenizer.model_max_length)
            args.seqlen = seqlen

    if "bloom" in model_name:
        args.low_gpu_mem_usage = False

    round = AutoRound
    if args.adam:
        round = AutoAdamRound

    layer_config = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.modeling_utils.Conv1D):
            if m.weight.shape[0] % 32 != 0 or m.weight.shape[1] % 32 != 0:
                layer_config[n] = {"bits": 32}
                print(
                    f"{n} will not be quantized due to its shape not being divisible by 32,"
                    " resulting in an exporting issue to autogptq")
    fp_layers_list = args.fp_layers_list.split(",")
    if bool(fp_layers_list):
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.modeling_utils.Conv1D):
                name = n.split('.')[-1]
                if n in fp_layers_list or name in fp_layers_list:
                    layer_config[n] = {"bits": 32}
                    print(
                        f"{n} will not be quantized.")
    lm_head_layer_name = "lm_head"
    for n, _ in model.named_modules():
        lm_head_layer_name = n
    if args.quant_lm_head:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code)
        if config.tie_word_embeddings and hasattr(model, "_tied_weights_keys"):
            tied_keys = model._tied_weights_keys
            for item in tied_keys:
                if lm_head_layer_name in item:  ##TODO extend to encoder-decoder layer, seq classification model
                    args.quant_lm_head = False
                    print(
                        f"warning, disable quant_lm_head as quantizing lm_head with tied weights has not been "
                        f"supported currently")
                    break
    if args.quant_lm_head:
        layer_config[lm_head_layer_name] = {"bits": args.bits}
        transformers_version = [int(item) for item in transformers.__version__.split('.')[:2]]
        if transformers_version[0] == 4 and transformers_version[1] < 38:
            error_message = "Please upgrade transformers>=4.38.0 to support lm-head quantization."
            raise EnvironmentError(error_message)

    if args.quant_lm_head and args.low_gpu_mem_usage:
        print(
            f"warning, low_gpu_mem_usage=False is strongly recommended"
            " if the whole model could be loaded to gpu")

    autoround = round(
        model, tokenizer, args.bits, args.group_size, sym=args.sym, batch_size=args.batch_size,
        dataset=args.dataset, seqlen=seqlen, nblocks=args.nblocks, iters=args.iters, lr=args.lr,
        minmax_lr=args.minmax_lr, enable_quanted_input=not args.disable_quanted_input, 
        device=device_str, amp=not args.disable_amp, nsamples=args.nsamples, seed=args.seed,
        low_gpu_mem_usage=args.low_gpu_mem_usage, scale_dtype=args.scale_dtype, 
        gradient_accumulate_steps=args.gradient_accumulate_steps, layer_config=layer_config, 
        enable_minmax_tuning=not args.disable_minmax_tuning, act_bits=args.act_bits,
        low_cpu_mem_usage=low_cpu_mem_usage, data_type=args.data_type,
        enable_norm_bias_tuning=args.enable_norm_bias_tuning)
    model, _ = autoround.quantize()
    model_name = args.model.rstrip("/")
    if args.low_cpu_mem_mode == 1 or args.low_cpu_mem_mode == 2:
        import shutil

        shutil.rmtree(args.low_cpu_mem_tmp_dir, ignore_errors=True)

    model.eval()
    if "cpu" not in device_str:
        torch.cuda.empty_cache()

    export_dir = args.output_dir + "/" + model_name.split('/')[-1] + f"-w{args.bits}g{args.group_size}"


    format_list = args.format.replace(' ', '').split(',')
    inplace = False if len(format_list) > 1 else True
    for format_ in format_list:
        eval_folder = f'{export_dir}-{format_}'
        autoround.save_quantized(eval_folder, format=format_, inplace=inplace)


    def get_library_version(library_name):
        try:
            version = subprocess.check_output(['pip', 'show', library_name]).decode().split('\n')[1].split(': ')[1]
            return version
        except subprocess.CalledProcessError:
            return "Library not found"


    lm_eval_version = get_library_version("lm-eval")

    if isinstance(tasks, str):
        tasks = tasks.split(',')

    if not args.disable_eval:
        print(f"Using the latest {lm_eval_version}")
        model_args = f"pretrained={eval_folder}"
        model_args = model_args + f",trust_remote_code={not args.disable_trust_remote_code}"
        user_model = None
        if args.act_bits <= 8:
            user_model = model.to(device_str)

        res = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            batch_size=args.eval_bs,
            user_model=user_model)

        print(make_table(res))


def eval(args):
    device_str = detect_device(args.device)
    model_args = f"pretrained={args.model},trust_remote_code={not args.disable_trust_remote_code}"
    if isinstance(args.tasks, str):
        tasks = args.tasks.split(',')
    res = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        device=device_str,
        batch_size=args.eval_bs)

    from lm_eval.utils import make_table  # pylint: disable=E0401

    print(make_table(res))


def run():
    args = setup_parser()
    if args.eval:
        eval(args)
    else:
        tune(args)


if __name__ == '__main__':
    run()
