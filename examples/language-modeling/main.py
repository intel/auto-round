import argparse
import sys

sys.path.insert(0, '../..')
parser = argparse.ArgumentParser()
import torch
import os
import transformers

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from transformers import set_seed

import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':

    parser.add_argument(
        "--model_name", default="facebook/opt-125m"
    )

    parser.add_argument("--bits", default=4, type=int,
                        help="number of  bits")

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--train_bs", default=8, type=int,
                        help="train batch size")

    parser.add_argument("--eval_bs", default=4, type=int,
                        help="eval batch size")

    parser.add_argument("--device", default="auto", type=str,
                        help="The device to be used for tuning. The default is set to auto/None,"
                             "allowing for automatic detection. Currently, device settings support CPU, GPU, and HPU.")

    parser.add_argument("--sym", action='store_true',
                        help=" sym quantization")

    parser.add_argument("--iters", default=200, type=int,
                        help=" iters")

    parser.add_argument("--use_quant_input", action='store_true',
                        help="whether to use the output of quantized block to tune the next block")

    parser.add_argument("--lr", default=None, type=float,
                        help="learning rate, if None, it will be set to 1.0/iters automatically")

    parser.add_argument("--minmax_lr", default=None, type=float,
                        help="minmax learning rate, if None,it will beset to be the same with lr")

    parser.add_argument("--seed", default=42, type=int,
                        help="seed")

    parser.add_argument("--eval_fp16_baseline", action='store_true',
                        help="whether to eval FP16 baseline")

    parser.add_argument("--amp", action='store_true',
                        help="amp is deprecated ")

    parser.add_argument("--adam", action='store_true',
                        help="adam")

    parser.add_argument("--seqlen", default=2048, type=int,
                        help="sequence length")

    parser.add_argument("--gradient_accumulate_steps", default=1, type=int, help="gradient accumulate steps")

    parser.add_argument("--n_blocks", default=1, type=int, help="num of blocks to tune together")

    parser.add_argument("--n_samples", default=512, type=int,
                        help="number of samples")

    parser.add_argument("--low_gpu_mem_usage", action='store_true',
                        help="low_gpu_mem_usage is deprecated")

    parser.add_argument("--enable_minmax_tuning", action='store_true',
                        help="enable_minmax_tuning is deprecated")

    parser.add_argument("--deployment_device", default='fake', type=str,
                        help="targeted inference acceleration platform,The options are 'fake', 'cpu' and 'gpu'."
                             "default to 'fake', indicating that it only performs fake quantization and won't be exported to any device.")

    parser.add_argument("--scale_dtype", default='fp32',
                        help="which scale data type to use for quantization, 'fp16', 'fp32' or 'bf16'.")

    parser.add_argument("--tasks",
                        default=['wikitext2', 'ptb-new', 'c4-new', 'lambada_openai', 'hellaswag', 'winogrande', 'piqa',
                                 "mmlu", "wikitext", "truthfulqa_mc1", "truthfulqa_mc2", "openbookqa", "boolq", "rte",
                                 "arc_easy", "arc_challenge"],
                        help="lm-eval tasks for lm_eval version 0.4")

    parser.add_argument("--output_dir", default="./tmp_autoround", type=str,
                        help="Where to store the final model.")

    parser.add_argument("--disable_eval", action='store_true',
                        help="Whether to do lmeval evaluation.")

    parser.add_argument("--disable_amp", action='store_true',
                        help="disable amp")

    parser.add_argument("--disable_low_gpu_mem_usage", action='store_true',
                        help="disable low_gpu_mem_usage")

    parser.add_argument("--disable_minmax_tuning", action='store_true',
                        help="whether disable  enable weight minmax tuning")

    parser.add_argument("--disable_trust_remote_code", action='store_true',
                        help="Whether to disable trust_remote_code")


    args = parser.parse_args()
    if args.low_gpu_mem_usage:
        print(
            "low_gpu_mem_usage is deprecated, it has been set to the default, use disable_low_gpu_mem_usage to turn it off")
    if args.enable_minmax_tuning:
        print(
            "enable_minmax_tuning is deprecated, it has been set to the default, use disable_minmax_tuning to turn it off")
    if args.amp:
        print(
            "amp is deprecated, it has been set to the default, use disable_amp to turn it off")

    set_seed(args.seed)
    tasks = args.tasks
    use_eval_legacy = False
    import subprocess


    def get_library_version(library_name):
        try:
            version = subprocess.check_output(['pip', 'show', library_name]).decode().split('\n')[1].split(': ')[1]
            return version
        except subprocess.CalledProcessError:
            return "Library not found"


    res = get_library_version("lm-eval")
    if res == "0.3.0":
        use_eval_legacy = True

    if isinstance(tasks, str):
        tasks = tasks.split(',')
    if not use_eval_legacy:
        from eval import eval_model
    else:
        from eval_legacy import eval_model
        if isinstance(tasks, list):
            if "mmlu" in tasks:
                tmp_tasks = tasks
                tasks = ["hendrycksTest-*" if x == "mmlu" else x for x in tmp_tasks]
            if "truthfulqa_mc1" in tasks or "truthfulqa_mc2" in tasks:
                tmp_tasks = tasks
                tasks = ["truthfulqa_mc" if "truthfulqa_mc" in x else x for x in tmp_tasks]
            tasks = list(set(tasks))
        if isinstance(args.tasks, str):
            tasks = ','.join(tasks)

    if 'fake' in args.deployment_device and not args.disable_eval:
        if use_eval_legacy:
            print("Using the legacy lm_eval(0.3.0)")
        else:
            print("Using the latest lm_eval(0.4.1)")

    model_name = args.model_name
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    print(model_name, flush=True)

    from auto_round.utils import detect_device

    device_str = detect_device(args.device)
    torch_dtype = "auto"
    if "hpu" in device_str:
        torch_dtype = torch.bfloat16
    torch_device = torch.device(device_str)

    is_glm = bool(re.search("chatglm", model_name.lower()))
    if is_glm:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype=torch_dtype, trust_remote_code=not args.disable_trust_remote_code
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

    if "llama" in model_name:
        from transformers import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code)

    if hasattr(tokenizer, "model_max_length"):
        if tokenizer.model_max_length < seqlen:
            print(f"change sequence length to {tokenizer.model_max_length} due to the limitation of model_max_length",
                  flush=True)
            seqlen = min(seqlen, tokenizer.model_max_length)
            args.seqlen = seqlen

    pt_dtype = torch.float16
    if (hasattr(model, 'config') and (model.dtype is torch.bfloat16 or model.config.torch_dtype is torch.bfloat16)):
        dtype = 'bfloat16'
        pt_dtype = torch.bfloat16
    else:
        if str(args.device) != "cpu":
            pt_dtype = torch.float16
            dtype = 'float16'
        else:
            pt_dtype = torch.float32
            dtype = 'float32'

    excel_name = f"{model_name}_{args.bits}_{args.group_size}"
    if args.eval_fp16_baseline:
        if args.disable_low_gpu_mem_usage:
            model = model.to(torch_device)
        excel_name += "_fp16.xlsx"
        eval_model(model_path=model_name, tasks=tasks, dtype=dtype, \
                   eval_bs=args.eval_bs, use_accelerate=not args.disable_low_gpu_mem_usage,
                   device=torch_device, excel_file=excel_name)
        exit()

    if args.disable_low_gpu_mem_usage:
        model = model.to(torch_device)

    round = AutoRound
    if args.adam:
        round = AutoAdamRound

    weight_config = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.modeling_utils.Conv1D):
            if m.weight.shape[0] % 32 != 0 or m.weight.shape[1] % 32 != 0:
                weight_config[n] = {"data_type": "fp"}
                print(
                    f"{n} will not be quantized due to its shape not being divisible by 32, resulting in an exporting issue to autogptq")

    autoround = round(model, tokenizer, args.bits, args.group_size, sym=args.sym, batch_size=args.train_bs,
                      seqlen=seqlen, n_blocks=args.n_blocks, iters=args.iters, lr=args.lr,
                      minmax_lr=args.minmax_lr, use_quant_input=args.use_quant_input, device=device_str,
                      amp=not args.disable_amp, n_samples=args.n_samples,
                      low_gpu_mem_usage=not args.disable_low_gpu_mem_usage,
                      seed=args.seed, gradient_accumulate_steps=args.gradient_accumulate_steps,
                      scale_dtype=args.scale_dtype, weight_config=weight_config,
                      enable_minmax_tuning=not args.disable_minmax_tuning)  ##TODO args pass
    model, _ = autoround.quantize()
    model_name = args.model_name.rstrip("/")

    model.eval()
    if args.device != "cpu":
        torch.cuda.empty_cache()

    export_dir = args.output_dir + "/" + model_name.split('/')[-1] + f"-autoround-w{args.bits}g{args.group_size}"
    output_dir = args.output_dir + "/" + model_name.split('/')[-1] + f"-autoround-w{args.bits}g{args.group_size}-qdq"
    deployment_device = args.deployment_device.split(',')
    if 'gpu' in deployment_device:
        autoround.save_quantized(f'{export_dir}-gpu', format="auto_gptq", use_triton=True, inplace=False)
    if "cpu" in deployment_device:
        autoround.save_quantized(output_dir=f'{export_dir}-cpu', format='itrex', inplace=False)
    if "fake" in deployment_device:
        model = model.to("cpu")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    if not args.disable_eval and "fake" in deployment_device:  ##support autogptq real eval later
        excel_name = f"{output_dir}_result.xlsx"
        output_dir += "/"
        print(excel_name, flush=True)
        eval_model(model_path=output_dir, tasks=tasks, dtype=dtype, limit=None,
                   eval_bs=args.eval_bs, use_accelerate=not args.disable_low_gpu_mem_usage,
                   device=torch_device, excel_file=excel_name)
