import os
import re
import sys
import argparse
import subprocess
from packaging import version

sys.path.insert(0, '../..')
parser = argparse.ArgumentParser()
import torch
import transformers

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import logging
import warnings
import numexpr

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dataset_logger = logging.getLogger("datasets")
dataset_logger.disabled = True
numexpr_logger = logging.getLogger("numexpr")
numexpr_logger.disabled = True

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

    parser.add_argument("--eval_bs", default=None, type=int,
                        help="eval batch size")

    parser.add_argument("--device", default="auto", type=str,
                        help="The device to be used for tuning. The default is set to auto/None,"
                             "allowing for automatic detection. Currently, device settings support CPU, GPU, and HPU.")

    parser.add_argument("--asym", action='store_true',
                        help=" asym quantization")

    parser.add_argument("--iters", default=200, type=int,
                        help=" iters")

    parser.add_argument("--dataset", default="NeelNanda/pile-10k", type=str,
                        help="The dataset for quantization training. It can be a custom one.")

    parser.add_argument("--enable_quanted_input", action='store_true',
                        help="enable_quanted_input is deprecated.")

    parser.add_argument("--lr", default=None, type=float,
                        help="learning rate, if None, it will be set to 1.0/iters automatically")

    parser.add_argument("--minmax_lr", default=None, type=float,
                        help="minmax learning rate, if None,it will beset to be the same with lr")

    parser.add_argument("--seed", default=42, type=int,
                        help="seed")

    parser.add_argument("--amp", action='store_true',
                        help="amp is deprecated ")

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

    parser.add_argument("--enable_minmax_tuning", action='store_true',
                        help="enable_minmax_tuning is deprecated")

    parser.add_argument("--deployment_device", default=None, type=str,
                        help="targeted inference acceleration platform,The options are 'fake', 'cpu', 'xpu', 'gpu(auto_gptq)' and 'auto_round'."
                             "default to 'auto_round', 'fake' indicating that it only performs fake quantization and won't be exported to any device.")

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
                        help="Choose which low cpu memory mode to use. Can significantly reduce cpu memory footprint but cost more time."
                             "1 means choose block-wise mode, load the weights of each block from disk when tuning and release the memory of the block after tuning."
                             "2 means choose layer-wise mode, load the weights of each layer from disk when tuning, minimum memory consumption and also slowest running speed."
                             "others means not use low cpu memory. Default to 0, not use low cpu memory.")

    parser.add_argument("--low_cpu_mem_tmp_dir", default=None, type=str,
                        help="temp work space to store the temporary files when using low cpu memory mode. Will remove after tuning.")

    parser.add_argument("--model_dtype", default=None, type=str,
                        help="force to convert the dtype, some backends supports fp16 dtype better")

    parser.add_argument("--act_bits", default=32, type=int,
                        help="activation bits")

    parser.add_argument("--fp_layers", default="", type=str,
                        help="List of Layers to maintain original data type")

    args = parser.parse_args()

    if args.enable_minmax_tuning:
        print(
            "enable_minmax_tuning is deprecated, it has been set to the default, use disable_minmax_tuning to turn it off")
    if args.amp:
        print(
            "amp is deprecated, it has been set to the default, use disable_amp to turn it off")
    if args.enable_quanted_input:
        print(
            "enable_quanted_input is deprecated. It has been set to the default; use disable_quanted_input to turn it off")

    if args.act_bits <= 8:
        print(
            "Warning, activation quantization is an experiment feature")

    tasks = args.tasks
    use_eval_legacy = False

    if args.format and args.deployment_device:
        assert False, "please only specify one of format and deployment_device"

    if args.deployment_device is None and args.format is None:
        args.format = "auto_round"

    if args.deployment_device:
        warnings.warn("The deployment_device is deprecated and will be removed in future version."
                      "Please use format instead", DeprecationWarning)
        if "gpu" in args.deployment_device and args.asym is True:
            print(
                "warning: The auto_gptq kernel has issues with asymmetric quantization. It is recommended to use sym quantization or --format='auto_round'")

        if "marlin" in args.deployment_device and args.asym is True:
            assert False, "marlin backend only supports sym quantization, please remove --asym"

    model_name = args.model_name
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    print(model_name, flush=True)

    from auto_round.utils import detect_device, detect_device_count

    device_str = detect_device(args.device)
    torch_dtype = "auto"
    if "hpu" in device_str:
        torch_dtype = torch.bfloat16
    torch_device = torch.device(device_str)

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
        if detect_device_count() > 1:
            model = model_cls.from_pretrained(
                model_name, low_cpu_mem_usage=True, torch_dtype=torch_dtype,
                trust_remote_code=not args.disable_trust_remote_code, device_map="auto"
            )
        else:
            model = model_cls.from_pretrained(
                model_name, low_cpu_mem_usage=True, torch_dtype=torch_dtype,
                trust_remote_code=not args.disable_trust_remote_code
            )

    from auto_round import (AutoRound,
                            AutoAdamRound)

    model = model.eval()

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

    excel_name = f"{model_name}_{args.bits}_{args.group_size}"
    if (hasattr(model, 'config') and (model.dtype is torch.bfloat16 or model.config.torch_dtype is torch.bfloat16)):
        dtype = 'bfloat16'
    else:
        if "cpu" not in device_str:
            dtype = 'float16'
        else:
            dtype = 'float32'

    excel_name = f"{model_name}_{args.bits}_{args.group_size}"
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
                    f"{n} will not be quantized due to its shape not being divisible by 32, resulting in an exporting issue to autogptq")
    fp_layers = args.fp_layers.split(",")
    if bool(fp_layers):
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.modeling_utils.Conv1D):
                name = n.split('.')[-1]
                if n in fp_layers or name in fp_layers:
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

    autoround = round(model, tokenizer, args.bits, args.group_size, sym=not args.asym, batch_size=args.train_bs,
                      dataset=args.dataset, seqlen=seqlen, nblocks=args.nblocks, iters=args.iters, lr=args.lr,
                      minmax_lr=args.minmax_lr, enable_quanted_input=not args.disable_quanted_input, device=device_str,
                      amp=not args.disable_amp, nsamples=args.nsamples,
                      low_gpu_mem_usage=args.low_gpu_mem_usage,
                      seed=args.seed, gradient_accumulate_steps=args.gradient_accumulate_steps,
                      scale_dtype=args.scale_dtype, layer_config=layer_config,
                      enable_minmax_tuning=not args.disable_minmax_tuning, act_bits=args.act_bits,
                      low_cpu_mem_usage=low_cpu_mem_usage, data_type=args.data_type,
                      enable_norm_bias_tuning=args.enable_norm_bias_tuning)
    model, _ = autoround.quantize()
    model_name = args.model_name.rstrip("/")
    if args.low_cpu_mem_mode == 1 or args.low_cpu_mem_mode == 2:
        import shutil

        shutil.rmtree(args.low_cpu_mem_tmp_dir, ignore_errors=True)

    model.eval()
    if "cpu" not in device_str:
        torch.cuda.empty_cache()

    export_dir = args.output_dir + "/" + model_name.split('/')[-1] + f"-w{args.bits}g{args.group_size}"
    output_dir = args.output_dir + "/" + model_name.split('/')[-1] + f"-w{args.bits}g{args.group_size}-qdq"

    eval_folder = None
    if args.format:
        format_list = args.format.replace(' ', '').split(',')
        inplace = False if len(format_list) > 1 else True
        for format_ in format_list:
            save_format_ = format_.replace(":", "-")
            save_format_ = save_format_.replace("_", "-")
            eval_folder = f'{export_dir}-{save_format_}'
            autoround.save_quantized(eval_folder, format=format_, inplace=inplace)
    else:
        deployment_device = args.deployment_device.split(',')
        gpu_formats = []
        for item in deployment_device:
            if item in ["gpu", "auto_gptq", "auto_round", "auto_awq"]:
                if item == "gpu":
                    if lm_head_layer_name in layer_config.keys() and layer_config[lm_head_layer_name][
                        "data_type"] == "int":
                        gpu_formats.append("auto_round")
                    else:
                        gpu_formats.append("auto_gptq")
                else:
                    gpu_formats.append(item)

        gpu_formats = list(set(gpu_formats))

        inplace = True if len(deployment_device) < 2 else False
        for gpu_format in gpu_formats:
            eval_folder = f'{export_dir}-{gpu_format}'
            autoround.save_quantized(eval_folder, format=gpu_format, inplace=inplace)

        if 'xpu' in deployment_device:
            autoround.save_quantized(f'{export_dir}-xpu', format="itrex_xpu", use_triton=True, inplace=inplace)
        if "cpu" in deployment_device:
            autoround.save_quantized(output_dir=f'{export_dir}-cpu', format='itrex', inplace=inplace)
        if "fake" in deployment_device:
            model = model.to("cpu")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            if eval_folder is None:
                eval_folder = output_dir

        if (not ('gpu' in deployment_device or len(gpu_formats) > 0)) and 'fake' not in deployment_device:
            print('does not support cpu, xpu model evaluation.')
            exit()  ## does not support cpu,xpu model eval

    if args.disable_eval:
        exit()

    from packaging.version import Version
    from auto_round.utils import get_library_version

    lm_eval_version = get_library_version("lm_eval")
    lm_eval_version = Version(lm_eval_version)
    if lm_eval_version == Version("0.3.0"):
        use_eval_legacy = True
        from eval_legacy import eval_model_legacy as eval_model
    else:
        use_eval_legacy = False
        from eval_legacy import eval_model

    # evaluation
    if use_eval_legacy:
        print("Using the legacy lm_eval(0.3.0)")
    else:
        print(f"Using the lm_eval version {lm_eval_version}")

    if isinstance(tasks, str):
        tasks = tasks.split(',')

    if lm_eval_version < Version("0.4.2"):
        if args.eval_bs is None:
            args.eval_bs = 1
        if use_eval_legacy:
            if "mmlu" in tasks:
                tmp_tasks = tasks
                tasks = ["hendrycksTest-*" if x == "mmlu" else x for x in tmp_tasks]
            if "truthfulqa_mc1" in tasks or "truthfulqa_mc2" in tasks:
                tmp_tasks = tasks
                tasks = ["truthfulqa_mc" if "truthfulqa_mc" in x else x for x in tmp_tasks]
            seen = set()
            tmp_tasks = tasks
            tasks = [x for x in tmp_tasks if not (x in seen or seen.add(x))]

        excel_name = f"{output_dir}_result.xlsx"
        output_dir += "/"
        print(excel_name, flush=True)
        eval_model(
            model_path=output_dir, tasks=tasks, dtype=dtype, limit=None,
            eval_bs=args.eval_bs, use_accelerate=args.low_gpu_mem_usage,
            device=torch_device, excel_file=excel_name,
            trust_remote_code=not args.disable_trust_remote_code)

    if lm_eval_version >= Version("0.4.2"):
        if args.eval_bs is None:
            args.eval_bs = "auto"
        from eval.evaluation import simple_evaluate

        model_args = f"pretrained={eval_folder}"
        model_args = model_args + f",trust_remote_code={not args.disable_trust_remote_code}"
        user_model = None
        if args.act_bits <= 8:
            if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                from accelerate.big_modeling import dispatch_model

                dispatch_model(model, model.hf_device_map)
                user_model = model
            else:
                user_model = model.to(device_str)
            if args.eval_bs == "auto":
                args.eval_bs = 16
            from auto_round.eval.evaluation import  simple_evaluate_user_model
            res = simple_evaluate_user_model(user_model, tokenizer,tasks=tasks,batch_size=args.eval_bs)
        else:
            res = simple_evaluate(model="hf", model_args=model_args,
                                  tasks=tasks,
                                  batch_size=args.eval_bs)
        from lm_eval.utils import make_table

        print(make_table(res))
