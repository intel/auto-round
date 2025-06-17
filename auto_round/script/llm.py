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
import sys
import logging
from auto_round.utils import (
    get_fp_layer_names,
    clear_memory,
    get_device_and_parallelism,
    get_model_dtype,
    str2bool,
    set_cuda_visible_devices)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class BasicArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--model", "--model_name", "--model_name_or_path", default="facebook/opt-125m", help="model name or path")

        self.add_argument('--eval', action='store_true', help="whether to use eval only mode")

        self.add_argument("--bits", default=4, type=int, help="number of weight bits")

        self.add_argument("--eval_bs", default=None, type=int, help="batch size in evaluation")

        self.add_argument(
            "--device",
            "--devices",
            default="0",
            type=str,
            help="the device to be used for tuning. "
                 "Currently, device settings support CPU, GPU, and HPU."
                 "The default is set to cuda:0,"
                 "allowing for automatic detection and switch to HPU or CPU."
                 "set --device 0,1,2 to use multiple cards.")

        self.add_argument("--asym", action='store_true', help="whether to use asym quantization")

        self.add_argument(
            "--dataset", default="NeelNanda/pile-10k", type=str, help="the dataset for quantization training")

        self.add_argument(
            "--minmax_lr",
            default=None,
            type=float,
            help="minmax learning rate, if None, it will beset to be the same with lr")

        self.add_argument("--seed", default=42, type=int, help="random seed")

        self.add_argument("--adam", action='store_true', help="whether to use adam optimizer instead of SignSGD")

        self.add_argument("--gradient_accumulate_steps", default=1, type=int, help="gradient accumulate steps")

        self.add_argument("--nblocks", default=1, type=int, help="how many blocks to tune together")

        self.add_argument("--low_gpu_mem_usage", action='store_true', help="offload intermediate features to cpu")

        self.add_argument("--format", default="auto_round", type=str, help="the format to save the model")

        self.add_argument("--data_type", "--dtype", default='int', help="data type for tuning, 'int', 'mx_fp' and etc")

        self.add_argument(
            "--scale_dtype",
            default='fp16',
            choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"],
            help="scale data type to use for quantization")

        self.add_argument(
            "--tasks",
            "--task",
            nargs='?',
            const="lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1,"
                  "openbookqa,boolq,arc_easy,arc_challenge",
            default=None,
            help="lm-eval tasks")

        self.add_argument(
            "--output_dir", default="./tmp_autoround", type=str, help="the directory to save quantized model")

        self.add_argument("--disable_eval", action='store_true',
                          help="whether to disable lm-eval evaluation after tuning")

        self.add_argument(
            "--eval_task_by_task",
            action="store_true",
            help="whether to eval task by task.")

        self.add_argument("--disable_amp", action='store_true', help="disable amp")

        self.add_argument(
            "--disable_minmax_tuning", action='store_true', help="whether to disable enable weight minmax tuning")

        self.add_argument("--enable_norm_bias_tuning", action='store_true', help="whether to enable norm bias tuning")

        self.add_argument(
            "--disable_trust_remote_code", action='store_true', help="whether to disable trust_remote_code")

        self.add_argument(
            "--disable_quanted_input",
            action='store_true',
            help="whether to disuse the output of quantized block to tune the next block")

        self.add_argument("--quant_lm_head", action='store_true', help="whether to quant lm_head")

        self.add_argument(
            "--low_cpu_mem_mode",
            default=0,
            type=int,
            choices=[0, 1, 2],
            help="choose which low cpu memory mode to use. "
                 "Can significantly reduce cpu memory footprint but cost more time."
                 "1 means choose block-wise mode, load the weights of each block"
                 " from disk when tuning and release the memory of the block after tuning."
                 "2 means choose layer-wise mode, load the weights of each layer from disk when tuning,"
                 " minimum memory consumption and also slowest running speed."
                 "others means not use low cpu memory. Default to 0, not use low cpu memory.")

        self.add_argument(
            "--low_cpu_mem_tmp_dir",
            default=None,
            type=str,
            help="temporary work space to store the temporary files "
                 "when using low cpu memory mode. Will remove after tuning.")

        self.add_argument(
            "--model_dtype",
            default=None,
            type=str,
            choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"],
            help="force to convert the dtype, some backends supports fp16 dtype better")

        self.add_argument("--act_bits", default=16, type=int, help="activation bits")

        self.add_argument(
            "--fp_layers", default="", type=str, help="list of Layer names to maintain original data type")

        self.add_argument(
            "--not_use_best_mse",
            action='store_true',
            help="whether to use the iter of best mes loss in the tuning phase")

        self.add_argument(
            "--to_quant_block_names",
            default=None,
            type=str,
            help="Names of quantitative blocks, please use commas to separate them.")

        self.add_argument("--enable_torch_compile", action='store_true',
                          help="whether to enable torch compile")

        self.add_argument("--act_data_type", "--act_dtype", default=None, type=str, help="activation data type")

        self.add_argument("--disable_act_dynamic", action='store_true', help="activation static quantization")

        self.add_argument("--disable_deterministic_algorithms", action='store_true',
                          help="disable torch deterministic algorithms.")

        self.add_argument("--device_map", default=None, type=str, help="device_map for block in tuning phase")

        self.add_argument(
            "--super_group_size", default=None, type=int, help="the number of super group size when use double quant.")

        self.add_argument(
            "--super_bits", default=None, type=int, help="number of scale and mins quant bits for double quant.")

        self.add_argument(
            "--eval_model_dtype",
            default=None,
            type=str,
            help="the torch_dytpe to load the model for evaluation.")

        self.add_argument("--disable_opt_rtn", action='store_true',
                          help="whether to disable optimization of the RTN mode(iters=0) (default is False).")


class EvalArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--model", "--model_name", "--model_name_or_path", default="facebook/opt-125m", help="model name or path")
        self.add_argument(
            "--device",
            "--devices",
            default="0",
            type=str,
            help="the device to be used for tuning. "
                 "Currently, device settings support CPU, GPU, and HPU."
                 "The default is set to cuda:0,"
                 "allowing for automatic detection and switch to HPU or CPU."
                 "set --device 0,1,2 to use multiple cards.")

        self.add_argument("--tasks", "--task",
                          default="lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1," \
                                  "truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge",
                          help="lm-eval tasks")
        self.add_argument(
            "--disable_trust_remote_code", action='store_true', help="whether to disable trust_remote_code")
        self.add_argument("--eval_bs", "--bs", "--batch_size", default=None, type=int, help="batch size in evaluation")
        self.add_argument("--eval_task_by_task", action='store_true', help="whether to eval task by task.")
        self.add_argument(
            "--eval_model_dtype",
            default=None,
            type=str,
            help="the torch_dytpe to load the model for evaluation.")


def setup_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--group_size", default=128, type=int, help="group size")

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=8, type=int, help="train batch size")

    parser.add_argument("--iters", "--iter", default=200, type=int, help="iteration to tune each block")

    parser.add_argument(
        "--seqlen", "--seq_len", default=2048, type=int, help="sequence length of the calibration samples")

    parser.add_argument("--nsamples", "--nsample", default=128, type=int, help="number of samples")

    parser.add_argument(
        "--lr", default=None, type=float, help="learning rate, if None, it will be set to 1.0/iters automatically")

    args = parser.parse_args()
    return args


def setup_best_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--group_size", default=128, type=int, help="group size")

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=8, type=int, help="train batch size")

    parser.add_argument("--iters", "--iter", default=1000, type=int, help="iterations to tune each block")

    parser.add_argument(
        "--seqlen", "--seq_len", default=2048, type=int, help="sequence length of the calibration samples")

    parser.add_argument("--nsamples", "--nsample", default=512, type=int, help="number of samples")

    parser.add_argument(
        "--lr", default=None, type=float, help="learning rate, if None, it will be set to 1.0/iters automatically")

    args = parser.parse_args()
    args.low_gpu_mem_usage = True

    return args


def setup_light_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--group_size", default=128, type=int, help="group size")

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=8, type=int, help="train batch size")

    parser.add_argument("--iters", "--iter", default=50, type=int, help="iterations to tune each block")

    parser.add_argument(
        "--seqlen", "--seq_len", default=2048, type=int, help="sequence length of the calibration samples")

    parser.add_argument("--nsamples", "--nsample", default=128, type=int, help="number of samples")

    parser.add_argument(
        "--lr", default=5e-3, type=float, help="learning rate, if None, it will be set to 1.0/iters automatically")

    args = parser.parse_args()

    return args


def setup_fast_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--group_size", default=128, type=int, help="group size")

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=4, type=int, help="train batch size")

    parser.add_argument("--iters", default=200, type=int, help="iterations to tune each block")

    parser.add_argument(
        "--seqlen", "--seq_len", default=512, type=int, help="sequence length of the calibration samples")

    parser.add_argument("--nsamples", "--nsample", default=128, type=int, help="number of samples")

    parser.add_argument(
        "--lr", default=None, type=float, help="learning rate, if None, it will be set to 1.0/iters automatically")

    args = parser.parse_args()

    return args


def setup_eval_parser():
    parser = EvalArgumentParser()
    args = parser.parse_args()
    return args


def tune(args):
    if args.disable_eval:
        logging.warning("`disable_eval` is deprecated and is now set by default.")

    if args.eval_bs is None:
        args.eval_bs = "auto"

    import transformers

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig

    from auto_round.utils import detect_device, get_library_version
    from auto_round.utils import logger

    if args.format is None:
        args.format = "auto_round"

    formats = args.format.lower().replace(' ', '').split(",")
    from auto_round.utils import SUPPORTED_FORMATS
    for format in formats:
        if format not in SUPPORTED_FORMATS:
            raise ValueError(f"{format} is not supported, we only support {SUPPORTED_FORMATS}")

    if "auto_gptq" in args.format and args.asym is True:
        logger.warning("the auto_gptq kernel has issues with asymmetric quantization. "
                       "It is recommended to use sym quantization or --format='auto_round'")

    if "marlin" in args.format and args.asym is True:
        raise RuntimeError("marlin backend only supports sym quantization, please remove --asym")

    ##must set this before import torch
    set_cuda_visible_devices(args.device)
    device_str, use_auto_mapping = get_device_and_parallelism(args.device)

    import torch
    if not args.disable_deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # logger.info("`torch.use_deterministic_algorithms` is enabled by default for reproducibility "
        #             "and can be disabled using the `--disable_deterministic_algorithms` argument.")

    if args.enable_torch_compile:
        logger.info("`torch.compile` is enabled to reduce tuning costs. "
                    "If it causes issues, you can disable it by removing `--enable_torch_compile` argument.")

    model_name = args.model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    logger.info(f"start to quantize {model_name}")
    torch_dtype = "auto"
    if device_str is not None and "hpu" in device_str:
        torch_dtype = torch.bfloat16

    from auto_round.utils import llm_load_model
    model, tokenizer, low_cpu_mem_usage = llm_load_model(
        model_name,
        torch_dtype=torch_dtype,
        use_auto_mapping=use_auto_mapping,
        trust_remote_code=not args.disable_trust_remote_code,
        device=device_str,
        low_cpu_mem_mode=args.low_cpu_mem_mode,
        low_cpu_mem_tmp_dir=args.low_cpu_mem_tmp_dir,
        model_dtype=args.model_dtype)

    from auto_round import AutoRound, AutoRoundAdam

    if "bloom" in model_name:
        args.low_gpu_mem_usage = False

    round = AutoRound
    if args.adam:
        round = AutoRoundAdam

    layer_config = {}
    not_quantize_layer_names = get_fp_layer_names(model, args.fp_layers)
    for name in not_quantize_layer_names:
        layer_config[name] = {"bits": 16}
    if len(not_quantize_layer_names) > 0:
        logger.info(f"{not_quantize_layer_names} will not be quantized.")
        for format in formats:
            if "auto_round" not in format and "fake" not in format and "awq" not in format:
                ##TODO gptq could support some mixed precision config
                logger.warning(f"mixed precision exporting does not support {format} currently")

    lm_head_layer_name = "lm_head"
    for n, _ in model.named_modules():
        lm_head_layer_name = n
    if args.quant_lm_head:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code)
        if config.tie_word_embeddings and hasattr(model, "_tied_weights_keys"):
            tied_keys = model._tied_weights_keys
            for item in tied_keys:
                if lm_head_layer_name in item:  ##TODO extend to encoder-decoder layer, seq classification model
                    args.quant_lm_head = False
                    logger.warning(
                        f"reset `quant_lm_head` to `False` as quantizing lm_head with tied weights has not been "
                        f"supported currently")
                    break

    if args.quant_lm_head:
        layer_config[lm_head_layer_name] = {"bits": args.bits}
        for format in formats:
            if "auto_round" not in format and "fake" not in format:
                auto_round_formats = [s for s in SUPPORTED_FORMATS if s.startswith("auto_round")]
                raise ValueError(
                    f"{format} is not supported for lm-head quantization, please change to {auto_round_formats}")

    if "auto_awq" in args.format:
        from auto_round.utils import check_awq_gemm_compatibility
        awq_supported, info = check_awq_gemm_compatibility(
            model, args.bits, args.group_size, not args.asym, layer_config)
        if not awq_supported:
            logger.warning(f"The AutoAWQ format may not be supported due to {info}")

    enable_torch_compile = True if "--enable_torch_compile" in sys.argv else False

    autoround = round(
        model,
        tokenizer,
        args.bits,
        args.group_size,
        sym=not args.asym,
        batch_size=args.batch_size,
        dataset=args.dataset,
        seqlen=args.seqlen,
        nblocks=args.nblocks,
        iters=args.iters,
        lr=args.lr,
        minmax_lr=args.minmax_lr,
        enable_quanted_input=not args.disable_quanted_input,
        device=device_str,
        amp=not args.disable_amp,
        nsamples=args.nsamples,
        seed=args.seed,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
        scale_dtype=args.scale_dtype,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
        layer_config=layer_config,
        enable_minmax_tuning=not args.disable_minmax_tuning,
        act_bits=args.act_bits,
        low_cpu_mem_usage=low_cpu_mem_usage,
        data_type=args.data_type,
        enable_norm_bias_tuning=args.enable_norm_bias_tuning,
        not_use_best_mse=args.not_use_best_mse,
        to_quant_block_names=args.to_quant_block_names,
        enable_torch_compile=enable_torch_compile,
        act_data_type=args.act_data_type,
        act_dynamic=not args.disable_act_dynamic,
        device_map=args.device_map,
        super_group_size=args.super_group_size,
        super_bits=args.super_bits,
        disable_opt_rtn=args.disable_opt_rtn,
    )

    model_name = args.model.rstrip("/")

    if model_name.split('/')[-1].strip('.') == "" and "gguf" not in args.format:
        export_dir = os.path.join(args.output_dir, f"w{autoround.bits}g{autoround.group_size}")
    elif model_name.split('/')[-1].strip('.') == "" and "gguf" in args.format:
        export_dir = args.output_dir
    elif model_name.split('./')[-1].strip('./') != "" and "gguf" in args.format:
        export_dir = os.path.join(args.output_dir,
                                  model_name.split('/')[-1] + "-gguf")
    else:
        export_dir = os.path.join(args.output_dir,
                                  model_name.split('/')[-1] + f"-w{autoround.bits}g{autoround.group_size}")

    model, folders = autoround.quantize_and_save(export_dir, format=args.format)

    if args.low_cpu_mem_mode == 1 or args.low_cpu_mem_mode == 2:
        import shutil
        shutil.rmtree(args.low_cpu_mem_tmp_dir, ignore_errors=True)

    model.eval()
    clear_memory()

    lm_eval_version = get_library_version("lm-eval")

    eval_folder = folders[-1]
    if args.tasks is None or args.tasks == "" or eval_folder is None:
        return

    tasks = args.tasks
    if isinstance(tasks, str):
        tasks = tasks.split(',')

    from lm_eval.utils import make_table  # pylint: disable=E0401

    logger.info(f"Using lm-eval version {lm_eval_version}")
    eval_gguf_model = False
    for file in os.listdir(eval_folder):
        if file.endswith("gguf"):
            eval_gguf_model = True
            break

    import time
    eval_model_dtype = get_model_dtype(args.eval_model_dtype, "auto")
    if args.act_bits <= 8 or eval_gguf_model:
        if eval_gguf_model:
            # gguf floder only contains one file
            for file in os.listdir(eval_folder):
                gguf_file = file

            logger.warning("evaluate gguf model is an experimental feature, the accuracy may be not correct.")
            if eval_model_dtype == "float32" or eval_model_dtype == "auto":
                logger.warning(
                    "set '--eval_model_dtype bf16' can significantly speed up evaluation for gguf model,"
                    " but may affect accuracy."
                )
            model = AutoModelForCausalLM.from_pretrained(
                eval_folder, gguf_file=gguf_file, device_map="auto", torch_dtype=eval_model_dtype)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(eval_folder, gguf_file=gguf_file)
        else:
            if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                from accelerate.big_modeling import dispatch_model

                dispatch_model(model, model.hf_device_map)
            else:
                device_str = detect_device(device_str)
                model = model.to(device_str)
            if model.dtype != eval_model_dtype and eval_model_dtype != "auto":
                model.to(getattr(torch, eval_model_dtype))

        if args.eval_task_by_task:
            eval_task_by_task(
                model,
                tokenizer=tokenizer,
                device=device_str,
                tasks=args.tasks,
                batch_size=args.eval_bs,
                eval_model_dtype=eval_model_dtype)
        else:
            if args.eval_bs is None or args.eval_bs == "auto":
                logger.warning("This API does not support auto currently, reset eval_bs to 16")
                args.eval_bs = 16
            from auto_round.eval.evaluation import simple_evaluate_user_model
            st = time.time()
            res = simple_evaluate_user_model(
                model,
                tokenizer,
                tasks=tasks,
                batch_size=args.eval_bs,
                device=device_str,
                eval_model_dtype=eval_model_dtype)
            print(make_table(res))
            print("evaluation running time=%ds" % (time.time() - st))
    else:
        if args.eval_task_by_task:
            eval_task_by_task(
                eval_folder,
                device=device_str,
                tasks=args.tasks,
                batch_size=args.eval_bs,
                eval_model_dtype=eval_model_dtype)
        else:
            from auto_round.eval.evaluation import simple_evaluate
            tasks, model_args, device_str = _eval_init(
                args.tasks, eval_folder, args.device, args.disable_trust_remote_code, dtype=eval_model_dtype)
            st = time.time()
            res = simple_evaluate(
                model="hf", model_args=model_args, tasks=tasks, device=device_str, batch_size=args.eval_bs)
            print(make_table(res))
            print("evaluation running time=%ds" % (time.time() - st))


def _eval_init(tasks, model_path, device, disable_trust_remote_code=False, dtype="auto"):
    set_cuda_visible_devices(device)
    device_str, parallelism = get_device_and_parallelism(device)
    if dtype != "auto":
        dtype = get_model_dtype(model_dtype=dtype)
    # ,add_bos_token={True}
    model_args = f'pretrained={model_path},trust_remote_code={not disable_trust_remote_code},dtype={dtype}'
    if parallelism:
        model_args += ",parallelize=True"
    if isinstance(tasks, str):
        tasks = tasks.split(',')
    return tasks, model_args, device_str


def eval(args):
    import time
    tasks, model_args, device_str = _eval_init(
        args.tasks, args.model, args.device, args.disable_trust_remote_code, args.eval_model_dtype)

    # load after _eval_int in order to make sure import torch after set CUDA_VISBILE_DEVICES
    from auto_round.eval.evaluation import simple_evaluate, simple_evaluate_user_model
    from auto_round.utils import logger

    if (batch_size := args.eval_bs) is None:
        batch_size = "auto:8"
    is_gguf_file = False
    if os.path.isfile(args.model) and args.model.endswith(".gguf"):
        is_gguf_file = True
        gguf_file = os.path.basename(args.model)
        model = os.path.dirname(args.model)
    else:
        for file in os.listdir(args.model):
            if file.endswith(".gguf"):
                is_gguf_file = True
                gguf_file = file
    eval_model_dtype = get_model_dtype(args.eval_model_dtype)
    if is_gguf_file:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from lm_eval.utils import make_table  # pylint: disable=E0401
        tokenizer = AutoTokenizer.from_pretrained(model, gguf_file=gguf_file)

        logger.warning("evaluate gguf model is an experimental feature, the accuracy may be not correct.")
        if eval_model_dtype == "float32" or eval_model_dtype == "auto":
            logger.warning(
                "set '--eval_model_dtype bf16' can significantly speed up evaluation for gguf model,"
                " but may affect accuracy."
            )
        model = AutoModelForCausalLM.from_pretrained(
            model, gguf_file=gguf_file, device_map="auto", torch_dtype=eval_model_dtype)
        model.eval()
        st = time.time()
        res = simple_evaluate_user_model(
            model, tokenizer, tasks=tasks, batch_size=batch_size, device=device_str)
        print(make_table(res))
        print("evaluation running time=%ds" % (time.time() - st))
    else:
        st = time.time()
        res = simple_evaluate(
            model="hf", model_args=model_args, tasks=tasks, device=device_str, batch_size=batch_size)
        from lm_eval.utils import make_table  # pylint: disable=E0401
        print(make_table(res))
        print("evaluation running time=%ds" % (time.time() - st))


def eval_task_by_task(
        model,
        device=None,
        tasks=None,
        tokenizer=None,
        batch_size=None,
        max_batch_size=64,
        trust_remote_code=True,
        eval_model_dtype=None,
        retry_times=3):
    set_cuda_visible_devices(device)
    device_str, parallelism = get_device_and_parallelism(device)

    # load after _eval_int in order to make sure import torch after set CUDA_VISBILE_DEVICES
    import traceback
    from auto_round.utils import logger
    from lm_eval import simple_evaluate as lm_simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from auto_round import AutoRoundConfig  # pylint: disable=E0611
    if batch_size is None:
        batch_size = "auto:8"
    is_gguf_file = False
    if not isinstance(model, str):
        parallelism = False
    else:
        if os.path.isfile(model) and model.endswith(".gguf"):
            is_gguf_file = True
            gguf_file = os.path.basename(model)
            model = os.path.dirname(model)
        else:
            for file in os.listdir(model):
                if file.endswith(".gguf"):
                    is_gguf_file = True
                    gguf_file = file
    eval_model_dtype = get_model_dtype(eval_model_dtype)
    if is_gguf_file:
        tokenizer = AutoTokenizer.from_pretrained(model, gguf_file=gguf_file)
        logger.warning("evaluate gguf model is an experimental feature, the accuracy may be not correct.")
        if eval_model_dtype == "float32" or eval_model_dtype == "auto":
            logger.warning(
                "set '--eval_model_dtype bf16' can significantly speed up evaluation for gguf model,"
                " but may affect accuracy."
            )

        model = AutoModelForCausalLM.from_pretrained(
            model, gguf_file=gguf_file, device_map="auto", torch_dtype=eval_model_dtype)
        model.eval()
        parallelism = False
    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=device_str,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        parallelize=parallelism,
        trust_remote_code=trust_remote_code,
        dtype=eval_model_dtype
    )

    if isinstance(tasks, str):
        tasks = tasks.replace(" ", "").split(",")

    from lm_eval.utils import make_table  # pylint: disable=E0401
    res_all = {}
    res_keys = ["results", "versions", "n-shot", "higher_is_better"]
    import time
    st = time.time()
    for task in tasks:
        while retry_times:
            try:
                res = lm_simple_evaluate(
                    model=hflm, model_args=None, device=device_str, tasks=task, batch_size=batch_size)
                break
            except Exception as e:
                if "CUDA out of memory" in str(e) or "MODULE:PT_DEVMEM" in str(e):
                    ori_batch_sizes = hflm.batch_sizes if hflm.batch_sizes else {"0": 64}
                    try:
                        for k, v in hflm.batch_sizes.items():
                            hflm.batch_sizes[k] = max(v // 2, 1)
                        logger.warning(
                            f"Out of memory, reset batch_size to {hflm.batch_sizes} and re-try.")
                        res = lm_simple_evaluate(
                            model=hflm, model_args=None, device=device_str, tasks=task, batch_size=1)
                        hflm.batch_sizes = ori_batch_sizes
                    except Exception as e:
                        traceback.print_exc()
                        pass
                else:
                    traceback.print_exc()
                    break
            retry_times -= 1

        if not res_all:
            res_all = res
        else:
            for key in res_keys:
                res_all[key].update(res[key])
        print(make_table(res_all))

    print("total eval time:", time.time() - st)
