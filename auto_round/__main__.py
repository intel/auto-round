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
import argparse
import logging
import os
import re
import sys

from auto_round.compressors import BaseCompressor
from auto_round.eval.eval_cli import EvalArgumentParser, _eval_init, eval, eval_task_by_task, eval_with_vllm
from auto_round.schemes import PRESET_SCHEMES
from auto_round.utils import (
    clear_memory,
    get_device_and_parallelism,
    get_model_dtype,
    set_cuda_visible_devices,
)


class BasicArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--model", "--model_name", "--model_name_or_path", default="facebook/opt-125m", help="model name or path"
        )

        self.add_argument("--mllm", action="store_true", help="whether to quant multi-modal model.")

        self.add_argument("--eval", action="store_true", help="whether to use eval only mode")

        self.add_argument(
            "--scheme",
            default="W4A16",
            type=str,
            # choices=["W4A16", "W2A16", "W3A16", "W8A16", "MXFP4", "MXFP8", "NVFP4", "FPW8A16", "FP8_STATIC"],
            help="quantization scheme",
        )

        self.add_argument("--bits", default=None, type=int, help="number of weight bits")
        self.add_argument("--group_size", default=None, type=int, help="group size")
        self.add_argument("--asym", action="store_true", help="whether to use asym quantization")
        self.add_argument("--data_type", "--dtype", default=None, help="data type for tuning, 'int', 'mx_fp' and etc")
        self.add_argument("--act_bits", default=None, type=int, help="activation bits")
        self.add_argument("--act_group_size", default=None, type=int, help="activation group size")
        self.add_argument(
            "--super_group_size", default=None, type=int, help="the number of super group size when use double quant."
        )

        self.add_argument(
            "--super_bits", default=None, type=int, help="number of scale and mins quant bits for double quant."
        )
        self.add_argument("--act_data_type", "--act_dtype", default=None, type=str, help="activation data type")

        self.add_argument("--disable_act_dynamic", action="store_true", help="activation static quantization")

        self.add_argument(
            "--device_map",
            "--device",
            "--devices",
            default="0",
            type=str,
            help="the device to be used for tuning. "
            "Currently, device settings support CPU, GPU, and HPU."
            "The default is set to cuda:0,"
            "allowing for automatic detection and switch to HPU or CPU."
            "set --device 0,1,2 to use multiple cards.",
        )

        self.add_argument(
            "--dataset", default="NeelNanda/pile-10k", type=str, help="the dataset for quantization training"
        )

        self.add_argument(
            "--minmax_lr",
            default=None,
            type=float,
            help="minmax learning rate, if None, it will beset to be the same with lr",
        )

        self.add_argument(
            "--mem_per_param_scale",
            default=13,
            type=float,
            help="Scale factor for memory per parameter, used to adjust memory usage estimation for tuning",
        )

        self.add_argument("--seed", default=42, type=int, help="random seed")

        self.add_argument("--adam", action="store_true", help="whether to use adam optimizer instead of SignSGD")

        self.add_argument("--gradient_accumulate_steps", default=1, type=int, help="gradient accumulate steps")

        self.add_argument("--nblocks", default=1, type=int, help="how many blocks to tune together")

        self.add_argument("--low_gpu_mem_usage", action="store_true", help="offload intermediate features to cpu")

        self.add_argument("--format", default="auto_round", type=str, help="the format to save the model")

        self.add_argument(
            "--scale_dtype",
            default="fp16",
            choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"],
            help="scale data type to use for quantization",
        )

        self.add_argument(
            "--output_dir", default="./tmp_autoround", type=str, help="the directory to save quantized model"
        )

        self.add_argument("--disable_amp", action="store_true", help="disable amp")

        self.add_argument(
            "--disable_minmax_tuning", action="store_true", help="whether to disable enable weight minmax tuning"
        )

        self.add_argument("--enable_norm_bias_tuning", action="store_true", help="whether to enable norm bias tuning")

        self.add_argument(
            "--disable_trust_remote_code", action="store_true", help="whether to disable trust_remote_code"
        )

        self.add_argument(
            "--disable_quanted_input",
            action="store_true",
            help="whether to disuse the output of quantized block to tune the next block",
        )

        self.add_argument("--quant_lm_head", action="store_true", help="whether to quant lm_head")

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
            "others means not use low cpu memory. Default to 0, not use low cpu memory.",
        )

        self.add_argument(
            "--low_cpu_mem_tmp_dir",
            default=None,
            type=str,
            help="temporary work space to store the temporary files "
            "when using low cpu memory mode. Will remove after tuning.",
        )

        self.add_argument(
            "--model_dtype",
            default=None,
            type=str,
            choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"],
            help="force to convert the dtype, some backends supports fp16 dtype better",
        )

        self.add_argument(
            "--fp_layers", default="", type=str, help="list of Layer names to maintain original data type"
        )

        self.add_argument(
            "--not_use_best_mse",
            action="store_true",
            help="whether to use the iter of best mes loss in the tuning phase",
        )

        self.add_argument(
            "--to_quant_block_names",
            default=None,
            type=str,
            help="Names of quantitative blocks, please use commas to separate them.",
        )

        self.add_argument("--enable_torch_compile", action="store_true", help="whether to enable torch compile")

        self.add_argument("--enable_alg_ext", action="store_true", help="whether to enable probably better algorithm")

        self.add_argument(
            "--disable_deterministic_algorithms",
            action="store_true",
            help="deprecated, disable torch deterministic algorithms.",
        )
        self.add_argument(
            "--enable_deterministic_algorithms", action="store_true", help="enable torch deterministic algorithms."
        )

        self.add_argument(
            "--disable_opt_rtn",
            action="store_true",
            help="whether to disable optimization of the RTN mode(iters=0) (default is False).",
        )

        ## ======================= MLLM =======================
        self.add_argument(
            "--quant_nontext_module",
            action="store_true",
            help="whether to quantize non-text module, e.g. vision component",
        )

        self.add_argument(
            "--extra_data_dir",
            default=None,
            type=str,
            help="dataset dir for storing images/audio/videos. "
            "Can be a dir path or multiple dir path with format as "
            "'image=path_to_image,video=path_to_video,audio=path_to_audio'"
            "By default, it will search in the relative path, "
            "and if not find, will automatic download.",
        )

        self.add_argument(
            "--template",
            default=None,
            type=str,
            help="the template for building training dataset. It can be a custom one.",
        )

        ## ======================= eval =======================
        self.add_argument(
            "--tasks",
            "--task",
            nargs="?",
            const="lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1,"
            "openbookqa,boolq,arc_easy,arc_challenge",
            default=None,
            help="lm-eval tasks",
        )

        self.add_argument("--eval_bs", default=None, type=int, help="batch size in evaluation")

        self.add_argument(
            "--limit",
            type=float,
            default=None,
            metavar="N|0<N<1",
            help="Limit the number of examples per task. "
            "If <1, limit is a percentage of the total number of examples.",
        )

        self.add_argument("--eval_task_by_task", action="store_true", help="whether to eval task by task.")

        self.add_argument(
            "--eval_model_dtype", default=None, type=str, help="the torch_dytpe to load the model for evaluation."
        )


def setup_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=8, type=int, help="train batch size")

    parser.add_argument("--iters", "--iter", default=200, type=int, help="iteration to tune each block")

    parser.add_argument(
        "--seqlen", "--seq_len", default=2048, type=int, help="sequence length of the calibration samples"
    )

    parser.add_argument("--nsamples", "--nsample", default=128, type=int, help="number of samples")

    parser.add_argument(
        "--lr", default=None, type=float, help="learning rate, if None, it will be set to 1.0/iters automatically"
    )

    args = parser.parse_args()
    return args


def setup_best_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=8, type=int, help="train batch size")

    parser.add_argument("--iters", "--iter", default=1000, type=int, help="iterations to tune each block")

    parser.add_argument(
        "--seqlen", "--seq_len", default=2048, type=int, help="sequence length of the calibration samples"
    )

    parser.add_argument("--nsamples", "--nsample", default=512, type=int, help="number of samples")

    parser.add_argument(
        "--lr", default=None, type=float, help="learning rate, if None, it will be set to 1.0/iters automatically"
    )

    args = parser.parse_args()
    args.low_gpu_mem_usage = True

    return args


def setup_light_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=8, type=int, help="train batch size")

    parser.add_argument("--iters", "--iter", default=50, type=int, help="iterations to tune each block")

    parser.add_argument(
        "--seqlen", "--seq_len", default=2048, type=int, help="sequence length of the calibration samples"
    )

    parser.add_argument("--nsamples", "--nsample", default=128, type=int, help="number of samples")

    parser.add_argument(
        "--lr", default=5e-3, type=float, help="learning rate, if None, it will be set to 1.0/iters automatically"
    )

    args = parser.parse_args()

    return args


def setup_fast_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=4, type=int, help="train batch size")

    parser.add_argument("--iters", default=200, type=int, help="iterations to tune each block")

    parser.add_argument(
        "--seqlen", "--seq_len", default=512, type=int, help="sequence length of the calibration samples"
    )

    parser.add_argument("--nsamples", "--nsample", default=128, type=int, help="number of samples")

    parser.add_argument(
        "--lr", default=None, type=float, help="learning rate, if None, it will be set to 1.0/iters automatically"
    )

    args = parser.parse_args()

    return args


def setup_eval_parser():

    parser = EvalArgumentParser()
    args = parser.parse_args()
    return args


def tune(args):
    if args.eval_bs is None:
        args.eval_bs = "auto"
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from auto_round.utils import detect_device, get_library_version, logger

    if args.format is None:
        args.format = "auto_round"

    formats = args.format.lower().replace(" ", "").split(",")
    from auto_round.utils import SUPPORTED_FORMATS

    for format in formats:
        if format not in SUPPORTED_FORMATS:
            raise ValueError(f"{format} is not supported, we only support {SUPPORTED_FORMATS}")

    if "auto_gptq" in args.format and args.asym is True:
        logger.warning(
            "the auto_gptq kernel has issues with asymmetric quantization. "
            "It is recommended to use sym quantization or --format='auto_round'"
        )

    if "marlin" in args.format and args.asym is True:
        raise RuntimeError("marlin backend only supports sym quantization, please remove --asym")

    # Must set this before import torch
    # set_cuda_visible_devices(args.device_map)
    device_str, use_auto_mapping = get_device_and_parallelism(args.device_map)

    import torch

    if args.enable_torch_compile:
        logger.info(
            "`torch.compile` is enabled to reduce tuning costs. "
            "If it causes issues, you can disable it by removing `--enable_torch_compile` argument."
        )

    model_name = args.model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    logger.info(f"start to quantize {model_name}")

    from auto_round import AutoRound

    if "bloom" in model_name:
        args.low_gpu_mem_usage = False

    if args.quant_lm_head:
        for format in formats:
            if "auto_round" not in format and "fake" not in format:
                auto_round_formats = [s for s in SUPPORTED_FORMATS if s.startswith("auto_round")]
                raise ValueError(
                    f"{format} is not supported for lm-head quantization, please change to {auto_round_formats}"
                )

    enable_torch_compile = True if "--enable_torch_compile" in sys.argv else False
    sym = None  # the default value should be None now
    if args.asym:  # if the scheme is asym, how to set it to sym is an issue
        sym = False
    act_dynamic = None
    if args.disable_act_dynamic:
        act_dynamic = False
    scheme = args.scheme.upper()
    if scheme not in PRESET_SCHEMES:
        raise ValueError(f"{scheme} is not supported. only {PRESET_SCHEMES.keys()} are supported ")
    if args.disable_deterministic_algorithms:
        logger.warning(
            "default not use deterministic_algorithms. disable_deterministic_algorithms is deprecated,"
            " please use enable_deterministic_algorithms instead. "
        )

    from auto_round.compressors import (
        ExtraConfig,
        MLLMExtraConfig,
        SchemeExtraConfig,
        TuningExtraConfig,
    )

    extra_config = ExtraConfig()
    tuning_config = TuningExtraConfig(
        amp=not args.disable_amp,
        disable_opt_rtn=args.disable_opt_rtn,
        enable_alg_ext=args.enable_alg_ext,
        enable_minmax_tuning=not args.disable_minmax_tuning,
        enable_norm_bias_tuning=args.enable_norm_bias_tuning,
        enable_quanted_input=not args.disable_quanted_input,
        enable_deterministic_algorithms=args.enable_deterministic_algorithms,
        lr=args.lr,
        minmax_lr=args.minmax_lr,
        mem_per_param_scale=args.mem_per_param_scale,
        nblocks=args.nblocks,
        to_quant_block_names=args.to_quant_block_names,
        scale_dtype=args.scale_dtype,
    )
    scheme_config = SchemeExtraConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=sym,
        data_type=args.data_type,
        act_bits=args.act_bits,
        act_group_size=args.act_group_size,
        act_data_type=args.act_data_type,
        act_dynamic=act_dynamic,
        super_bits=args.super_bits,
        super_group_size=args.super_group_size,
        quant_lm_head=args.quant_lm_head,
        fp_layers=args.fp_layers,
    )
    mllm_config = MLLMExtraConfig(
        quant_nontext_module=args.quant_nontext_module, extra_data_dir=args.extra_data_dir, template=args.template
    )
    extra_config.tuning_config = tuning_config
    extra_config.scheme_config = scheme_config
    extra_config.mllm_config = mllm_config

    autoround: BaseCompressor = AutoRound(
        model=model_name,
        scheme=scheme,
        dataset=args.dataset,
        iters=args.iters,
        seqlen=args.seqlen,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
        device_map=args.device_map,
        enable_torch_compile=enable_torch_compile,
        seed=args.seed,
        not_use_best_mse=args.not_use_best_mse,
        enable_adam=args.adam,
        extra_config=extra_config,
    )

    model_name = args.model.rstrip("/")

    if model_name.split("/")[-1].strip(".") == "" and "gguf" not in args.format:
        if autoround.group_size <= 0:
            if "fp" in autoround.act_data_type:
                suffix = f"afp{autoround.act_bits}"
            else:
                suffix = f"a{autoround.act_bits}"
        else:
            suffix = f"g{autoround.group_size}"
        export_dir = os.path.join(args.output_dir, f"w{autoround.bits}{suffix}")
    elif model_name.split("/")[-1].strip(".") == "" and "gguf" in args.format:
        export_dir = args.output_dir
    elif model_name.split("./")[-1].strip("./") != "" and "gguf" in args.format:
        export_dir = os.path.join(args.output_dir, model_name.split("/")[-1] + "-gguf")
    else:
        if autoround.group_size <= 0:
            if "fp" in autoround.act_data_type:
                suffix = f"afp{autoround.act_bits}"
            else:
                suffix = f"a{autoround.act_bits}"
        else:
            suffix = f"g{autoround.group_size}"
        export_dir = os.path.join(args.output_dir, model_name.split("/")[-1] + f"-w{autoround.bits}{suffix}")

    model, folders = autoround.quantize_and_save(export_dir, format=args.format)  # pylint: disable=E1101
    tokenizer = autoround.tokenizer  # pylint: disable=E1101

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
        tasks = tasks.split(",")

    from lm_eval.utils import make_table  # pylint: disable=E0401

    logger.info(f"Using lm-eval version {lm_eval_version}")
    eval_gguf_model = False
    for file in os.listdir(eval_folder):
        if file.endswith("gguf"):
            eval_gguf_model = True
            break

    import time

    eval_model_dtype = get_model_dtype(args.eval_model_dtype, "auto")

    if (autoround.act_bits <= 8 and formats[-1] == "fake") or eval_gguf_model:
        if eval_gguf_model:
            # for file in os.listdir(eval_folder):
            #     gguf_file = file
            gguf_file = None
            gguf_format = None  # Initialize gguf_format to None
            # gguf folder only contains one file
            for format in formats:
                if format.startswith("gguf"):
                    gguf_format = format.split(":")[-1].upper()
            if gguf_format is None:  # Validate gguf_format after the loop
                logger.error("No valid gguf format found in formats. Please check the input.")
                sys.exit(-1)
            for file in os.listdir(eval_folder):
                if gguf_format in file:
                    gguf_file = file

            logger.warning("evaluating gguf model is an experimental feature, the accuracy may be not correct.")
            if eval_model_dtype == "float32" or eval_model_dtype == "auto":
                logger.warning(
                    "set '--eval_model_dtype bf16' can significantly speed up evaluation for gguf model,"
                    " but may affect accuracy."
                )
            if gguf_file is None:
                logger.error("Cannot find correct gguf file for evaluation, please check.")
                sys.exit(-1)
            model = AutoModelForCausalLM.from_pretrained(
                eval_folder, gguf_file=gguf_file, device_map="auto", torch_dtype=eval_model_dtype
            )
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
                limit=args.limit,
                batch_size=args.eval_bs,
                eval_model_dtype=eval_model_dtype,
            )
        else:
            if args.eval_bs is None or args.eval_bs == "auto":
                logger.warning("This API does not support auto currently, reset eval_bs to 16")
                args.eval_bs = 16
            from auto_round.eval.evaluation import simple_evaluate_user_model

            st = time.time()
            add_bos_token = False
            if "llama" in args.model.lower():
                add_bos_token = True
            res = simple_evaluate_user_model(
                model,
                tokenizer,
                tasks=tasks,
                batch_size=args.eval_bs,
                limit=args.limit,
                device=device_str,
                eval_model_dtype=eval_model_dtype,
                add_bos_token=add_bos_token,
            )
            print(make_table(res))
            print("evaluation running time=%ds" % (time.time() - st))
    else:
        if args.eval_task_by_task:
            eval_task_by_task(
                eval_folder,
                device=device_str,
                tasks=args.tasks,
                batch_size=args.eval_bs,
                limit=args.limit,
                eval_model_dtype=eval_model_dtype,
            )
        else:
            from auto_round.eval.evaluation import simple_evaluate

            tasks, model_args, device_str = _eval_init(
                args.tasks, eval_folder, args.device_map, args.disable_trust_remote_code, dtype=eval_model_dtype
            )
            st = time.time()
            if "llama" in args.model.lower():
                model_args += ",add_bos_token=True"
            res = simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=tasks,
                device=device_str,
                batch_size=args.eval_bs,
                limit=args.limit,
            )
            print(make_table(res))
            print("evaluation running time=%ds" % (time.time() - st))


def run_eval():
    args = setup_eval_parser()
    if args.eval_task_by_task:
        eval_task_by_task(
            model=args.model,
            device=args.device,
            tasks=args.tasks,
            batch_size=args.eval_bs,
            trust_remote_code=not args.disable_trust_remote_code,
            eval_model_dtype=args.eval_model_dtype,
        )
    else:
        eval(args)


def run():
    if "--eval" in sys.argv:
        sys.argv.remove("--eval")
        run_eval()
    else:
        args = setup_parser()
        tune(args)


def run_mllm():
    sys.argv.append("--mllm")
    run()


def run_best():
    args = setup_best_parser()
    tune(args)


def run_light():
    args = setup_light_parser()
    tune(args)


def run_fast():
    args = setup_fast_parser()
    tune(args)


if __name__ == "__main__":
    run()
