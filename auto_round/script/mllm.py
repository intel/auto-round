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
import os
import sys

from auto_round.logger import logger
from auto_round.schemes import PRESET_SCHEMES
from auto_round.utils import (
    clear_memory,
    get_device_and_parallelism,
    get_fp_layer_names,
    is_debug_mode,
    set_cuda_visible_devices,
)


class BasicArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--model",
            "--model_name",
            "--model_name_or_path",
            default="Qwen/Qwen2-VL-2B-Instruct",
            help="model name or path",
        )

        self.add_argument("--eval", action="store_true", help="whether to use eval only mode.")

        self.add_argument(
            "--scheme",
            default="W4A16",
            type=str,
            help="quantization scheme",
        )

        self.add_argument("--bits", default=None, type=int, help="number of weight bits")
        self.add_argument("--group_size", default=None, type=int, help="group size")
        self.add_argument("--asym", action="store_true", help="whether to use asym quantization")
        self.add_argument("--data_type", "--dtype", default=None, help="data type for tuning, 'int', 'mx_fp' and etc")
        self.add_argument("--act_bits", default=None, type=int, help="activation bits")

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
            "set --device 0,1,2 to use multiple cards.",
        )

        self.add_argument(
            "--dataset",
            type=str,
            default=None,
            help="the dataset for quantization training."
            " current support NeelNanda/pile-10k,liuhaotian/llava_conv_58k,"
            "liuhaotian/llava_instruct_80k,liuhaotian/llava_instruct_150k"
            "It can be a custom one. Default is NeelNanda/pile-10k",
        )

        self.add_argument(
            "--lr", default=None, type=float, help="learning rate, if None, it will be set to 1.0/iters automatically"
        )

        self.add_argument(
            "--minmax_lr",
            default=None,
            type=float,
            help="minmax learning rate, if None,it will beset to be the same with lr",
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
            "--disable_minmax_tuning", action="store_true", help="whether disable enable weight minmax tuning"
        )

        self.add_argument("--enable_norm_bias_tuning", action="store_true", help="whether enable norm bias tuning")

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

        self.add_argument("--fp_layers", default="", type=str, help="layers to maintain original data type")

        self.add_argument(
            "--not_use_best_mse",
            action="store_true",
            help="whether to use the iter of best mes loss in the tuning phase",
        )

        self.add_argument("--enable_torch_compile", action="store_true", help="whether to enable torch compile")

        self.add_argument(
            "--disable_deterministic_algorithms",
            action="store_true",
            help="deprecated, disable torch deterministic algorithms.",
        )

        self.add_argument(
            "--enable_deterministic_algorithms", action="store_true", help="enable torch deterministic algorithms."
        )

        ## ======================= VLM =======================
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

        self.add_argument(
            "--truncation",
            action="store_true",
            help="whether to truncate sequences at the maximum length."
            " Default True for pile and False for llava dataset.",
        )

        self.add_argument(
            "--to_quant_block_names",
            default=None,
            type=str,
            help="Names of quantitative blocks, please use commas to separate them.",
        )

        self.add_argument("--device_map", default=None, type=str, help="device_map for block in tuning phase")

        self.add_argument(
            "--disable_opt_rtn",
            action="store_true",
            help="whether to disable optimization of the RTN mode(iters=0) (default is False).",
        )


def setup_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=8, type=int, help="train batch size")

    parser.add_argument("--iters", "--iter", default=200, type=int, help=" iters")

    parser.add_argument(
        "--seqlen",
        "--seq_len",
        default=None,
        type=int,
        help="sequence length, default 2048 for text-only, 512 for liuhaotian/llava",
    )

    parser.add_argument("--nsamples", "--nsample", default=128, type=int, help="number of samples")

    args = parser.parse_args()
    return args


def setup_lmeval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "--model_name", "--model_name_or_path", help="model name or path")
    parser.add_argument(
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
    parser.add_argument(
        "--tasks",
        type=str,
        default="MMBench_DEV_EN_V11,ScienceQA_VAL,TextVQA_VAL,POPE",
        help="eval tasks for VLMEvalKit.",
    )
    # Args that only apply to Video Dataset
    parser.add_argument(
        "--nframe",
        type=int,
        default=8,
        help="the number of frames to sample from a video," " only applicable to the evaluation of video benchmarks.",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help="a video may associate with multiple questions, if pack==True,"
        " will ask all questions for a video in a single",
    )
    parser.add_argument("--fps", type=float, default=-1, help="set the fps for a video.")
    # Work Dir
    # Infer + Eval or Infer Only
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "infer"],
        help="when mode set to 'all', will perform both inference and evaluation;"
        " when set to 'infer' will only perform the inference.",
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default=None,
        help="path for VLMEvalKit to store the eval data. Default will store in ~/LMUData",
    )
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument("--retry", type=int, default=None, help="retry numbers for API VLMs")
    # Explicitly Set the Judge Model
    parser.add_argument("--judge", type=str, default=None, help="whether is a judge model.")
    # Logging Utils
    parser.add_argument("--verbose", action="store_true", help="whether to display verbose information.")
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument("--ignore", action="store_true", help="ignore failed indices. ")
    # Rerun: will remove all evaluation temp files
    parser.add_argument(
        "--rerun", action="store_true", help="if true, will remove all evaluation temp files and rerun."
    )
    parser.add_argument("--output_dir", default="./eval_result", type=str, help="the directory to save quantized model")
    args = parser.parse_args()
    return args


def tune(args):
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    if args.format is None:
        args.format = "auto_round"

    formats = args.format.replace(" ", "").split(",")
    from auto_round.utils import SUPPORTED_FORMATS

    for format in formats:
        if format not in SUPPORTED_FORMATS:
            raise ValueError(f"{format} is not supported, we only support {SUPPORTED_FORMATS}")

    # Must set this before import torch
    # set_cuda_visible_devices(args.device_map)
    device_str, use_auto_mapping = get_device_and_parallelism(args.device_map)

    import torch

    model_name = args.model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    logger.info(f"start to quantize {model_name}")
    torch_dtype = "auto"
    if device_str is not None and "hpu" in device_str:
        torch_dtype = torch.bfloat16

    # load_model
    from auto_round.utils import mllm_load_model

    model, processor, tokenizer, image_processor = mllm_load_model(
        model_name,
        device="cpu",
        torch_dtype=torch_dtype,
        use_auto_mapping=False,
        trust_remote_code=not args.disable_trust_remote_code,
        model_dtype=args.model_dtype,
    )

    from auto_round import AutoRoundMLLM

    seqlen = args.seqlen
    if seqlen is not None and hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        seqlen = min(seqlen, model.config.max_position_embeddings)

    if seqlen is not None and hasattr(tokenizer, "model_max_length"):
        if tokenizer.model_max_length < seqlen:
            logger.info(
                f"change sequence length to {tokenizer.model_max_length} due to the limitation of model_max_length"
            )
            seqlen = min(seqlen, tokenizer.model_max_length)
            args.seqlen = seqlen

    model = model.eval()

    round = AutoRoundMLLM

    layer_config = {}
    not_quantize_layer_names = get_fp_layer_names(model, args.fp_layers)
    for name in not_quantize_layer_names:
        layer_config[name] = {"bits": 16, "act_bits": 16}
    if len(not_quantize_layer_names) > 0:
        logger.info(f"{not_quantize_layer_names} will not be quantized.")
        for format in formats:
            if "auto_round" not in format and "fake" not in format and "awq" not in format:
                ##TODO gptq could support some mixed precision config
                logger.warning(f"mixed precision exporting does not support {format} currently")

    if args.fp_layers != "":
        fp_layers = args.fp_layers.replace(" ", "").split(",")
        for n, m in model.named_modules():
            if not isinstance(m, (torch.nn.Linear, transformers.pytorch_utils.Conv1D)):
                continue
            for fp_layer in fp_layers:
                if fp_layer in n:
                    layer_config[n] = {"bits": 16}
                    logger.info(f"{n} will not be quantized.")
        if len(layer_config) > 0:
            for format in formats:
                if "auto_round" not in format and "fake" not in format:
                    ##TODO gptq, awq could support some mixed precision config
                    logger.warning(f"mixed precision exporting does not support {format} currently")

    if args.quant_lm_head:
        for format in formats:
            if "auto_round" not in format and "fake" not in format:
                auto_round_formats = [s for s in SUPPORTED_FORMATS if s.startswith("auto_round")]
                raise ValueError(
                    f"{format} is not supported for lm-head quantization, please change to {auto_round_formats}"
                )

    if args.quant_lm_head and args.low_gpu_mem_usage:
        print("warning, low_gpu_mem_usage=False is strongly recommended if the whole model could be loaded to " "gpu")

    if "--truncation" not in sys.argv:
        args.truncation = None

    if "auto_awq" in args.format:
        from auto_round.utils import check_awq_gemm_compatibility

        awq_supported, info = check_awq_gemm_compatibility(
            model, args.bits, args.group_size, not args.asym, layer_config
        )
        if not awq_supported:
            logger.warning(f"The AutoAWQ format may not be supported due to {info}")

    enable_torch_compile = True if "--enable_torch_compile" in sys.argv else False

    model_kwargs = {
        "use_auto_mapping": use_auto_mapping,
        "trust_remote_code": not args.disable_trust_remote_code,
        "model_dtype": args.model_dtype,
    }

    sym = None  # the default value should be None now
    if args.asym:  # if the scheme is asym, how to set it to sym is an issue
        sym = False

    scheme = args.scheme.upper()
    if scheme not in PRESET_SCHEMES:
        raise ValueError(f"{scheme} is not supported. only {PRESET_SCHEMES.keys()} are supported ")
    if args.disable_deterministic_algorithms:
        logger.warning(
            "default not use deterministic_algorithms. disable_deterministic_algorithms is deprecated,"
            " please use enable_deterministic_algorithms instead. "
        )
    autoround = round(
        model,
        tokenizer,
        scheme=args.scheme,
        processor=processor,
        image_processor=image_processor,
        dataset=args.dataset,
        extra_data_dir=args.extra_data_dir,
        bits=args.bits,
        group_size=args.group_size,
        sym=sym,
        batch_size=args.batch_size,
        seqlen=seqlen,
        nblocks=args.nblocks,
        iters=args.iters,
        lr=args.lr,
        minmax_lr=args.minmax_lr,
        amp=not args.disable_amp,
        enable_quanted_input=not args.disable_quanted_input,
        truncation=args.truncation,
        nsamples=args.nsamples,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
        seed=args.seed,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
        scale_dtype=args.scale_dtype,
        layer_config=layer_config,
        template=args.template,
        enable_minmax_tuning=not args.disable_minmax_tuning,
        act_bits=args.act_bits,
        quant_nontext_module=args.quant_nontext_module,
        not_use_best_mse=args.not_use_best_mse,
        to_quant_block_names=args.to_quant_block_names,
        enable_torch_compile=enable_torch_compile,
        device_map=args.device_map,
        model_kwargs=model_kwargs,
        data_type=args.data_type,
        disable_opt_rtn=args.disable_opt_rtn,
        enable_deterministic_algorithms=args.enable_deterministic_algorithms,
    )

    model_name = args.model.rstrip("/")

    if model_name.split("/")[-1].strip(".") == "" and "gguf" not in args.format:
        export_dir = os.path.join(args.output_dir, f"w{autoround.bits}g{autoround.group_size}")
    elif model_name.split("/")[-1].strip(".") == "" and "gguf" in args.format:
        export_dir = args.output_dir
    elif model_name.split("./")[-1].strip("./") != "" and "gguf" in args.format:
        export_dir = os.path.join(args.output_dir, model_name.split("/")[-1] + "-gguf")
    else:
        export_dir = os.path.join(
            args.output_dir, model_name.split("/")[-1] + f"-w{autoround.bits}g{autoround.group_size}"
        )

    model, folders = autoround.quantize_and_save(export_dir, format=args.format)

    if args.low_cpu_mem_mode == 1 or args.low_cpu_mem_mode == 2:
        import shutil

        shutil.rmtree(args.low_cpu_mem_tmp_dir, ignore_errors=True)

    model.eval()
    clear_memory()


def vlmeval(args):
    set_cuda_visible_devices(args.device_map)
    device_str, parallelism = get_device_and_parallelism(args.device_map)
    if parallelism:
        os.environ["AUTO_SPLIT"] = "1"
    if isinstance(args.tasks, str):
        args.tasks = args.tasks.replace(" ", "").split(",")
    from auto_round.mllm import mllm_eval

    mllm_eval(
        args.model,
        work_dir=args.output_dir,
        data_store_dir=args.eval_data_dir,
        dataset=args.tasks,
        pack=args.pack,
        fps=args.fps,
        nframe=args.nframe,
        rerun=args.rerun,
        judge=args.judge,
        verbose=args.verbose,
        mode=args.mode,
        ignore=args.ignore,
    )


def setup_lmms_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "--model_name", "--model_name_or_path", help="model name or path")
    parser.add_argument(
        "--tasks",
        default="pope,textvqa_val,scienceqa,mmbench_en",
        help="To get full list of tasks, use the command lmms-eval --tasks list",
    )
    parser.add_argument("--output_dir", default="./eval_result", type=str, help="the directory to save quantized model")
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "--bs",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total"
        " number of examples.",
    )
    args = parser.parse_args()
    return args


def lmms_eval(args):
    set_cuda_visible_devices(args.device_map)
    device_str, parallelism = get_device_and_parallelism(args.device_map)

    from auto_round.mllm import lmms_eval

    results = lmms_eval(
        model=args.model,
        tasks=args.tasks,
        output_dir=args.output_dir,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=device_str,
        use_cache=None,
        apply_chat_template=False,
    )
    return results
