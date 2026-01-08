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

from auto_round.auto_scheme import AutoScheme
from auto_round.compressors import BaseCompressor
from auto_round.eval.eval_cli import EvalArgumentParser, _eval_init, eval, eval_task_by_task
from auto_round.schemes import PRESET_SCHEMES
from auto_round.utils import (
    clear_memory,
    get_device_and_parallelism,
    get_model_dtype,
)

RECIPES = {
    "default": {"batch_size": 8, "iters": 200, "seqlen": 2048, "nsamples": 128, "lr": None},
    "best": {"batch_size": 8, "iters": 1000, "seqlen": 2048, "nsamples": 512, "lr": None},
    "light": {"batch_size": 8, "iters": 50, "seqlen": 2048, "nsamples": 128, "lr": 5e-3},
    "fast": {"batch_size": 4, "iters": 200, "seqlen": 512, "nsamples": 128, "lr": None},
}


class BasicArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "model",
            default=None,
            nargs="?",
            help="Path to the pre-trained model or model identifier from huggingface.co/models. "
            "Examples: 'facebook/opt-125m', 'bert-base-uncased', or local path like '/path/to/model'",
        )
        basic = self.add_argument_group("Basic Arguments")
        basic.add_argument(
            "--model_name",
            "--model",
            "--model_name_or_path",
            default="facebook/opt-125m",
            help="Path to the pre-trained model or model identifier from huggingface.co/models. "
            "Examples: 'facebook/opt-125m', 'bert-base-uncased', or local path like '/path/to/model'",
        )
        basic.add_argument("--model_dtype", default=None, help="model dtype used to load the pre-trained model")
        basic.add_argument(
            "--platform",
            default="hf",
            help="Platform to load the pre-trained model. Options: [hf, model_scope]."
            " hf stands for huggingface and model_scope stands for model scope.",
        )
        basic.add_argument(
            "--scheme",
            default="W4A16",
            type=str,
            # choices=["W4A16", "W2A16", "W3A16", "W8A16", "MXFP4", "MXFP8", "NVFP4", "FPW8A16", "FP8_STATIC"],
            help="Quantization scheme to use. "
            "W4A16: 4-bit weights with 16-bit activations (default). "
            "Other options include W2A16, W3A16, W8A16 for different bit widths, "
            "and MXFP4/MXFP8/NVFP4 for different data type.",
        )
        basic.add_argument(
            "--batch_size",
            "--train_bs",
            "--bs",
            default=None,
            type=int,
            help="The batch size for tuning/calibration."
            "Larger batch sizes may improve stability but require more memory.",
        )
        basic.add_argument(
            "--avg_bits", "--target_bits", default=None, type=float, help="for auto scheme, number of avg weight bits"
        )
        basic.add_argument(
            "--options", default=None, type=str, help="for auto scheme, options for auto scheme, e.g. 'W4A16,W8A16'"
        )

        basic.add_argument(
            "--iters",
            "--iter",
            default=None,
            type=int,
            help="Number of iterations to tune each block. "
            "More iterations may lead to better quantization quality but take longer.",
        )
        basic.add_argument(
            "--seqlen",
            "--seq_len",
            default=None,
            type=int,
            help="Sequence length of the calibration samples"
            "Longer sequences capture more context but use more memory.",
        )
        basic.add_argument(
            "--nsamples",
            "--nsample",
            default=None,
            type=int,
            help="Number of calibration samples to use for quantization.",
        )
        basic.add_argument(
            "--device_map",
            "--device",
            "--devices",
            default="0",
            type=str,
            help="The device to be used for tuning. "
            "Currently, device settings support CPU, GPU, and HPU."
            "The default is set to cuda:0,"
            "allowing for automatic detection and switch to HPU or CPU."
            "set --device 0,1,2 to use multiple cards.",
        )
        basic.add_argument(
            "--dataset",
            default="NeelNanda/pile-10k",
            type=str,
            help="Calibration dataset for quantization. "
            "Should be a dataset from huggingface datasets or local path. ",
        )
        basic.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
        basic.add_argument("--adam", action="store_true", help="Use Adam optimizer instead of SignSGD.")
        basic.add_argument(
            "--low_gpu_mem_usage",
            action="store_true",
            help="Enable memory-efficient mode by offloading intermediate features to CPU. "
            "Useful when working with large models that don't fit in GPU memory.",
        )
        basic.add_argument("--low_cpu_mem_usage", action="store_true", help="Lower CPU memory mode. Defaults to False.")
        basic.add_argument(
            "--format",
            "--formats",
            default="auto_round",
            type=str,
            help="Output format for the quantized model."
            "'auto_round' is the recommended format"
            "use command `auto_round list format` to show all supported formats with support scheme.",
        )
        basic.add_argument(
            "--output_dir",
            default="./tmp_autoround",
            type=str,
            help="Directory to save the quantized model and related files",
        )
        basic.add_argument(
            "--not_use_best_mse",
            action="store_true",
            help="Disable using the iteration with best MSE loss during tuning.",
        )
        basic.add_argument(
            "--enable_torch_compile", action="store_true", help="Enable PyTorch compilation for faster execution. "
        )

        tuning = self.add_argument_group("Tuning Arguments")
        tuning.add_argument(
            "--ignore_scale_zp_bits",
            action="store_true",
            help="for auto scheme whether ignore scale zp bits calculation ",
        )
        tuning.add_argument(
            "--lr",
            default=None,
            type=float,
            help="Learning rate for tuning. " "If None, automatically sets to 1.0/iters. ",
        )
        tuning.add_argument(
            "--minmax_lr",
            default=None,
            type=float,
            help="Learning rate specifically for min-max tuning. " "If None, uses the same value as --lr. ",
        )
        tuning.add_argument(
            "--momentum",
            default=0,
            type=float,
            help="Momentum factor for the optimizer. Default is 0 (no momentum).",
        )
        tuning.add_argument(
            "--gradient_accumulate_steps",
            default=1,
            type=int,
            help="Number of steps to accumulate gradients before updating weights. "
            "Effectively increases batch size without requiring more GPU memory. "
            "Useful for large models with limited memory.",
        )
        tuning.add_argument(
            "--nblocks",
            default=1,
            type=int,
            help="Number of blocks to tune simultaneously. "
            "Higher values may speed up tuning but require more memory. "
            "Recommended to keep at 1 for stability with large models.",
        )
        tuning.add_argument(
            "--scale_dtype",
            default=None,
            choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"],
            help="Data type for quantization scales. "
            "fp16/bf16: lower memory, fp32: higher precision. "
            "Choose based on your hardware support and accuracy requirements.",
        )
        tuning.add_argument(
            "--disable_amp",
            action="store_true",
            help="Disable Automatic Mixed Precision (AMP). "
            "AMP speeds up training but may affect numerical stability in some cases.",
        )
        tuning.add_argument(
            "--disable_minmax_tuning",
            action="store_true",
            help="Disable weight min-max range tuning. "
            "Not recommended as it may significantly reduce quantization accuracy.",
        )
        tuning.add_argument(
            "--enable_norm_bias_tuning", action="store_true", help="Enable normalization layer bias tuning. "
        )
        tuning.add_argument(
            "--disable_quanted_input",
            action="store_true",
            help="Use original (non-quantized) inputs for each block instead of"
            " quantized outputs from previous blocks. ",
        )
        tuning.add_argument(
            "--to_quant_block_names",
            default=None,
            type=str,
            help="Specific blocks to quantize, separated by commas. "
            "Example: 'block1,block2,block3'. "
            "If None, all blocks will be quantized.",
        )
        tuning.add_argument(
            "--enable_alg_ext",
            action="store_true",
            help="Enable experimental algorithms that may provide better quantization results. "
            "These are newer methods that might improve accuracy but are less tested.",
        )
        tuning.add_argument(
            "--disable_deterministic_algorithms",
            action="store_true",
            help="deprecated, disable torch deterministic algorithms.",
        )
        tuning.add_argument(
            "--enable_deterministic_algorithms",
            action="store_true",
            help="Enable PyTorch deterministic algorithms for reproducible results. ",
        )
        tuning.add_argument(
            "--disable_opt_rtn",
            "--disable-opt-rtn",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Disable optimization for RTN (Round-To-Nearest) mode when iters=0. "
            "RTN is fast but less accurate; keeping optimization enabled is recommended.",
        )

        scheme = self.add_argument_group("Scheme Arguments")
        scheme.add_argument("--bits", default=None, type=int, help="Number of bits for weight quantization. ")
        scheme.add_argument("--group_size", default=None, type=int, help="Group size for weight quantization.")
        scheme.add_argument("--asym", action="store_true", help="Use asymmetric quantization instead of symmetric.")
        scheme.add_argument(
            "--data_type",
            "--dtype",
            default=None,
            help="Data type for quantization. Options: 'int' for integer, 'mx_fp' for mixed floating-point, etc.",
        )
        scheme.add_argument(
            "--act_bits",
            default=None,
            type=int,
            help="Number of bits for activation quantization. "
            "Activation quantization significantly impacts performance and accuracy.",
        )
        scheme.add_argument(
            "--act_group_size",
            default=None,
            type=int,
            help="Group size for activation quantization. " "Similar to weight group size but for activations.",
        )
        scheme.add_argument(
            "--act_data_type", "--act_dtype", default=None, type=str, help="Data type for activation quantization. "
        )
        scheme.add_argument(
            "--disable_act_dynamic", action="store_true", help="Use static instead of dynamic activation quantization. "
        )
        scheme.add_argument(
            "--shared_layers",
            type=str,
            nargs="+",
            action="append",
            default=None,
            help="[mix-precision] ensure that listed layers are using same data type for quantization",
        )
        scheme.add_argument(
            "--quant_lm_head",
            action="store_true",
            help="Quantize the lm_head. " "Usually kept in higher precision for better output quality.",
        )
        scheme.add_argument(
            "--ignore_layers",
            "--fp_layers",
            default="",
            type=str,
            help="List of layer names to keep in original precision (not quantized). "
            "Useful for preserving critical layers. Separate multiple names with commas.",
        )
        scheme.add_argument(
            "--static_kv_dtype",
            default=None,
            type=str,
            choices=["fp8", "float8_e4m3fn"],
            help="Data type for static quantize key and value. ",
        )

        scheme.add_argument(
            "--static_attention_dtype",
            default=None,
            type=str,
            choices=["fp8", "float8_e4m3fn"],
            help="Data type for static quantize attention. ",
        )
        gguf = self.add_argument_group("Double Quant Arguments")
        gguf.add_argument(
            "--super_group_size", default=None, type=int, help="Super group size for double quantization."
        )
        gguf.add_argument(
            "--super_bits",
            default=None,
            type=int,
            help="Number of bits for scale and zero-point quantization in double quantization. ",
        )

        ## ======================= eval =======================
        eval_args = self.add_argument_group("eval arguments")
        eval_args.add_argument(
            "--disable_trust_remote_code",
            action="store_true",
            help="Disable trusting remote code when loading models. "
            "Use for security if you don't trust the model source.",
        )
        eval_args.add_argument(
            "--tasks",
            "--task",
            nargs="?",
            const="lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1,"
            "openbookqa,boolq,arc_easy,arc_challenge",
            default=None,
            help="LM-Evaluation-Harness tasks to run. "
            "Specify specific tasks like 'mmlu,wikitext' for custom evaluation.",
        )
        eval_args.add_argument("--eval_bs", default=None, type=int, help="Batch size for evaluation.")
        eval_args.add_argument(
            "--limit",
            type=float,
            default=None,
            metavar="N|0<N<1",
            help="Limit the number of examples per task. "
            "Integer: exact number of examples (e.g., 1000). "
            "Float between 0-1: fraction of total examples.",
        )
        eval_args.add_argument(
            "--eval_task_by_task", action="store_true", help="Evaluate tasks sequentially instead of batching. "
        )
        eval_args.add_argument(
            "--eval_model_dtype",
            default=None,
            type=str,
            help="Torch data type for model loading during evaluation. "
            "Options: 'float16', 'bfloat16', 'float32'. "
            "Should match your hardware capabilities for best performance.",
        )
        eval_args.add_argument("--add_bos_token", action="store_true", help="add BOS token")

        ## ======================= MLLM =======================
        mllm_args = self.add_argument_group("Multimodal Large Language Model(MLLM) arguments")
        mllm_args.add_argument(
            "--mllm",
            action="store_true",
            help="[Deprecated] AutoRound now automatically detects and uses MLLM mode when needed.",
        )
        mllm_args.add_argument(
            "--quant_nontext_module",
            action="store_true",
            help="Quantize non-text modules (vision/audio/video components). "
            "Enables full multimodal model quantization but may affect visual quality.",
        )
        mllm_args.add_argument(
            "--extra_data_dir",
            default=None,
            type=str,
            help="Directory containing multimodal data (images/audio/videos). "
            "Can be a single directory or specify types: "
            "'image=/path/to/images,video=/path/to/videos,audio=/path/to/audio'. "
            "If not found locally, will attempt to download standard datasets.",
        )
        mllm_args.add_argument(
            "--template",
            default=None,
            type=str,
            help="Custom template for building training datasets. "
            "Useful for specialized multimodal tasks or custom data formats.",
        )

        ## ======================= diffusion model eval =======================
        diffusion_args = self.add_argument_group("diffusion model arguments")
        diffusion_args.add_argument(
            "--prompt_file",
            default=None,
            type=str,
            help="File containing prompts for evaluation, one per line. "
            "Use this for batch evaluation with multiple prompts.",
        )
        diffusion_args.add_argument(
            "--prompt",
            default=None,
            type=str,
            help="Single prompt for quick testing. " "Overrides prompt_file if both are specified.",
        )
        diffusion_args.add_argument(
            "--metrics",
            "--metric",
            default="clip",
            help="Evaluation metrics for generated images. "
            "'clip': CLIP score measuring text-image alignment. "
            "'clip-iqa': CLIP-based image quality assessment. "
            "'imagereward': Learned metric for image quality.",
        )
        diffusion_args.add_argument(
            "--image_save_dir",
            default="./tmp_image_save",
            type=str,
            help="Directory to save generated images during evaluation. " "Useful for visual inspection of results.",
        )
        diffusion_args.add_argument(
            "--guidance_scale",
            default=7.5,
            type=float,
            help="Classifier-free guidance scale for diffusion models. "
            "Higher values (7-20) make the model follow the prompt more closely. "
            "Lower values give more creative/random results.",
        )

        diffusion_args.add_argument(
            "--num_inference_steps",
            default=50,
            type=int,
            help="Number of denoising steps in the diffusion process. "
            "More steps (50-100) usually give better quality but take longer. "
            "Fewer steps (10-30) are faster but lower quality.",
        )

        diffusion_args.add_argument(
            "--generator_seed",
            default=None,
            type=int,
            help="Random seed for image generation reproducibility. "
            "Using the same seed produces identical results across runs.",
        )


def list_item():
    args = argparse.ArgumentParser()
    args.add_argument("item", type=str, help="item to list, e.g., format")
    args = args.parse_args()
    if args.item == "format" or args.item == "formats":
        from auto_round.formats import OutputFormat

        print("AutoRound supported output formats and quantization scheme:")
        print(OutputFormat.get_support_matrix())


def start(recipe="default"):
    recipe = RECIPES[recipe]
    parser = BasicArgumentParser()
    args = parser.parse_args()
    for k, v in recipe.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    tune(args)


def tune(args):
    assert args.model or args.model_name, "[model] or --model MODEL_NAME should be set."
    if args.model is None:
        args.model = args.model_name
    if args.eval_bs is None:
        args.eval_bs = "auto"
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.versions import require_version

    if args.tasks is not None:
        require_version(
            "lm_eval>=0.4.2",
            "lm-eval is required for evaluation, please install it with `pip install 'lm-eval>=0.4.2'`",
        )

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
        DiffusionExtraConfig,
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
        ignore_layers=args.ignore_layers,
        static_kv_dtype=args.static_kv_dtype,
        static_attention_dtype=args.static_attention_dtype,
    )
    mllm_config = MLLMExtraConfig(
        quant_nontext_module=args.quant_nontext_module, extra_data_dir=args.extra_data_dir, template=args.template
    )
    diffusion_config = DiffusionExtraConfig(
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator_seed=args.generator_seed,
    )
    extra_config.tuning_config = tuning_config
    extra_config.scheme_config = scheme_config
    extra_config.mllm_config = mllm_config
    extra_config.diffusion_config = diffusion_config

    layer_config = {}

    if args.avg_bits is not None:
        if args.options is None:
            raise ValueError("please set --options for auto scheme")
        scheme = AutoScheme(
            options=args.options,
            avg_bits=args.avg_bits,
            shared_layers=args.shared_layers,
            ignore_scale_zp_bits=args.ignore_scale_zp_bits,
        )

    autoround: BaseCompressor = AutoRound(
        model=model_name,
        platform=args.platform,
        scheme=scheme,
        dataset=args.dataset,
        iters=args.iters,
        seqlen=args.seqlen,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        device_map=args.device_map,
        enable_torch_compile=enable_torch_compile,
        seed=args.seed,
        not_use_best_mse=args.not_use_best_mse,
        enable_adam=args.adam,
        extra_config=extra_config,
        layer_config=layer_config,
        model_dtype=args.model_dtype,
        momentum=args.momentum,
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

    model.eval()
    clear_memory()

    eval_model_dtype = get_model_dtype(args.eval_model_dtype, "auto")

    # diffusion model has different evaluation path
    if getattr(autoround, "diffusion", False):
        pipe = autoround.pipe
        pipe.to(model.dtype)
        pipe.transformer = model
        device_str = detect_device(device_str)
        pipe = pipe.to(device_str)
        if pipe.dtype != eval_model_dtype and eval_model_dtype != "auto":
            pipe.to(getattr(torch, eval_model_dtype))

        gen_kwargs = {
            "guidance_scale": args.guidance_scale,
            "output_type": "pil",
            "num_inference_steps": args.num_inference_steps,
            "generator": (
                None
                if args.generator_seed is None
                else torch.Generator(device=pipe.device).manual_seed(args.generator_seed)
            ),
        }
        if not os.path.exists(args.image_save_dir):
            os.makedirs(args.image_save_dir)

        if args.prompt is not None:
            outputs = pipe(prompt=args.prompt, **gen_kwargs)
            outputs.images[0].save(os.path.join(args.image_save_dir, "img.png"))
            logger.info(
                f"Image generated with prompt {args.prompt} is saved as {os.path.join(args.image_save_dir, 'img.png')}"
            )

        if args.prompt_file is not None:
            from auto_round.compressors.diffusion import diffusion_eval

            metrics = args.metrics.split(",")
            diffusion_eval(pipe, args.prompt_file, metrics, args.image_save_dir, 1, gen_kwargs)
        return

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

    if "llama" in args.model.lower() and not args.add_bos_token:
        logger.warning("set add_bos_token=True for llama model.")
        args.add_bos_token = True
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
                add_bos_token=args.add_bos_token,
            )
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
                limit=args.limit,
                device=device_str,
                eval_model_dtype=eval_model_dtype,
                add_bos_token=args.add_bos_token,
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
                mllm=autoround.mllm,  # pylint: disable=E1101
                add_bos_token=args.add_bos_token,
            )
        else:
            from auto_round.eval.evaluation import simple_evaluate

            tasks, model_args, device_str = _eval_init(
                args.tasks, eval_folder, args.device_map, args.disable_trust_remote_code, dtype=eval_model_dtype
            )
            st = time.time()
            model_args += f",add_bos_token={args.add_bos_token}"
            if autoround.mllm:  # pylint: disable=E1101
                model_type = "hf-multimodal"
                if args.eval_bs is None or args.eval_bs == "auto":
                    logger.warning("hf-multimodal models does not support auto currently, reset eval_bs to 16")
                    args.eval_bs = 16
            else:
                model_type = "hf"
            res = simple_evaluate(
                model=model_type,
                model_args=model_args,
                tasks=tasks,
                device=device_str,
                batch_size=args.eval_bs,
                limit=args.limit,
            )
            print(make_table(res))
            print("evaluation running time=%ds" % (time.time() - st))


def setup_eval_parser():
    parser = EvalArgumentParser()
    args = parser.parse_args()
    return args


def run_eval():
    from auto_round.logger import logger
    from auto_round.utils import is_mllm_model

    args = setup_eval_parser()
    assert args.model or args.model_name, "[model] or --model MODEL_NAME should be set."

    if args.model is None:
        args.model = args.model_name
    if "llama" in args.model.lower() and not args.add_bos_token:
        logger.warning("set add_bos_token=True for llama model.")
        args.add_bos_token = True
    if is_mllm_model(args.model):
        args.mllm = True

    if args.eval_task_by_task:
        eval_task_by_task(
            model=args.model,
            device=args.device_map,
            tasks=args.tasks,
            batch_size=args.eval_bs,
            trust_remote_code=not args.disable_trust_remote_code,
            eval_model_dtype=args.eval_model_dtype,
            add_bos_token=args.add_bos_token,
        )
    else:
        eval(args)


def run():
    if "list" in sys.argv or "--list" in sys.argv:
        if "list" in sys.argv:
            sys.argv.remove("list")
        if "--list" in sys.argv:
            sys.argv.remove("--list")
        list_item()
        exit()
    if "--eval" in sys.argv or "eval" in sys.argv:
        if "--eval" in sys.argv:
            sys.argv.remove("--eval")
        if "eval" in sys.argv:
            sys.argv.remove("eval")
        run_eval()
    else:
        start()


def run_best():
    start("best")


def run_light():
    start("light")


def run_fast():
    start("fast")


if __name__ == "__main__":
    run()
