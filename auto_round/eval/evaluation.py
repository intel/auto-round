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
from typing import Optional, Union

from auto_round.logger import logger
from auto_round.utils import dispatch_model_block_wise

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def simple_evaluate_user_model(
    user_model,
    tokenizer,
    batch_size: Optional[int] = 1,
    limit: Optional[Union[int, float]] = None,
    max_batch_size: Optional[int] = 64,
    eval_model_dtype="auto",
    add_bos_token: bool = False,
    mllm: bool = False,
    **kwargs,
):
    import lm_eval  # pylint: disable=E0401
    from lm_eval.models.huggingface import HFLM  # pylint: disable=E0401

    if mllm:
        from lm_eval.models.hf_vlms import HFMultimodalLM  # pylint: disable=E0401

        if batch_size is None or batch_size == "auto":
            logger.warning("hf-multimodal models does not support auto currently, reset eval_bs to 16")
            batch_size = 16
        hflm = HFMultimodalLM(
            pretrained=user_model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            dtype=eval_model_dtype,
            add_bos_token=add_bos_token,
        )
    else:
        hflm = HFLM(
            pretrained=user_model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            dtype=eval_model_dtype,
            add_bos_token=add_bos_token,
        )
    return lm_eval.simple_evaluate(
        model=hflm, model_args=None, batch_size=batch_size, max_batch_size=max_batch_size, limit=limit, **kwargs
    )


def simple_evaluate(
    model,
    model_args: Optional[Union[str, dict]] = None,
    batch_size: Optional[int] = None,
    limit: Optional[Union[int, float]] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    **kwargs,
):
    import lm_eval  # pylint: disable=E0401

    return lm_eval.simple_evaluate(
        model=model,
        model_args=model_args,
        batch_size=batch_size,
        limit=limit,
        max_batch_size=max_batch_size,
        device=device,
        **kwargs,
    )


def evaluate_diffusion_model(autoround, model, args):
    """
    Evaluate diffusion models.

    Args:
        autoround: AutoRound instance
        model: Quantized model
        args: Command line arguments
    """
    import torch

    from auto_round.utils import detect_device, get_model_dtype, logger

    # Prepare inference pipeline
    pipe = autoround.pipe
    pipe.to(model.dtype)
    pipe.transformer = model
    device_str = detect_device(args.device_map if hasattr(args, "device_map") else "0")
    pipe = pipe.to(device_str)

    # Set evaluation dtype
    eval_model_dtype = get_model_dtype(args.eval_model_dtype, "auto")
    if pipe.dtype != eval_model_dtype and eval_model_dtype != "auto":
        pipe.to(getattr(torch, eval_model_dtype))

    # Prepare generation kwargs
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

    # Create image save directory
    if not os.path.exists(args.image_save_dir):
        os.makedirs(args.image_save_dir)

    # Single prompt generation
    if args.prompt is not None:
        outputs = pipe(prompt=args.prompt, **gen_kwargs)
        save_path = os.path.join(args.image_save_dir, "img.png")
        outputs.images[0].save(save_path)
        logger.info(f"Image generated with prompt {args.prompt} is saved as {save_path}")

    # Batch prompt evaluation
    if args.prompt_file is not None:
        from auto_round.compressors.diffusion import diffusion_eval

        metrics = args.metrics.split(",")
        diffusion_eval(pipe, args.prompt_file, metrics, args.image_save_dir, 1, gen_kwargs)


def load_gguf_model_for_eval(eval_folder, formats, args):
    """
    Load GGUF model for evaluation.

    Args:
        eval_folder: Path to saved model
        formats: List of export formats
        args: Command line arguments

    Returns:
        model, tokenizer: Loaded model and tokenizer
    """
    import sys

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from auto_round.utils import get_model_dtype, logger

    # Find corresponding GGUF format
    gguf_format = None
    for format in formats:
        if format.startswith("gguf"):
            gguf_format = format.split(":")[-1].upper()
            break

    if gguf_format is None:
        logger.error("No valid gguf format found in formats. Please check the input.")
        sys.exit(-1)

    # Find matching GGUF file
    gguf_file = None
    for file in os.listdir(eval_folder):
        if gguf_format in file:
            gguf_file = file
            break

    if gguf_file is None:
        logger.error("Cannot find correct gguf file for evaluation, please check.")
        sys.exit(-1)

    # Load model and tokenizer
    logger.warning("evaluating gguf model is an experimental feature, the accuracy may be not correct.")
    eval_model_dtype = get_model_dtype(args.eval_model_dtype, "auto")

    if eval_model_dtype in ["float32", "auto"]:
        logger.warning(
            "set '--eval_model_dtype bf16' can significantly speed up evaluation for gguf model,"
            " but may affect accuracy."
        )

    model = AutoModelForCausalLM.from_pretrained(
        eval_folder, gguf_file=gguf_file, device_map="auto", torch_dtype=eval_model_dtype
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(eval_folder, gguf_file=gguf_file)

    return model, tokenizer


def prepare_model_for_eval(model, device_map, eval_model_dtype):
    """
    Prepare model for evaluation.

    Args:
        model: Quantized model
        device_map: Device string
        eval_model_dtype: Evaluation data type

    Returns:
        model: Prepared model
    """
    import torch

    from auto_round.utils import detect_device

    # Handle multi-device model
    if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
        from accelerate.big_modeling import dispatch_model

        dispatch_model(model, model.hf_device_map)
    else:
        dispatch_model_block_wise(model, device_map)

    # Convert dtype
    if model.dtype != eval_model_dtype and eval_model_dtype != "auto":
        model.to(getattr(torch, eval_model_dtype))

    return model


def evaluate_with_model_instance(model, tokenizer, device_str, args):
    """
    Evaluate with model instance.
    Applicable to fake quantization and GGUF models.

    Args:
        model: Model instance
        tokenizer: Tokenizer
        device_str: Device string
        args: Command line arguments
    """
    import time

    from lm_eval.utils import make_table  # pylint: disable=E0401

    from auto_round.eval.eval_cli import eval_task_by_task
    from auto_round.utils import get_model_dtype, logger

    tasks = args.tasks
    if isinstance(tasks, str):
        tasks = tasks.split(",")

    # Task-by-task evaluation
    if args.eval_task_by_task:
        eval_task_by_task(
            model,
            tokenizer=tokenizer,
            device=device_str,
            tasks=args.tasks,
            limit=args.limit,
            batch_size=args.eval_bs,
            eval_model_dtype=get_model_dtype(args.eval_model_dtype, "auto"),
            add_bos_token=args.add_bos_token,
        )
    else:
        # Batch evaluation
        eval_bs = args.eval_bs
        if eval_bs is None or eval_bs == "auto":
            logger.warning("This API does not support auto currently, reset eval_bs to 16")
            eval_bs = 16

        st = time.time()
        res = simple_evaluate_user_model(
            model,
            tokenizer,
            tasks=tasks,
            batch_size=eval_bs,
            limit=args.limit,
            device=device_str,
            eval_model_dtype=get_model_dtype(args.eval_model_dtype, "auto"),
            add_bos_token=args.add_bos_token,
        )
        print(make_table(res))
        print("evaluation running time=%ds" % (time.time() - st))


def evaluate_with_model_path(eval_folder, device_str, autoround, args):
    """
    Evaluate with model path.
    Applicable to other quantization formats.

    Args:
        eval_folder: Path to saved model
        device_str: Device string
        autoround: AutoRound instance
        args: Command line arguments
    """
    import time

    from lm_eval.utils import make_table  # pylint: disable=E0401

    from auto_round.eval.eval_cli import _eval_init, eval_task_by_task
    from auto_round.utils import get_model_dtype, logger

    tasks = args.tasks
    if isinstance(tasks, str):
        tasks = tasks.split(",")

    # Task-by-task evaluation
    if args.eval_task_by_task:
        eval_task_by_task(
            eval_folder,
            device=device_str,
            tasks=args.tasks,
            batch_size=args.eval_bs,
            limit=args.limit,
            eval_model_dtype=get_model_dtype(args.eval_model_dtype, "auto"),
            mllm=autoround.mllm,
            add_bos_token=args.add_bos_token,
        )
    else:
        # Batch evaluation
        tasks, model_args, device_str = _eval_init(
            args.tasks,
            eval_folder,
            args.device_map,
            args.disable_trust_remote_code,
            dtype=get_model_dtype(args.eval_model_dtype, "auto"),
        )

        st = time.time()
        model_args += f",add_bos_token={args.add_bos_token}"

        # Choose evaluation method based on model type
        if autoround.mllm:
            model_type = "hf-multimodal"
            eval_bs = args.eval_bs
            if eval_bs is None or eval_bs == "auto":
                logger.warning("hf-multimodal models does not support auto currently, reset eval_bs to 16")
                eval_bs = 16
        else:
            model_type = "hf"
            eval_bs = args.eval_bs

        res = simple_evaluate(
            model=model_type,
            model_args=model_args,
            tasks=tasks,
            device=device_str,
            batch_size=eval_bs,
            limit=args.limit,
        )
        print(make_table(res))
        print("evaluation running time=%ds" % (time.time() - st))


def run_model_evaluation(model, tokenizer, autoround, folders, formats, device_str, args):
    """
    Run model evaluation.
    Unified evaluation entry point that dispatches to different evaluation logic based on model type.

    Args:
        model: Quantized model
        tokenizer: Tokenizer
        autoround: AutoRound instance
        folders: List of export folders
        formats: List of export formats
        device_str: Device string
        args: Command line arguments
    """
    from auto_round.utils import get_library_version, get_model_dtype, logger

    # Handle diffusion models separately
    if getattr(autoround, "diffusion", False):
        evaluate_diffusion_model(autoround, model, args)
        return

    # Check if evaluation is needed for language models
    eval_folder = folders[-1] if folders else None
    if args.tasks is None or args.tasks == "" or eval_folder is None:
        return

    # Handle vllm backend evaluation
    if hasattr(args, "eval_backend") and args.eval_backend == "vllm":
        from auto_round.eval.eval_cli import eval_with_vllm

        # Create a minimal args object with essential parameters
        vllm_args = type("Args", (), {})()
        # Required parameters
        vllm_args.model = eval_folder
        vllm_args.tasks = args.tasks
        vllm_args.device_map = getattr(args, "device_map", device_str)
        # Optional common parameters
        vllm_args.eval_bs = getattr(args, "eval_bs", None)
        vllm_args.mllm = getattr(args, "mllm", None)
        vllm_args.limit = getattr(args, "limit", None)
        vllm_args.eval_model_dtype = getattr(args, "eval_model_dtype", None)
        vllm_args.disable_trust_remote_code = getattr(args, "disable_trust_remote_code", False)
        vllm_args.add_bos_token = getattr(args, "add_bos_token", False)
        vllm_args.seed = getattr(args, "seed", 42)
        # VLLM-specific parameters
        vllm_args.vllm_args = getattr(args, "vllm_args", None)
        eval_with_vllm(vllm_args)
        return

    lm_eval_version = get_library_version("lm-eval")
    logger.info(f"Using lm-eval version {lm_eval_version}")

    # Handle Llama model special case
    if "llama" in args.model.lower() and not args.add_bos_token:
        logger.warning("set add_bos_token=True for llama model.")
        args.add_bos_token = True

    # Check if GGUF model
    eval_gguf_model = any(file.endswith("gguf") for file in os.listdir(eval_folder))

    # Determine if model instance evaluation is needed
    need_model_instance = (autoround.act_bits <= 8 and formats[-1] == "fake") or eval_gguf_model

    if need_model_instance:
        # Load or prepare model instance
        if eval_gguf_model:
            model, tokenizer = load_gguf_model_for_eval(eval_folder, formats, args)
        else:
            eval_model_dtype = get_model_dtype(args.eval_model_dtype, "auto")
            model = prepare_model_for_eval(model, args.device_map, eval_model_dtype)

        # Evaluate with model instance
        evaluate_with_model_instance(model, tokenizer, device_str, args)
    else:
        # Evaluate with model path
        evaluate_with_model_path(eval_folder, device_str, autoround, args)
