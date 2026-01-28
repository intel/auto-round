# Copyright (c) 2025 Intel Corporation
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
import importlib.util
import os
import time

from transformers.utils.versions import require_version

from auto_round.utils import (
    get_device_and_parallelism,
    get_device_str,
    get_model_dtype,
    set_cuda_visible_devices,
)


class EvalArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "model",
            default=None,
            nargs="?",
            help="Path to the pre-trained model or model identifier from huggingface.co/models. "
            "Examples: 'facebook/opt-125m', 'bert-base-uncased', or local path like '/path/to/model'",
        )
        self.add_argument(
            "--model_name",
            "--model",
            "--model_name_or_path",
            default="facebook/opt-125m",
            help="Path to the pre-trained model or model identifier from huggingface.co/models. "
            "Examples: 'facebook/opt-125m', 'bert-base-uncased', or local path like '/path/to/model'",
        )
        self.add_argument("--mllm", action="store_true", help="whether to eval multi-modal model.")
        self.add_argument(
            "--device_map",
            "--device",
            "--devices",
            default="0",
            type=str,
            help="the device to be used for evaluation. "
            "The default is set to 0,"
            "allowing for automatic detection and switch to any devices."
            "set --device 0,1,2 to use multiple cards.",
        )

        self.add_argument(
            "--tasks",
            "--task",
            default="lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1,"
            "truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge",
            help="LM-Evaluation-Harness tasks to run. "
            "Specify specific tasks like 'mmlu,wikitext' for custom evaluation.",
        )
        self.add_argument(
            "--disable_trust_remote_code",
            action="store_true",
            help="Disable trusting remote code when loading models. "
            "Use for security if you don't trust the model source.",
        )
        self.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
        self.add_argument(
            "--eval_bs", "--bs", "--batch_size", default=None, type=int, help="The batch size for evaluation"
        )
        self.add_argument(
            "--eval_task_by_task", action="store_true", help="Evaluate tasks sequentially instead of batching. "
        )
        self.add_argument(
            "--eval_model_dtype",
            default=None,
            type=str,
            help="Torch data type for model loading during evaluation. "
            "Options: 'float16', 'bfloat16', 'float32'. "
            "Should match your hardware capabilities for best performance.",
        )
        self.add_argument(
            "--limit",
            type=float,
            default=None,
            metavar="N|0<N<1",
            help="Limit the number of examples per task. "
            "Integer: exact number of examples (e.g., 1000). "
            "Float between 0-1: fraction of total examples.",
        )
        self.add_argument(
            "--eval_backend",
            default="hf",
            type=str,
            choices=["hf", "vllm"],
            help="Backend to use for model evaluation. Use hf backend for evaluation by default.",
        )
        self.add_argument("--add_bos_token", action="store_true", help="add BOS token")
        self.add_argument(
            "--vllm_args",
            default=None,
            type=str,
            help="(for vllm) Custom vllm arguments in format: 'arg1=value1,arg2=value2'. "
            "Example: 'tensor_parallel_size=2,gpu_memory_utilization=0.9'",
        )


def _eval_init(tasks, model_path, device, disable_trust_remote_code=False, dtype="auto"):
    set_cuda_visible_devices(device)
    device_str, parallelism = get_device_and_parallelism(device)
    if dtype != "auto":
        dtype = get_model_dtype(model_dtype=dtype)
    # ,add_bos_token={True}
    model_args = f"pretrained={model_path},trust_remote_code={not disable_trust_remote_code},dtype={dtype}"
    if parallelism:
        model_args += ",parallelize=True"
    if isinstance(tasks, str):
        tasks = tasks.split(",")
    return tasks, model_args, device_str


def eval(args):
    require_version(
        "lm_eval>=0.4.2", "lm-eval is required for evaluation, please install it with `pip install 'lm-eval>=0.4.2'`"
    )

    if args.eval_backend == "vllm":
        assert isinstance(args.model, str), "vllm evaluation only supports model name or path."
        eval_with_vllm(args)
        return
    tasks, model_args, device_str = _eval_init(
        args.tasks, args.model, args.device_map, args.disable_trust_remote_code, args.eval_model_dtype
    )

    # load after _eval_int in order to make sure import torch after set CUDA_VISIBLE_DEVICES
    from auto_round.eval.evaluation import simple_evaluate, simple_evaluate_user_model
    from auto_round.utils import logger

    if (batch_size := args.eval_bs) is None:
        batch_size = "auto:8"

    model, tokenizer, is_gguf_file, gguf_file = _load_gguf_model_if_needed(args.model, args.eval_model_dtype)

    if is_gguf_file:
        from lm_eval.utils import make_table  # pylint: disable=E0401

        st = time.time()
        res = simple_evaluate_user_model(
            model,
            tokenizer,
            tasks=tasks,
            batch_size=batch_size,
            device=device_str,
            limit=args.limit,
            add_bos_token=args.add_bos_token,
        )
        print(make_table(res))
        print("evaluation running time=%ds" % (time.time() - st))
    else:
        st = time.time()
        if "auto" in str(batch_size) and args.mllm:
            logger.warning("Batch size 'auto' is not yet supported for hf-multimodal models, reset to 16")
            batch_size = 16
        model_args += f",add_bos_token={args.add_bos_token}"
        res = simple_evaluate(
            model="hf" if not args.mllm else "hf-multimodal",
            model_args=model_args,
            tasks=tasks,
            device=device_str,
            batch_size=batch_size,
            limit=args.limit,
        )
        from lm_eval.utils import make_table  # pylint: disable=E0401

        print(make_table(res))
        print("evaluation running time=%ds" % (time.time() - st))


def eval_with_vllm(args):
    import time

    from lm_eval import evaluator  # pylint: disable=E0401
    from lm_eval.models.vllm_causallms import VLLM  # pylint: disable=E0401
    from lm_eval.models.vllm_vlms import VLLM_VLM  # pylint: disable=E0401
    from lm_eval.utils import make_table  # pylint: disable=E0401

    st = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device_str, _ = get_device_and_parallelism(args.device_map)
    eval_model_dtype = get_model_dtype(args.eval_model_dtype, "auto")
    if (batch_size := args.eval_bs) is None:
        batch_size = "auto:8"
    if isinstance(args.tasks, str):
        tasks = args.tasks.split(",")

    # Parse custom vllm_args if provided
    custom_vllm_kwargs = parse_vllm_args(getattr(args, "vllm_args", None))

    # Build vllm kwargs with base parameters
    vllm_kwargs = {
        "pretrained": args.model,
        "dtype": eval_model_dtype,
        "trust_remote_code": not args.disable_trust_remote_code,
        "add_bos_token": args.add_bos_token,
        "device": device_str,
        "batch_size": batch_size,
        "allow_deprecated_quantization": True,  # for vLLM==0.14.0
    }

    # Override with custom vllm_args if provided
    if custom_vllm_kwargs:
        from auto_round.logger import logger

        logger.info(f"Overriding VLLM parameters with custom args: {custom_vllm_kwargs}")
        vllm_kwargs.update(custom_vllm_kwargs)

    device = get_device_str()
    environ_mapping = {
        "cuda": "CUDA_VISIBLE_DEVICES",
        "xpu": "ZE_AFFINITY_MASK",
        "hpu": "HABANA_VISIBLE_MODULES",
    }
    if "tensor_parallel_size" not in vllm_kwargs:
        # Parse device_map to determine tensor_parallel_size and set CUDA_VISIBLE_DEVICES
        # Only accept formats like "0" or "0,1,2"
        assert device in environ_mapping, f"Device {device} not supported for vllm tensor parallelism."
        environ_name = environ_mapping[device]
        device_map = args.device_map
        device_ids = [d.strip() for d in str(device_map).split(",") if d.strip().isdigit()]
        if device_ids:
            device_id_str = ",".join(device_ids)
            os.environ[environ_name] = device_id_str
            tensor_parallel_size = len(device_ids)
            vllm_kwargs["tensor_parallel_size"] = tensor_parallel_size
            from auto_round.logger import logger

            logger.info(
                f"Set {environ_name}={os.environ[environ_name]}, " f"tensor_parallel_size={tensor_parallel_size}"
            )

    vllm_lm = VLLM_VLM(**vllm_kwargs) if args.mllm else VLLM(**vllm_kwargs)
    res = evaluator.simple_evaluate(
        model=vllm_lm,
        tasks=tasks,
        limit=args.limit,
    )

    print(make_table(res))
    print("evaluation running time=%ds" % (time.time() - st))


def eval_task_by_task(
    model,
    device=None,
    tasks=None,
    tokenizer=None,
    batch_size=None,
    limit=None,
    max_batch_size=64,
    trust_remote_code=True,
    eval_model_dtype=None,
    retry_times=3,
    mllm=False,
    add_bos_token=False,
):
    require_version(
        "lm_eval>=0.4.2", "lm-eval is required for evaluation, please install it with `pip install 'lm-eval>=0.4.2'`"
    )

    set_cuda_visible_devices(device)
    device_str, parallelism = get_device_and_parallelism(device)

    # load after _eval_int in order to make sure import torch after set CUDA_VISIBLE_DEVICES
    from lm_eval.models.huggingface import HFLM  # pylint: disable=E0401

    from auto_round.utils import logger

    if batch_size is None:
        batch_size = "auto:8"

    if not isinstance(model, str):
        parallelism = False
        is_gguf_file = False
        gguf_file = None
    else:
        model, tokenizer, is_gguf_file, gguf_file = _load_gguf_model_if_needed(model, eval_model_dtype)
        if is_gguf_file:
            parallelism = False

    eval_model_dtype = get_model_dtype(eval_model_dtype)
    if mllm:
        if batch_size is None or batch_size == "auto":
            logger.warning("hf-multimodal models does not support auto currently, reset eval_bs to 16")
            batch_size = 16
        from lm_eval.models.hf_vlms import HFMultimodalLM  # pylint: disable=E0401

        hflm = HFMultimodalLM(
            pretrained=model,
            tokenizer=tokenizer,
            device=device_str,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            parallelize=parallelism,
            trust_remote_code=trust_remote_code,
            dtype=eval_model_dtype,
            add_bos_token=add_bos_token,
        )
    else:
        hflm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            device=device_str,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            parallelize=parallelism,
            trust_remote_code=trust_remote_code,
            dtype=eval_model_dtype,
            add_bos_token=add_bos_token,
        )

    _evaluate_tasks_with_retry(tasks, hflm, device_str, batch_size, limit, retry_times)


def _load_gguf_model_if_needed(model_path, eval_model_dtype=None):
    """Detect and load GGUF model if the path points to a GGUF file.

    Args:
        model_path: Path to model or GGUF file
        eval_model_dtype: Data type for model evaluation

    Returns:
        Tuple of (model, tokenizer, is_gguf_file, gguf_file_name)
        If not a GGUF file, returns (model_path, None, False, None)
    """
    from auto_round.utils import logger

    is_gguf_file = False
    gguf_file = None
    tokenizer = None
    model = model_path

    # Check if model_path is a string before processing
    if isinstance(model_path, str):
        if os.path.isfile(model_path) and model_path.endswith(".gguf"):
            is_gguf_file = True
            gguf_file = os.path.basename(model_path)
            model = os.path.dirname(model_path)
        elif os.path.exists(model_path):
            for file in os.listdir(model_path):
                if file.endswith(".gguf"):
                    is_gguf_file = True
                    gguf_file = file
                    break

    if is_gguf_file:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        eval_model_dtype = get_model_dtype(eval_model_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model, gguf_file=gguf_file)

        logger.warning("evaluating gguf model is an experimental feature, the accuracy may be not correct.")
        if eval_model_dtype == "float32" or eval_model_dtype == "auto":
            logger.warning(
                "set '--eval_model_dtype bf16' can significantly speed up evaluation for gguf model,"
                " but may affect accuracy."
            )

        model = AutoModelForCausalLM.from_pretrained(
            model, gguf_file=gguf_file, device_map="auto", torch_dtype=eval_model_dtype
        )
        model.eval()

    return model, tokenizer, is_gguf_file, gguf_file


def _evaluate_tasks_with_retry(tasks, hflm, device_str, batch_size, limit, retry_times):
    """Evaluate tasks with automatic retry on OOM errors.

    Args:
        tasks: List of task names to evaluate
        hflm: HuggingFace LM model instance
        device_str: Device string for evaluation
        batch_size: Batch size for evaluation
        limit: Limit number of examples per task
        retry_times: Number of retry attempts on failure

    Returns:
        Aggregated results dictionary containing results, versions, n-shot, and higher_is_better
    """
    import time
    import traceback

    import lm_eval  # pylint: disable=E0401
    from lm_eval.utils import make_table  # pylint: disable=E0401

    from auto_round.utils import logger

    if isinstance(tasks, str):
        tasks = tasks.replace(" ", "").split(",")

    res_all = {}
    res_keys = ["results", "versions", "n-shot", "higher_is_better"]
    st = time.time()

    for task in tasks:
        current_retry_times = retry_times
        while current_retry_times:
            try:
                res = lm_eval.simple_evaluate(
                    model=hflm, model_args=None, device=device_str, tasks=task, batch_size=batch_size, limit=limit
                )
                break
            except Exception as e:
                cuda_error_msg = traceback.format_exc()
                try:
                    ori_batch_sizes = hflm.batch_sizes if hflm.batch_sizes else {"0": 64}
                    if not hflm.batch_sizes:
                        hflm.batch_sizes = ori_batch_sizes.copy()
                    try:
                        for k, v in hflm.batch_sizes.items():
                            hflm.batch_sizes[k] = max(v // 2, 1)
                        logger.warning(f"Out of memory, reset batch_size to {hflm.batch_sizes} and re-try.")
                        res = lm_eval.simple_evaluate(
                            model=hflm, model_args=None, device=device_str, tasks=task, batch_size=1, limit=limit
                        )
                        hflm.batch_sizes = ori_batch_sizes
                    except Exception as e:
                        traceback.print_exc()
                        pass
                except Exception as e:
                    logger.error(cuda_error_msg)
                    traceback.print_exc()
                    break
            current_retry_times -= 1

        if not res_all:
            res_all = res
        else:
            for key in res_keys:
                if key not in res_all:
                    continue
                else:
                    res_all[key].update(res[key])
        print(make_table(res_all))

    print("total eval time:", time.time() - st)


def parse_vllm_args(vllm_args_str):
    """Parse custom vllm arguments from string format.

    Args:
        vllm_args_str: String containing vllm arguments in format:
                      "--arg1=value1,--arg2=value2" or "arg1=value1,arg2=value2"

    Returns:
        Dictionary of parsed arguments with appropriate types (int, float, bool, or string)

    Example:
        >>> parse_vllm_args("--tensor_parallel_size=2,--gpu_memory_utilization=0.9")
        {'tensor_parallel_size': 2, 'gpu_memory_utilization': 0.9}
    """
    from auto_round.logger import logger

    custom_vllm_kwargs = {}

    if not vllm_args_str:
        return custom_vllm_kwargs

    logger.info(f"Parsing custom vllm arguments: {vllm_args_str}")

    for arg_pair in vllm_args_str.split(","):
        arg_pair = arg_pair.strip()
        # Normalize: replace space separator with '=' (e.g., "--arg value" -> "--arg=value")
        if "=" not in arg_pair and " " in arg_pair:
            parts = arg_pair.split(None, 1)  # Split on whitespace, max 2 parts
            if len(parts) == 2:
                arg_pair = f"{parts[0]}={parts[1]}"
        if "=" in arg_pair:
            # Remove leading '--' if present
            arg_pair = arg_pair.removeprefix("--")
            key, value = arg_pair.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Try to convert value to appropriate type
            try:
                # Try int first
                if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                    custom_vllm_kwargs[key] = int(value)
                # Try float
                elif "." in value:
                    custom_vllm_kwargs[key] = float(value)
                # Try boolean
                elif value.lower() in ("true", "false"):
                    custom_vllm_kwargs[key] = value.lower() == "true"
                # Keep as string
                else:
                    custom_vllm_kwargs[key] = value
                logger.info(
                    f"  Parsed vllm arg: {key}={custom_vllm_kwargs[key]}"
                    + f" (type: {type(custom_vllm_kwargs[key]).__name__})"
                )
            except Exception as e:
                logger.warning(f"  Failed to parse vllm arg '{key}={value}': {e}, keeping as string")
                custom_vllm_kwargs[key] = value

    return custom_vllm_kwargs
