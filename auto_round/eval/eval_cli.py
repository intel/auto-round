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
import json
import os
import time

from auto_round.utils import (
    clear_memory,
    get_device_and_parallelism,
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
            help="the device to be used for tuning. "
            "Currently, device settings support CPU, GPU, and HPU."
            "The default is set to cuda:0,"
            "allowing for automatic detection and switch to HPU or CPU."
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
        self.add_argument(
            "--task_configs",
            type=str,
            default=None,
            help=(
                "Optional per-task configuration in JSON or simplified format. "
                "Example JSON: "
                '\'{"gsm8k_llama": {"apply_chat_template": true, "fewshot_as_multiturn": true}, '
                ' "hellaswag": {"num_fewshot": 10}}\' '
                "You can also provide a JSON file path like 'task_configs.json'."
            ),
        )
        self.add_argument(
            "--disable_thinking",
            action="store_true",
            help=("whether to disable thinking mode of chat_template."),
        )
        self.add_argument("--max_length", default=None, type=int, help="max generation length for eval")

        # vllm related arguments
        vllm_args = self.add_argument_group("vllm backend arguments")
        vllm_args.add_argument("--revision", default=None, type=str, help="model revision for vllm")
        vllm_args.add_argument("--tokenizer", default=None, type=str, help="tokenizer to use with vllm")
        vllm_args.add_argument(
            "--tokenizer_mode", default="auto", type=str, help="tokenizer mode for vllm (e.g. auto/fast/slow)"
        )
        vllm_args.add_argument("--tokenizer_revision", default=None, type=str, help="tokenizer revision for vllm")
        vllm_args.add_argument("--add_bos_token", action="store_true", help="add BOS token when using vllm")
        vllm_args.add_argument("--prefix_token_id", default=None, type=int, help="prefix token id for vllm")
        vllm_args.add_argument("--tensor_parallel_size", default=1, type=int, help="tensor parallel size for vllm")
        vllm_args.add_argument("--data_parallel_size", default=1, type=int, help="data parallel size for vllm")
        vllm_args.add_argument("--quantization", default=None, type=str, help="quantization setting for vllm")
        vllm_args.add_argument("--max_gen_toks", default=256, type=int, help="max generation tokens for vllm")
        vllm_args.add_argument("--swap_space", default=4, type=float, help="swap space (GB) for vllm")
        vllm_args.add_argument("--max_batch_size", default=None, type=int, help="max batch size for vllm")
        vllm_args.add_argument("--max_length", default=None, type=int, help="max generation length for vllm")
        vllm_args.add_argument("--max_model_len", default=None, type=int, help="maximum model sequence length for vllm")
        vllm_args.add_argument(
            "--gpu_memory_utilization", default=0.9, type=float, help="target GPU memory utilization for vllm"
        )
        vllm_args.add_argument("--lora_local_path", default=None, type=str, help="local LoRA path for vllm")


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
    is_gguf_file = False
    if os.path.exists(args.model):
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
        from lm_eval.utils import make_table  # pylint: disable=E0401
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        st = time.time()
        res = simple_evaluate_user_model(
            model, tokenizer, tasks=tasks, batch_size=batch_size, device=device_str, limit=args.limit
        )
        print(make_table(res))
        print("evaluation running time=%ds" % (time.time() - st))
    else:
        st = time.time()
        if "auto" in str(batch_size) and args.mllm:
            logger.warning("Batch size 'auto' is not yet supported for hf-multimodal models, reset to 16")
            batch_size = 16
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
    task_configs=None,  # e.g. {"gsm8k": {"apply_chat_template": True, "fewshot_as_multiturn": True}}
    disable_thinking=False,
    max_length=None,  # default to align with model's original setting
):
    """
    Evaluate each LM-eval task sequentially, with optional per-task overrides.

    Args:
        model (str | nn.Module): Model path or loaded model.
        device (str): Device id (e.g. "0" or "cuda:0").
        tasks (list[str] | str): Tasks to run, separated by comma.
        tokenizer: HuggingFace tokenizer.
        batch_size: Eval batch size (default: "auto:8").
        limit: Number of samples or fraction per task.
        task_configs (dict): Optional task-specific settings like fewshot/chat.
    """
    if isinstance(task_configs, str):
        if os.path.isfile(task_configs):
            with open(task_configs, "r") as f:
                task_configs = json.load(f)
        else:
            try:
                task_configs = json.loads(task_configs)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid --task_configs format: {e}")
    elif task_configs is None:
        task_configs = {}

    set_cuda_visible_devices(device)
    device_str, parallelism = get_device_and_parallelism(device)

    # load after _eval_int in order to make sure import torch after set CUDA_VISIBLE_DEVICES
    import traceback

    from lm_eval import simple_evaluate as lm_simple_evaluate  # pylint: disable=E0611
    from lm_eval.models.hf_vlms import HFMultimodalLM
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from auto_round.utils import logger

    if batch_size is None:
        batch_size = "auto:8"

    # -------------------------------
    # Load model (support gguf)
    # -------------------------------
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
        parallelism = False

    # -------------------------------
    # Build LM-eval model wrapper
    # -------------------------------
    if disable_thinking:  ## align with fp-quant
        from functools import partial

        tokenizer.apply_chat_template = partial(tokenizer.apply_chat_template, enable_thinking=False)
    # check the max_length
    init_kwargs = {}
    if max_length is not None:
        init_kwargs["max_length"] = max_length

    if mllm:
        if batch_size is None or batch_size == "auto":
            logger.warning("hf-multimodal models does not support auto currently, reset eval_bs to 16")
            batch_size = 16
        hflm = HFMultimodalLM(
            pretrained=model,
            tokenizer=tokenizer,
            device=device_str,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            parallelize=parallelism,
            trust_remote_code=trust_remote_code,
            dtype=eval_model_dtype,
            **init_kwargs,
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
            **init_kwargs,
        )

    if isinstance(tasks, str):
        tasks = tasks.replace(" ", "").split(",")

    from lm_eval.utils import make_table  # pylint: disable=E0611

    res_all = {}
    res_keys = ["results", "versions", "n-shot", "higher_is_better"]
    import time

    st = time.time()
    for task in tasks:
        task_cfg = task_configs.get(task, {})
        num_fewshot = task_cfg.get("num_fewshot")
        apply_chat_template = task_cfg.get("apply_chat_template", False)
        batch_size = task_cfg.get("batch_size", batch_size)
        fewshot_as_multiturn = task_cfg.get("fewshot_as_multiturn", False)
        logger.info(f"=== Running task: {task} ===")
        logger.info(
            f"Task config: fewshot={num_fewshot}, apply_chat_template={apply_chat_template},"
            f"fewshot_as_multiturn={fewshot_as_multiturn}, batch_size={batch_size}"
        )
        while retry_times:
            try:
                res = lm_simple_evaluate(
                    model=hflm,
                    model_args=None,
                    device=device_str,
                    tasks=task,
                    batch_size=batch_size,
                    limit=limit,
                    num_fewshot=num_fewshot,
                    apply_chat_template=apply_chat_template,
                    fewshot_as_multiturn=fewshot_as_multiturn,
                )
                break
            except Exception as e:
                cuda_error_msg = traceback.format_exc()
                try:
                    ori_batch_sizes = hflm.batch_sizes if hflm.batch_sizes else {"0": 64}
                    try:
                        for k, v in hflm.batch_sizes.items():
                            hflm.batch_sizes[k] = max(v // 2, 1)
                        logger.warning(f"Out of memory, reset batch_size to {hflm.batch_sizes} and re-try.")
                        res = lm_simple_evaluate(
                            model=hflm,
                            model_args=None,
                            device=device_str,
                            tasks=task,
                            batch_size=1,
                            limit=limit,
                            num_fewshot=num_fewshot,
                            apply_chat_template=apply_chat_template,
                            fewshot_as_multiturn=fewshot_as_multiturn,
                        )
                        hflm.batch_sizes = ori_batch_sizes
                    except Exception as e:
                        traceback.print_exc()
                        pass
                except Exception as e:
                    logger.error(cuda_error_msg)
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


def eval_with_vllm(args):
    import time

    from lm_eval import evaluator  # pylint: disable=E0401
    from lm_eval.models.vllm_causallms import VLLM  # pylint: disable=E0401
    from lm_eval.utils import make_table  # pylint: disable=E0401

    st = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device_str, _ = get_device_and_parallelism(args.device_map)
    eval_model_dtype = get_model_dtype(args.eval_model_dtype, "auto")
    if (batch_size := args.eval_bs) is None:
        batch_size = "auto:8"

    vllm_lm = VLLM(
        pretrained=args.model,
        dtype=eval_model_dtype,
        revision=args.revision,
        trust_remote_code=not args.disable_trust_remote_code,
        tokenizer=args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        tokenizer_revision=args.tokenizer_revision,
        add_bos_token=args.add_bos_token,
        prefix_token_id=args.prefix_token_id,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        max_gen_toks=args.max_gen_toks,
        swap_space=args.swap_space,
        batch_size=batch_size,
        max_batch_size=args.max_batch_size,
        max_length=args.max_length,
        max_model_len=args.max_model_len,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        device=device_str,
        data_parallel_size=args.data_parallel_size,
        lora_local_path=args.lora_local_path,
    )
    res = evaluator.simple_evaluate(
        model=vllm_lm,
        tasks=args.tasks,
        limit=args.limit,
    )

    print(make_table(res))
    print("evaluation running time=%ds" % (time.time() - st))
