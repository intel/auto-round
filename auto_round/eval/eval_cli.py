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
            "--model", "--model_name", "--model_name_or_path", default="facebook/opt-125m", help="model name or path"
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
            help="lm-eval tasks",
        )
        self.add_argument(
            "--disable_trust_remote_code", action="store_true", help="whether to disable trust_remote_code"
        )
        self.add_argument("--eval_bs", "--bs", "--batch_size", default=None, type=int, help="batch size in evaluation")
        self.add_argument("--eval_task_by_task", action="store_true", help="whether to eval task by task.")
        self.add_argument(
            "--eval_model_dtype", default=None, type=str, help="the torch_dytpe to load the model for evaluation."
        )
        self.add_argument(
            "--limit",
            type=float,
            default=None,
            metavar="N|0<N<1",
            help="Limit the number of examples per task. "
            "If <1, limit is a percentage of the total number of examples.",
        )
        self.add_argument("--eval_backend", default="hf", type=str, help="Use hf backend for evaluation by default.")
        # vllm related arguments
        self.add_argument("--revision", default=None, type=str, help="model revision for vllm")
        self.add_argument("--tokenizer", default=None, type=str, help="tokenizer to use with vllm")
        self.add_argument(
            "--tokenizer_mode", default="auto", type=str, help="tokenizer mode for vllm (e.g. auto/fast/slow)"
        )
        self.add_argument("--tokenizer_revision", default=None, type=str, help="tokenizer revision for vllm")
        self.add_argument("--add_bos_token", action="store_true", help="add BOS token when using vllm")
        self.add_argument("--prefix_token_id", default=None, type=int, help="prefix token id for vllm")
        self.add_argument("--tensor_parallel_size", default=1, type=int, help="tensor parallel size for vllm")
        self.add_argument("--data_parallel_size", default=1, type=int, help="data parallel size for vllm")
        self.add_argument("--quantization", default=None, type=str, help="quantization setting for vllm")
        self.add_argument("--max_gen_toks", default=256, type=int, help="max generation tokens for vllm")
        self.add_argument("--swap_space", default=4, type=float, help="swap space (GB) for vllm")
        self.add_argument("--max_batch_size", default=None, type=int, help="max batch size for vllm")
        self.add_argument("--max_length", default=None, type=int, help="max generation length for vllm")
        self.add_argument("--max_model_len", default=None, type=int, help="maximum model sequence length for vllm")
        self.add_argument(
            "--gpu_memory_utilization", default=0.9, type=float, help="target GPU memory utilization for vllm"
        )
        self.add_argument("--lora_local_path", default=None, type=str, help="local LoRA path for vllm")


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
    assert args.eval_backend in ["hf", "vllm"], "Currently only 'vllm' and 'hf' evaluation backends are supported."

    if args.eval_backend == "vllm":
        try:
            assert isinstance(args.model, str), "vllm evaluation only supports model name or path."
            eval_with_vllm(args)
            return
        except Exception as e:  # pragma: no cover
            print(f"vllm evaluation failed: {e}, fallback to default hf backend evaluation.")
            args.eval_backend = "hf"
            clear_memory()

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
):
    set_cuda_visible_devices(device)
    device_str, parallelism = get_device_and_parallelism(device)

    # load after _eval_int in order to make sure import torch after set CUDA_VISIBLE_DEVICES
    import traceback

    from lm_eval import simple_evaluate as lm_simple_evaluate  # pylint: disable=E0611
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from auto_round.utils import logger

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
    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=device_str,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        parallelize=parallelism,
        trust_remote_code=trust_remote_code,
        dtype=eval_model_dtype,
    )

    if isinstance(tasks, str):
        tasks = tasks.replace(" ", "").split(",")

    from lm_eval.utils import make_table  # pylint: disable=E0611

    res_all = {}
    res_keys = ["results", "versions", "n-shot", "higher_is_better"]
    import time

    st = time.time()
    for task in tasks:
        while retry_times:
            try:
                res = lm_simple_evaluate(
                    model=hflm, model_args=None, device=device_str, tasks=task, batch_size=batch_size, limit=limit
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
    device_str, _ = get_device_and_parallelism(args.device)
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
