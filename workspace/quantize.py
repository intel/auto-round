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

import torch
import transformers

######################## HPU Memory Optimization ###########################
# ensure that unnecessary memory is released during quantization.
os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
os.environ.setdefault("PT_HPU_WEIGHT_SHARING", "0")
if int(os.getenv("WORLD_SIZE", "0")) > 0:
    os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
    os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")
from neural_compressor.torch.utils import is_hpex_available

if is_hpex_available():
    import habana_frameworks.torch.core as htcore
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    from neural_compressor.torch.utils import get_used_hpu_mem_MB

    htcore.hpu_set_env()


def initialize_model_and_tokenizer(model_name_or_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    # using memory mapping with torch_dtype=config.torch_dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=config.torch_dtype)
    # shard model for multi-cards and enable hpu graph
    from neural_compressor.torch.utils import local_rank, logger, world_size

    if world_size > 1:
        ds_inference_kwargs = {
            "dtype": config.torch_dtype,
            "tensor_parallel": {"tp_size": world_size},
        }
        import deepspeed

        ds_model = deepspeed.init_inference(model, **ds_inference_kwargs)
        model = ds_model.module
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Habana FP8 quantization.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name or path"
    )
    parser.add_argument("--output_dir", type=str, default="saved_results", help="model name or path")
    parser.add_argument("--device", type=str, default="hpu", help="device")
    parser.add_argument("--dtype", type=str, default="mx_fp4", help="model name or path")
    parser.add_argument("--quantize", action="store_true", help="whether to quantize model")
    parser.add_argument("--tune", action="store_true", help="whether to autoround model")
    parser.add_argument("--autoround", action="store_true", help="whether to autoround model")
    parser.add_argument("--iters", default=None, type=int, help="iters for autoround.")
    parser.add_argument("--seqlen", default=None, type=int, help="sequence length for autoround.")
    parser.add_argument("--nsamples", default=None, type=int, help="number of samples for autoround.")
    parser.add_argument("--target_bits", default=5, type=float, help="number of samples for autoround.")
    parser.add_argument("--target_loss_ratio", default=1.2, type=float, help="number of samples for autoround.")
    parser.add_argument(
        "--use_hpu_graph", action="store_true", help="whether to use hpu graph mode to accelerate performance"
    )
    parser.add_argument(
        "--enable_block_wise_calibration", action="store_true", help="whether to use block-wise calibration"
    )
    parser.add_argument(
        "--disable_optimum_habana", action="store_true", help="whether to use adapt_transformers_to_gaudi"
    )
    parser.add_argument("--mp_ratio", default="1/3", type=str, help="number of samples for autoround.")
    parser.add_argument("--save", action="store_true", help="whether to save the quantized model")
    parser.add_argument("--load", action="store_true", help="whether to load the quantized model")
    parser.add_argument("--save_path", type=str, default="saved_results", help="path to save the quantized model")
    parser.add_argument("--quant_lm_head", action="store_true", help="performance measurement")
    parser.add_argument("--accuracy", action="store_true", help="accuracy measurement")
    parser.add_argument("--performance", action="store_true", help="performance measurement")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size for accuracy measurement.")
    parser.add_argument("--num_fewshot", default=0, type=int, help="num_fewshot of lm_eval.")
    parser.add_argument(
        "--mxfp8_mod_list",
        type=str,
        nargs="*",
        default=[],  # 默认值
        help="List of module names or patterns for MXFP8 quantization.",
    )
    parser.add_argument(
        "--fp8_mod_list",
        type=str,
        nargs="+",  # 接受一个或多个字符串作为列表
        default=[],  # 默认值
        help="List of module names or patterns for FP8 quantization.",
    )
    parser.add_argument(
        "--bf16_mod_list",
        type=str,
        nargs="+",  # 接受一个或多个字符串作为列表
        default=[],  # 默认值
        help="List of module names or patterns for MXFP8 quantization.",
    )
    parser.add_argument(
        "--dump_stats_path", type=str, default="./hqt_output/measure", help="path and prefix to calibration info file."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",  # 接受一个或多个字符串作为列表
        default=[
            "piqa",
            "hellaswag",
            "mmlu",
            "winogrande",
            "lambada_openai",
        ],  # 默认值
        help="tasks for accuracy validation, text-generation and code-generation tasks are different.",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="NeelNanda/pile-10k", help="dataset name for calibration dataloader"
    )
    parser.add_argument("--limit", type=int, default=None, help="number of samples for accuracy evaluation")
    args = parser.parse_args()
    print("Target data type:", args.dtype)

    model, tokenizer = initialize_model_and_tokenizer(args.model_name_or_path)
    if args.quantize:
        lm_head_config = {
            "group_size": 32 if "mx" in args.dtype else 16,
            "data_type": args.dtype,
            "act_data_type": "fp4_v2_with_global_scale" if "fp4_v2" in args.dtype else args.dtype,
        }
        layer_config = {"lm_head": lm_head_config}
        from auto_round import AutoRound

        autoround = AutoRound(
            model,
            tokenizer,
            device=args.device,
            iters=200 if args.tune else 0,
            low_gpu_mem_usage=True,
            group_size=32 if "mx" in args.dtype else 16,
            data_type=args.dtype,
            act_data_type="fp4_v2_with_global_scale" if "fp4_v2" in args.dtype else args.dtype,
            layer_config=layer_config if args.quant_lm_head else None,
        )
        autoround.quantize()
        model = autoround.model

    if args.autoround:
        from deepspeed.module_inject import (
            LinearAllreduce,
            LinearLayer,
        )

        MXFP4_MODULE_MAPPING = {
            torch.nn.Linear: None,
            torch.nn.EmbeddingBag: None,
            LinearLayer: None,
            LinearAllreduce: None,
        }

        from auto_round import AutoRound

        def match_pattern(name, pattern):
            for pat in pattern:
                if pat in name:
                    return True
            return False

        layer_config = {}
        fp8_config = {
            "bits": 8,
            "data_type": "fp8",
            "act_data_type": "fp8",
        }
        mxfp4_config = {
            "bits": 4,
            "group_size": 32,
            "data_type": "mx_fp4",
            "act_data_type": "mx_fp4",
        }
        mxfp8_config = {
            "bits": 8,
            "group_size": 32,
            "data_type": "mx_fp8",
            "act_data_type": "mx_fp8",
        }
        module_name_to_quantize: list[str] = [
            n for n, m in model.named_modules() if isinstance(m, tuple(MXFP4_MODULE_MAPPING.keys()))
        ]
        for name in module_name_to_quantize:
            if match_pattern(name, args.mxfp8_mod_list):
                layer_config.update({name: mxfp8_config})
            if match_pattern(name, args.fp8_mod_list):
                layer_config.update({name: fp8_config})
        if args.quant_lm_head:
            layer_config.update({"lm_head": mxfp8_config})

        from auto_round import AutoRound

        autoround = AutoRound(
            model,
            tokenizer,
            device=args.device,
            low_gpu_mem_usage=True,
            group_size=32 if "mx" in args.dtype else 16,
            data_type=args.dtype,
            act_data_type="fp4_v2_with_global_scale" if "fp4_v2" in args.dtype else args.dtype,
            layer_config=layer_config,
        )

        recipe_results = autoround._generate_recipe(
            mp_config={
                "mp_ratio": float(eval(args.mp_ratio)),
            },
        )
        autoround.layer_config = recipe_results["recipe"]
        autoround.quantize()
        model = autoround.model

    # preprocess model for accuracy and performance measurement
    if not args.load and not args.autoround and not args.quantize:
        # compare fp8 with bf16, not fp32.
        model = model.to(torch.bfloat16)
    model = model.eval().to(args.device)
    print(model)

    if args.accuracy:
        if is_hpex_available():
            model = wrap_in_hpu_graph(model)
            htcore.hpu_inference_initialize(model, mark_only_scales_as_const=True)
            from neural_compressor.evaluation.lm_eval import LMEvalParser, evaluate

            tasks = ",".join(args.tasks)
            eval_args = LMEvalParser(
                model="hf",
                user_model=model,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                tasks=tasks,
                device="hpu",
                pad_to_buckets=True,
                num_fewshot=args.num_fewshot,
                limit=args.limit,
                add_bos_token=True,
            )
            results = evaluate(eval_args)
            torch.hpu.synchronize()
            all_accuracy = {}
            for task_name, task_results in results["results"].items():
                if task_name in ["hellaswag", "lambada_openai", "piqa", "winogrande", "mmlu"]:
                    accu = task_results["acc,none"]
                    all_accuracy[task_name] = accu
                    print(f"Accuracy for {task_name}: {accu:.4f}")
            print(f"Overall accuracy: {sum(all_accuracy.values())/len(all_accuracy):.4f}")
        else:
            # model = torch.compile(model)
            args.tasks = ["piqa", "hellaswag", "mmlu", "gsm8k"]
            all_accuracy = {}
            test_gsm8k = False
            test_normal = False
            if "gsm8k" in args.tasks:
                test_gsm8k = True
                args.tasks.remove("gsm8k")
            if args.tasks:
                test_normal = True
            import lm_eval
            from lm_eval.models.huggingface import HFLM

            if test_normal:
                lm = HFLM(
                    pretrained=model,
                    tokenizer=tokenizer,
                    add_bos_token=True,
                    batch_size=args.batch_size,
                )
                results = lm_eval.simple_evaluate(
                    lm,
                    tasks=args.tasks,
                    limit=args.limit,
                )
                for task_name, task_results in results["results"].items():
                    if task_name in ["hellaswag", "lambada_openai", "piqa", "winogrande", "mmlu"]:
                        accu = task_results["acc,none"]
                        all_accuracy[task_name] = accu
            ########################## gms8k #########################
            if test_gsm8k:
                lm = HFLM(
                    pretrained=model,
                    tokenizer=tokenizer,
                    add_bos_token=False,
                    batch_size=args.batch_size,
                )
                results_gsm8k = lm_eval.simple_evaluate(
                    lm,
                    tasks=["gsm8k"],
                    limit=args.limit,
                )
                for task_name, task_results in results_gsm8k["results"].items():
                    accu = task_results["exact_match,strict-match"]
                    all_accuracy[task_name] = accu
            ########################## gms8k end #########################
            for task_name, accu in all_accuracy.items():
                print(f"Accuracy for {task_name}: {accu:.4f}")
            print(f"Overall accuracy: {sum(all_accuracy.values())/len(all_accuracy):.4f}")
