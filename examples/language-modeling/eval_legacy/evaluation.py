
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path
import torch
import logging
import pprint
import re
import shutil
import transformers
from typing import TYPE_CHECKING, Optional, Union
import time
    
if __name__ == "__main__":
    import sys
    sys.path.insert(0, './')

EXT_TASKS = ['wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new']
fewshots_dict = {}
fewshots_dict['paper'] = {
    "lambada_openai": [0],
    "hellaswag": [0],
    "winogrande": [0],
    "piqa": [0],
    "mmlu": [0],
    "wikitext": [0],
    "truthfulqa_mc1": [0],
    "truthfulqa_mc2": [0],
    "openbookqa": [0],
    "boolq": [0],
    "rte": [0],
    "arc_easy": [0],
    "arc_challenge": [0],
    "gsm8k": [0],
    "ceval-valid": [0],
    "cmmlu": [0],
}
fewshots_dict['leadboard'] = {
    "hellaswag": [10],
    "winogrande": [5],
    "arc_easy": [25],
    "arc_challenge": [25],
    "mmlu": [5],
    "drop": [3],
    "gsm8k": [5],
}
fewshots_dict['all'] = {
    "lambada_openai": [0],
    "hellaswag": [0, 10],
    "winogrande": [0, 5],
    "piqa": [0],
    "coqa": [],  ## coqa is not enabled in llamav1 models
    "truthfulqa_mc1": [0],
    "truthfulqa_mc2": [0],
    "openbookqa": [0],
    "boolq": [0],
    "rte": [0],
    "arc_easy": [0, 25],
    "arc_challenge": [0, 25],
    "mmlu": [0, 5],
    "wikitext": [0],
    "drop": [3],
    "gsm8k": [5]
}


def simple_evaluate(
    model,
    model_args: Optional[Union[str, dict, None]] = None,
    tasks=None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    decontamination_ngrams_path=None,
    write_out: bool = False,
    log_samples: bool = True,
    gen_kwargs: str = None,
    task_manager=None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 1234,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    lm=None
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model, transformers.PreTrainedModel object or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param cache_requests: bool, optional
        Speed up evaluation by caching the building of dataset requests. `None` if not caching.
    :param rewrite_requests_cache: bool, optional
        Rewrites all of the request cache if set to `True`. `None` if not desired.
    :param delete_requests_cache: bool, optional
        Deletes all of the request cache if set to `True`. `None` if not desired.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.

    :return
        Dictionary of results
    """
    from lm_eval.tasks import TaskManager
    import random
    import numpy as np
    import transformers
    import lm_eval.api
    import lm_eval.tasks
    import lm_eval.models
    import lm_eval.api.metrics
    import lm_eval.api.registry
    from lm_eval.evaluator import evaluate
    from lm_eval.logging_utils import add_env_info, get_git_commit_hash
    from lm_eval.tasks import get_task_dict
    
    from lm_eval.utils import (
        eval_logger,
        positional_deprecated,
        run_task_tests,
        simple_parse_args_string,
    )
    
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    if delete_requests_cache:
        eval_logger.info("Deleting requests cache...")
        delete_cache()
    
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        eval_logger.info(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        eval_logger.info(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        eval_logger.info(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)
    
    if tasks is None:
        tasks = []
        assert (
            tasks != []
        ), "No tasks specified, or no tasks found. Please verify the task names."

    if lm == None:
        if gen_kwargs is not None:
            gen_kwargs = simple_parse_args_string(gen_kwargs)
            eval_logger.warning(
                "generation_kwargs specified through cli, these settings will update set parameters in yaml tasks. Ensure 'do_sample=True' for non-greedy decoding!"
            )
            if gen_kwargs == "":
                gen_kwargs = None

        if isinstance(model, str):
            if model_args is None:
                model_args = ""

            elif isinstance(model_args, dict):
                lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
                    model_args,
                    {
                        "batch_size": batch_size,
                        "max_batch_size": max_batch_size,
                        "device": device,
                    },
                )

            else:
                lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
                    model_args,
                    {
                        "batch_size": batch_size,
                        "max_batch_size": max_batch_size,
                        "device": device,
                    },
                )
        elif isinstance(model, transformers.PreTrainedModel):
            lm = lm_eval.api.registry.get_model("hf")(
                pretrained=model,
                batch_size=batch_size,
                max_batch_size=max_batch_size,
            )
            use_cache = None
        else:
            assert isinstance(model, lm_eval.api.model.LM)
            lm = model

        if use_cache is not None:
            print(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
            lm = lm_eval.api.model.CachingLM(
                lm,
                "lm_cache/"
                + (model if isinstance(model, str) else model.model.config._name_or_path)
                + "_"
                + model_args.replace("=", "-").replace(",", "_").replace("/", "-")
                + ".db",
            )
            
    if task_manager is None:
        task_manager = TaskManager(verbosity)

    eval_logger.info(
        "get_task_dict has been updated to accept an optional argument, `task_manager`"
        "Read more here:https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage"
    )
    task_dict = get_task_dict(tasks, task_manager)
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if isinstance(task_obj, tuple):
            _, task_obj = task_obj
            if task_obj is None:
                continue

        if task_obj.get_config("output_type") == "generate_until":
            if gen_kwargs is not None:
                task_obj.set_config(
                    key="generation_kwargs", value=gen_kwargs, update=True
                )

            if predict_only:
                log_samples = True
                eval_logger.info(
                    f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                )
                # we have to change the class properties post-hoc. This is pretty hacky.
                task_obj.override_metric(metric_name="bypass")

        if num_fewshot is not None:
            if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                eval_logger.info(
                    f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                )
            else:
                eval_logger.warning(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )
                task_obj.set_config(key="num_fewshot", value=num_fewshot)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        bootstrap_iters=bootstrap_iters,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        log_samples=log_samples,
        verbosity=verbosity,
    )

    if lm.rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
            "batch_size": batch_size,
            "batch_sizes": (
                list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []
            ),
            "device": device,
            "use_cache": use_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "gen_kwargs": gen_kwargs,
        }
        results["git_hash"] = get_git_commit_hash()
        add_env_info(results)  # additional environment info to results
        return results, lm
    else:
        return None


def eval_model(model_path, tasks=["lambada_openai", "hellaswag", "winogrande", "piqa"],
               eval_bs=32, use_accelerate=True, dtype=None, limit=None, trust_remote_code=True,
               device="cuda:0", seed=0, nsamples=128, mark="paper", excel_file="tmp.xlsx"):
    print("evaluation with official lm-eval", flush=True)
    try:
        import lm_eval.api
        import lm_eval.tasks
        import lm_eval.models
        import lm_eval.api.metrics
        import lm_eval.api.registry
        from lm_eval.evaluator import evaluate
    
    except:
        raise ImportError("""follow requirements to install dependencies.""")
    
    org_s = time.time()
    if dtype == None:
        from eval.utils import convert_dtype_torch2str_hf
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if hasattr(config, "torch_dtype"):
            dtype = convert_dtype_torch2str_hf(config.torch_dtype)
        else:
            dtype = "float16"
    print(f"Using {dtype} as evaluation data type.")
    
    external_tasks = []
    if isinstance(tasks, str):
        tasks = tasks.split(',')
    for each in EXT_TASKS:
        if each in tasks:
            external_tasks.append(each)
            tasks.remove(each)

    results = {}
    model = None
    lm = None

    for tmp_tasks in tasks:
        try:
            num_fewshot = fewshots_dict[mark][tmp_tasks]
            print(f'********* {tmp_tasks} evaluate ************')
            task_s = time.time()
            for shot in num_fewshot:
                model_type = "hf"
                model_args = f'pretrained={model_path},tokenizer={model_path},dtype={dtype},trust_remote_code={trust_remote_code}'
                if 'gpu' in model_path:
                    model_args = f'pretrained={model_path},tokenizer={model_path},dtype={dtype},autogptq=True,gptq_use_triton=True,trust_remote_code={trust_remote_code}'
                if use_accelerate: # bool(re.search("chatglm", model_path.lower()))
                    model_args += f',parallelize=True'

                if "wikitext" in tmp_tasks:
                    tmp_eval_bs = 1
                else:
                    tmp_eval_bs = eval_bs
                tmp_results, lm = simple_evaluate(model=model_type, model_args=model_args, tasks=tmp_tasks,
                                                  num_fewshot=shot, limit=limit, batch_size=tmp_eval_bs,
                                                  max_batch_size=tmp_eval_bs, lm=lm, device=str(device))
                if 'mmlu' in tmp_tasks and 'cmmlu' not in tmp_tasks:
                    sub_name = f'hendrycksTest-* {shot}-shot'
                else:
                    sub_name = f'{tmp_tasks} {shot}-shot'
                print(f'{sub_name}: ')
                pprint.pprint(tmp_results["results"])
                print(f"\n{sub_name} cost time: {time.time() - task_s}\n")
                results[sub_name] = {}
                if 'mmlu' in tmp_tasks and 'cmmlu' not in tmp_tasks:
                    for cata in ['humanities', 'other', 'stem', 'sociology']:
                        results[sub_name][cata] = tmp_results['results'][f'mmlu_{cata}']
                    results[sub_name]['avg'] = tmp_results['results'][f'mmlu']
                else:
                    results[sub_name] = tmp_results['results']
        except Exception as e:
            print(f'********* {tmp_tasks} ERROR ************')
            print(str(e))
            continue

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = lm.model
    # for external tasks
    # maybe adjust for specific model
    # if hasattr(lm.model.config, "max_position_embeddings"):
    #     lm.model.seqlen = lm.model.config.max_position_embeddings
    # else:
    #     ## for llama-1, opt
    #     lm.model.seqlen = 2048

    # if "opt" in model_name:
    #     seqlen = model.config.max_position_embeddings
    #     model.seqlen = model.config.max_position_embeddings
    # else:
    #     seqlen = 2048
    #     model.seqlen = seqlen

    model.seqlen = 2048
    from eval.utils import get_loaders, eval_ppl_same_with_gptq
    for dataset in external_tasks:
        try:
            dataloader, testloader = get_loaders(
                dataset, nsamples=nsamples, seed=seed,
                tokenizer=tokenizer, seqlen=model.seqlen
            )
            ppl = eval_ppl_same_with_gptq(model, testloader, device)
            print(dataset, ppl)

            results.update({dataset: ppl})
        except Exception as e:
            print(str(e))
            continue

    print(results, flush=True)
    print("cost time: ", time.time() - org_s)
    import pickle
    from collections import OrderedDict
    new_dict = OrderedDict()
    new_dict["model"] = "tmp"
    new_dict["paper-avg"] = 0
    new_dict["leaderboard-avg"] = 0
    new_dict["wikitext2"] = 0
    new_dict["ptb-new"] = 0
    new_dict["c4-new"] = 0
    new_dict["wikitext 0-shot_word_perplexity"] = 0
    new_dict["wikitext 0-shot_byte_perplexity"] = 0
    new_dict["wikitext 0-shot_bits_per_byte"] = 0
    
    # Special handling of mmlu for compatibility with the old excel results sequence
    search_str = 'hendrycksTest'
    mmlu_matching_keys = [key for key in results.keys() if search_str.lower() in key.lower()]
    for key in mmlu_matching_keys:
        data = results.pop(key)
        for sub_key in data.keys():
            for sub_sub_key in data[sub_key].keys():
                new_key = key + "-" + sub_key + "-" + sub_sub_key
                if "std" in new_key or "alias" in new_key:
                    continue
                new_key = new_key.split(",")[0]
                new_dict[new_key] = data[sub_key][sub_sub_key]
                new_key = new_key + "_norm"
                new_dict[new_key] = data[sub_key][sub_sub_key]

    for key in results.keys():
        if key == "model" or key == "paper-avg" or key == "leaderboard-avg":
            continue
        data = results[key]
        if not isinstance(data, dict):
            new_dict[key] = results[key]
            continue
        for sub_key in data.keys():
            for sub_sub_key in data[sub_key].keys():
                new_key = key + "_" + sub_sub_key
                if "std" in new_key or "alias" in new_key:
                    continue
                new_key = new_key.split(",")[0]
                new_dict[new_key] = data[sub_key][sub_sub_key]
                    

    import pandas as pd
    df = pd.DataFrame(data=new_dict, index=[0])
    df.to_excel(excel_file)


if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="/models/opt-125m/"
    )
    parser.add_argument(
        "--eval_bs", default=1,
    )
    parser.add_argument(
        "--tasks", default=['wikitext2', 'ptb-new', 'c4-new', 'lambada_openai', 'hellaswag', 'winogrande', 'piqa',
                    "mmlu", "wikitext", "truthfulqa_mc1", "truthfulqa_mc2", "openbookqa", "boolq", "rte",
                    "arc_easy", "arc_challenge"]
    )
    parser.add_argument(
        "--excel_path", default=None,
        help="The path to save eval results with excel format."
    )

    args = parser.parse_args()
    s = time.time()

    test_tasks = args.tasks
    if isinstance(test_tasks, str):
        test_tasks = test_tasks.split(',')
    model_name = args.model_name.rstrip('/')
    if args.excel_path is None:
        excel_name = model_name.split('/')[-1] + ".xlsx"
    else:
        excel_name = args.excel_path
    eval_model(model_path=args.model_name,
               tasks=test_tasks,
               eval_bs=args.eval_bs, limit=None, excel_file=excel_name)

    print("cost time: ", time.time() - s)

