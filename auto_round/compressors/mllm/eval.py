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

# Copyright 2023 VLMEvalKit Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2024 LMMs-Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
import time
from functools import partial

import numpy as np

from auto_round.logger import logger
from auto_round.utils import LazyImport

vlmeval = LazyImport("vlmeval")

MODEL_TYPE_TO_VLMEVAL_MODEL = {
    # model_name
    "Qwen-VL": dict(cls="QwenVL"),
    "Qwen-VL-Chat": dict(cls="QwenVLChat"),
    "Qwen2-VL": dict(cls="Qwen2VLChat", min_pixels=1280 * 28 * 28, max_pixels=16384 * 28 * 28, verbose=False),
    "Llama-3.2": dict(cls="llama_vision"),
    "Phi-3-vision": dict(cls="Phi3Vision"),
    "Phi-3.5-vision": dict(cls="Phi3_5Vision"),
    "llava_v1.5": dict(cls="LLaVA"),
    "llava_v1.6": dict(cls="LLaVA_Next"),
    "llava-onevision-qwen2": dict(cls="LLaVA_OneVision"),
    "cogvlm2": dict(cls="CogVlm"),
    "SliME": dict(cls="SliME"),
    "Eagle": dict(cls="Eagle"),
    "Molmo": dict(cls="molmo"),
    # config.model_type
    "qwen2_vl": dict(cls="Qwen2VLChat", min_pixels=1280 * 28 * 28, max_pixels=16384 * 28 * 28),
    "qwen": dict(cls="QwenVL"),
    "qwen_chat": dict(cls="QwenVLChat"),
    "llava": dict(cls="LLaVA"),
    "llava_next": dict(cls="LLaVA_Next"),
    "phi3_v": dict(cls="Phi3Vision"),
    "mllama": dict(cls="llama_vision"),
    "glm-4v-9b": dict(cls="GLM4v"),
}


def mllm_eval(
    pretrained_model_name_or_path: str,
    work_dir: str,
    dataset: list,
    data_store_dir: None,
    pack: bool = False,
    use_subtitle: bool = False,
    fps: float = -1,
    nframe: int = 8,
    rerun: bool = False,
    judge: bool = False,
    verbose: bool = False,
    mode: str = "all",
    ignore: bool = False,
):
    model = None
    if data_store_dir is not None:
        if not os.path.exists(data_store_dir):
            oldmask = os.umask(000)
            os.makedirs(data_store_dir, mode=0o777)
            os.umask(oldmask)
        os.environ["LMUData"] = data_store_dir

    model_name = pretrained_model_name_or_path
    if "/" in model_name:
        model_name = model_name[:-1] if model_name[-1] == "/" else model_name
        model_name = model_name.split("/")[-1]

    if model_name in MODEL_TYPE_TO_VLMEVAL_MODEL:
        model_type = model_name
    else:
        model_type = None
        split_name = model_name.split("-")
        for i in range(len(split_name), 0, -1):
            tmp = "-".join(split_name[0:i])
            if tmp in MODEL_TYPE_TO_VLMEVAL_MODEL:
                model_type = tmp
                break
        if model_type is None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
            model_type = config.model_type
            if "chat" in model_name.lower():
                model_type += "_chat"

    kwargs = MODEL_TYPE_TO_VLMEVAL_MODEL[model_type]
    kwargs["model_path"] = pretrained_model_name_or_path
    model_cls = kwargs.pop("cls")
    model_cls = getattr(vlmeval.vlm, model_cls)
    if "verbose" in kwargs:
        kwargs["verbose"] = verbose
    vlmeval.config.supported_VLM[model_name] = partial(model_cls, **kwargs)

    pred_root = os.path.join(work_dir, model_name)
    os.makedirs(pred_root, exist_ok=True)

    st = time.time()
    rt_file = open(f"{pred_root}/{model_name}_eval_cost.txt", "w")
    for dataset_name in dataset:
        task_st = time.time()
        try:
            dataset_kwargs = {}
            if dataset_name in ["MMLongBench_DOC", "DUDE", "DUDE_MINI", "SLIDEVQA", "SLIDEVQA_MINI"]:
                dataset_kwargs["model"] = model_name
            if dataset_name == "MMBench-Video":
                dataset_kwargs["pack"] = pack
            if dataset_name == "Video-MME":
                dataset_kwargs["use_subtitle"] = use_subtitle

            dataset = vlmeval.dataset.build_dataset(dataset_name, **dataset_kwargs)
            if dataset is None:
                logger.error(f"Dataset {dataset_name} is not valid, will be skipped. ")
                continue

            result_file = f"{pred_root}/{model_name}_{dataset_name}.xlsx"
            if fps > 0:  # For Video Dataset, set the fps for priority
                if dataset_name == "MVBench":
                    raise ValueError("MVBench does not support fps setting, please transfer to MVBench_MP4!")
                nframe = 0
            if dataset_name in ["MMBench-Video"]:
                packstr = "pack" if pack else "nopack"
                if nframe > 0:
                    result_file = f"{pred_root}/{model_name}_{dataset_name}_{nframe}frame_{packstr}.xlsx"
                else:
                    result_file = f"{pred_root}/{model_name}_{dataset_name}_{fps}fps_{packstr}.xlsx"
            elif dataset.MODALITY == "VIDEO":
                if pack:
                    logger.info(f"{dataset_name} not support Pack Mode, directly change to unpack")
                    pack = False
                packstr = "pack" if pack else "nopack"
                if nframe > 0:
                    result_file = f"{pred_root}/{model_name}_{dataset_name}_{nframe}frame_{packstr}.xlsx"
                else:
                    result_file = f"{pred_root}/{model_name}_{dataset_name}_{fps}fps_{packstr}.xlsx"
                if dataset_name in ["Video-MME"]:
                    subtitlestr = "subs" if use_subtitle else "nosubs"
                    result_file = result_file.replace(".xlsx", f"_{subtitlestr}.xlsx")

            if dataset.TYPE == "MT":
                result_file = result_file.replace(".xlsx", ".tsv")

            if os.path.exists(result_file) and rerun:
                import re

                pattern = re.compile(f"{model_name}_{dataset_name}_[(openai)|(gpt)|(auxmatch)].*")
                for file_name in os.listdir(pred_root):
                    if pattern.match(file_name):
                        os.remove(os.path.join(pred_root, file_name))

            if model is None:
                model = model_name  # which is only a name

            # Perform the Inference
            if dataset.MODALITY == "VIDEO":
                model = vlmeval.inference_video.infer_data_job_video(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    nframe=nframe,
                    pack=pack,
                    verbose=verbose,
                    subtitle=use_subtitle,
                    fps=fps,
                )
            elif dataset.TYPE == "MT":
                model = vlmeval.inference_mt.infer_data_job_mt(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=verbose,
                    ignore_failed=ignore,
                )
            else:
                model = vlmeval.inference.infer_data_job(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=verbose,
                    ignore_failed=ignore,
                )

            # Set the judge kwargs first before evaluation or dumping
            judge_kwargs = {
                "verbose": verbose,
            }
            if judge is not None:
                judge_kwargs["model"] = judge
            else:
                if dataset.TYPE in ["MCQ", "Y/N"] or vlmeval.smp.listinstr(["MathVerse"], dataset_name):
                    judge_kwargs["model"] = "chatgpt-0125"
                elif vlmeval.smp.listinstr(
                    ["MMVet", "MathVista", "LLaVABench", "MMBench-Video", "MathVision"], dataset_name
                ):
                    judge_kwargs["model"] = "gpt-4-turbo"
                elif vlmeval.smp.listinstr(
                    ["MMLongBench", "MMDU", "DUDE", "DUDE_MINI", "SLIDEVQA", "SLIDEVQA_MINI"], dataset_name
                ):
                    judge_kwargs["model"] = "gpt-4o"
            if "OPENAI_API_KEY_JUDGE" in os.environ and len(os.environ["OPENAI_API_KEY_JUDGE"]):
                judge_kwargs["key"] = os.environ["OPENAI_API_KEY_JUDGE"]
            if "OPENAI_API_BASE_JUDGE" in os.environ and len(os.environ["OPENAI_API_BASE_JUDGE"]):
                judge_kwargs["api_base"] = os.environ["OPENAI_API_BASE_JUDGE"]

            if dataset_name in ["MMMU_TEST"]:
                result_json = vlmeval.utils.result_transfer.MMMU_result_transfer(result_file)
                logger.info(
                    f"Transfer MMMU_TEST result to json for official evaluation, " f"json file saved in {result_json}"
                )  # noqa: E501
                continue
            elif "MMT-Bench_ALL" in dataset_name:
                submission_file = vlmeval.utils.result_transfer.MMTBench_result_transfer(result_file, **judge_kwargs)
                logger.info(
                    f"Extract options from prediction of MMT-Bench FULL split for official evaluation "
                    f"(https://eval.ai/web/challenges/challenge-page/2328/overview), "
                    f"submission file saved in {submission_file}"
                )  # noqa: E501
                continue
            elif "MLLMGuard_DS" in dataset_name:
                logger.info("The evaluation of MLLMGuard_DS is not supported yet. ")  # noqa: E501
                continue
            elif "AesBench_TEST" == dataset_name:
                logger.info(
                    f"The results are saved in {result_file}. "
                    f"Please send it to the AesBench Team via huangyipo@hotmail.com."
                )  # noqa: E501
                continue
            elif dataset_name in ["DocVQA_TEST", "InfoVQA_TEST", "Q-Bench1_TEST", "A-Bench_TEST"]:
                logger.info(
                    f"{dataset_name} is a test split without ground-truth. "
                    "Thus only the inference part is supported for those datasets. "
                )  # noqa: E501

            if dataset_name in [
                "MMBench_TEST_CN",
                "MMBench_TEST_EN",
                "MMBench",
                "MMBench_CN",
                "MMBench_TEST_CN_V11",
                "MMBench_TEST_EN_V11",
                "MMBench_V11",
                "MMBench_CN_V11",
            ]:
                if not vlmeval.smp.MMBenchOfficialServer(dataset_name):
                    logger.error(
                        f"Can not evaluate {dataset_name} on non-official servers, " "will skip the evaluation. "
                    )
                    continue

            if mode == "all":
                eval_results = dataset.evaluate(result_file, **judge_kwargs)
                import pandas as pd

                if eval_results is not None:
                    if not isinstance(eval_results, dict) and not isinstance(eval_results, pd.DataFrame):
                        raise TypeError("Unsupported eval result type")
                    logger.info(f"The evaluation of model {model_name} x dataset {dataset_name} has finished! ")
                    logger.info("Evaluation Results:")
                if isinstance(eval_results, dict):
                    logger.info("\n" + json.dumps(eval_results, indent=4))
                elif isinstance(eval_results, pd.DataFrame):
                    if len(eval_results) < len(eval_results.columns):
                        eval_results = eval_results.T
                    try:
                        import tabulate  # pylint: disable=E0401

                        logger.info("\n" + tabulate.tabulate(eval_results))
                    except:
                        logger.info(eval_results.to_string())
            rt_file.write("%s cost: %.4fs\n" % (dataset_name, time.time() - task_st))
        except Exception as e:
            logger.exception(
                f"Model {model_name} x Dataset {dataset_name} combination failed: {e}, " "skipping this combination."
            )
            continue
    rt_file.write("%d tasks cost: %.4fs\n" % (len(dataset), time.time() - st))
    rt_file.close()


MODEL_TYPE_TO_LMMS_MODEL = {
    # model_name
    "Qwen-VL": "qwen_vl",
    "Qwen2-VL": "qwen2_vl",
    "cogvlm2": "cogvlm2",
    "llava_v1.5": "llava",
    "Llama-3.2": "llama_vision",
    "Phi-3-vision": "phi3v",
    "Phi-3.5-vision": "phi3v",
    # model_type
    "qwen2_vl": "qwen2_vl",
    "qwen": "qwen_vl",
    "llava": "llava",
    "phi3_v": "phi3v",
    "mllama": "llama_vision",
}

_lmms_eval = LazyImport("lmms_eval")


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def lmms_eval(
    model,
    tasks,
    output_dir=None,
    num_fewshot=None,
    limit=None,
    batch_size=1,
    max_batch_size=None,
    device="cpu",
    use_cache=None,
    apply_chat_template=False,
):
    from transformers.utils.versions import require_version

    require_version("lmms_eval", "lmms_eval need to be installed, `pip install lmms_eval`")
    if isinstance(tasks, str):
        tasks = tasks.replace(" ", "").split(",")

    model_name = model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    model_name = model_name.split("/")[-1]

    model_type = None
    split_name = model_name.split("-")
    for i in range(len(split_name), 0, -1):
        tmp = "-".join(split_name[0:i])
        if tmp in MODEL_TYPE_TO_LMMS_MODEL:
            model_type = tmp
            break
    if model_type is None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        model_type = config.model_type

    if model_type not in MODEL_TYPE_TO_LMMS_MODEL:
        raise ValueError(f"{model_type} is not support by lmms.")

    if MODEL_TYPE_TO_LMMS_MODEL[model_type] == "phi3v":
        model_args = f"model_id_name={model}"
    else:
        model_args = f"pretrained={model}"
    if MODEL_TYPE_TO_LMMS_MODEL[model_type] == "llama_vision":
        model_args += f",device_map={device}"

    class CliArgs:
        output_path = output_dir

    results = _lmms_eval.evaluator.simple_evaluate(
        model=MODEL_TYPE_TO_LMMS_MODEL[model_type],
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=device,
        use_cache=use_cache,
        apply_chat_template=apply_chat_template,
        cli_args=CliArgs(),
    )

    # print and save result
    print(_lmms_eval.utils.make_table(results))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        from datetime import datetime

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{model_name}_{now}_result.json")
        json.dump(results, open(output_file, "w"), indent=4, default=_handle_non_serializable)

    return results
