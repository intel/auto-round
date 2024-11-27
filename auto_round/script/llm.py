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

from auto_round.utils import detect_device


class BasicArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--model", "--model_name", "--model_name_or_path", default="facebook/opt-125m",
                          help="model name or path")

        self.add_argument('--eval', action='store_true',
                          help="whether to use eval only mode")

        self.add_argument("--bits", default=4, type=int,
                          help="number of weight bits")

        self.add_argument("--eval_bs", default=None, type=int,
                          help="batch size in evaluation")

        self.add_argument("--device", "--devices", default="auto", type=str,
                          help="the device to be used for tuning. The default is set to auto,"
                               "allowing for automatic detection."
                               "Currently, device settings support CPU, GPU, and HPU.")

        self.add_argument("--asym", action='store_true',
                          help="whether to use asym quantization")

        self.add_argument("--dataset", default="NeelNanda/pile-10k", type=str,
                          help="the dataset for quantization training")

        self.add_argument("--lr", default=None, type=float,
                          help="learning rate, if None, it will be set to 1.0/iters automatically")

        self.add_argument("--minmax_lr", default=None, type=float,
                          help="minmax learning rate, if None, it will beset to be the same with lr")

        self.add_argument("--seed", default=42, type=int,
                          help="random seed")

        self.add_argument("--adam", action='store_true',
                          help="whether to use adam optimizer instead of SignSGD")

        self.add_argument("--gradient_accumulate_steps", default=1, type=int,
                          help="gradient accumulate steps")

        self.add_argument("--nblocks", default=1, type=int,
                          help="how many blocks to tune together")

        self.add_argument("--low_gpu_mem_usage", action='store_true',
                          help="offload intermediate features to cpu")

        self.add_argument("--format", default="auto_round", type=str,
                          help="the format to save the model"
                          )

        self.add_argument("--data_type", "--dtype", default='int',
                          help="data type for tuning, 'int', 'mx_fp' and etc")

        self.add_argument("--scale_dtype", default='fp16', choices=["fp16", "float16",
                                                                    "bf16", "bfloat16", "fp32", "float32"],
                          help="scale data type to use for quantization")

        self.add_argument("--tasks",
                          default="lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1," \
                                  "truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge",
                          help="lm-eval tasks")

        self.add_argument("--output_dir", default="./tmp_autoround", type=str,
                          help="the directory to save quantized model")

        self.add_argument("--disable_eval", action='store_true',
                          help="whether to do lm-eval evaluation after tuning")

        self.add_argument("--disable_amp", action='store_true',
                          help="disable amp")

        self.add_argument("--disable_minmax_tuning", action='store_true',
                          help="whether to disable enable weight minmax tuning")

        self.add_argument("--enable_norm_bias_tuning", action='store_true',
                          help="whether to enable norm bias tuning")

        self.add_argument("--disable_trust_remote_code", action='store_true',
                          help="whether to disable trust_remote_code")

        self.add_argument("--disable_quanted_input", action='store_true',
                          help="whether to disuse the output of quantized block to tune the next block")

        self.add_argument("--quant_lm_head", action='store_true',
                          help="whether to quant lm_head")

        self.add_argument("--low_cpu_mem_mode", default=0, type=int, choices=[0, 1, 2],
                          help="choose which low cpu memory mode to use. "
                               "Can significantly reduce cpu memory footprint but cost more time."
                               "1 means choose block-wise mode, load the weights of each block"
                               " from disk when tuning and release the memory of the block after tuning."
                               "2 means choose layer-wise mode, load the weights of each layer from disk when tuning,"
                               " minimum memory consumption and also slowest running speed."
                               "others means not use low cpu memory. Default to 0, not use low cpu memory.")

        self.add_argument("--low_cpu_mem_tmp_dir", default=None, type=str,
                          help="temporary work space to store the temporary files "
                               "when using low cpu memory mode. Will remove after tuning.")

        self.add_argument("--model_dtype", default=None, type=str, choices=["fp16", "float16",
                                                                            "bf16", "bfloat16", "fp32", "float32"],
                          help="force to convert the dtype, some backends supports fp16 dtype better")

        self.add_argument("--act_bits", default=16, type=int,
                          help="activation bits")

        self.add_argument("--fp_layers", default="", type=str,
                          help="list of Layer names to maintain original data type")

        self.add_argument("--not_use_best_mse", action='store_true',
                          help="whether to use the iter of best mes loss in the tuning phase")

        self.add_argument("--to_quant_block_names", default=None, type=str,
                          help="Names of quantitative blocks, please use commas to separate them.")

        self.add_argument("--enable_torch_compile", default=None, type=bool,
                          help="whether to enable torch compile")


def setup_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=8, type=int,
                        help="train batch size")

    parser.add_argument("--iters", "--iter", default=200, type=int,
                        help="iteration to tune each block")

    parser.add_argument("--seqlen", "--seq_len", default=2048, type=int,
                        help="sequence length of the calibration samples")

    parser.add_argument("--nsamples", default=128, type=int,
                        help="number of samples")

    args = parser.parse_args()
    return args


def setup_best_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=8, type=int,
                        help="train batch size")

    parser.add_argument("--iters", "--iter", default=1000, type=int,
                        help="iterations to tune each block")

    parser.add_argument("--seqlen", "--seq_len", default=2048, type=int,
                        help="sequence length of the calibration samples")

    parser.add_argument("--nsamples", default=512, type=int,
                        help="number of samples")

    args = parser.parse_args()
    args.low_gpu_mem_usage = True

    return args


def setup_fast_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=4, type=int,
                        help="train batch size")

    parser.add_argument("--iters", default=200, type=int,
                        help="iterations to tune each block")

    parser.add_argument("--seqlen", "--seq_len", default=512, type=int,
                        help="sequence length of the calibration samples")

    parser.add_argument("--nsamples", default=128, type=int,
                        help="number of samples")

    args = parser.parse_args()

    return args


def tune(args):
    tasks = args.tasks
    if args.format is None:
        args.format = "auto_round"
    supported_formats = ["auto_round", "auto_gptq", "auto_awq", "auto_round:auto_gptq", "auto_round:auto_awq",
                         "auto_gptq:marlin", "itrex", "iterx_xpu", "fake"]
    formats = args.format.replace(' ', '').split(",")
    for format in formats:
        if format not in supported_formats:
            raise ValueError(f"{format} is not supported, we only support {supported_formats}")

    if "auto_gptq" in args.format and args.asym is True:
        print(
            "warning: The auto_gptq kernel has issues with asymmetric quantization. "
            "It is recommended to use sym quantization or --format='auto_round'")

    if "marlin" in args.format and args.asym is True:
        assert False, "marlin backend only supports sym quantization, please remove --asym"

    ##must set this before import torch
    import os
    devices = args.device.replace(" ", "").split(',')
    use_auto_mapping = False
    if all(s.isdigit() for s in devices):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            current_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            current_visible_devices = current_visible_devices.split(',')
            indices = [int(device) for device in devices]
            try:
                pick_device = [current_visible_devices[i] for i in indices]
            except:
                raise ValueError(
                    "Invalid '--device' value: It must be smaller than the number of available devices. "
                    "For example, with CUDA_VISIBLE_DEVICES=4,5, "
                    "--device 0,1 is valid, but --device 4,5 is not supported.")
            visible_devices = ','.join(pick_device)
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
            args.device = ",".join(map(str, range(len(devices))))
            devices = args.device.replace(" ", "").split(',')
        if len(devices) > 1:  ##for 70B model on single card, use auto will cause some layer offload to cpu
            use_auto_mapping = True

    import re
    import torch
    import transformers

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig, AutoProcessor
    from lm_eval.utils import make_table  # pylint: disable=E0401

    from auto_round import AutoRoundConfig
    from auto_round.eval.evaluation import simple_evaluate
    from auto_round.utils import detect_device, get_library_version, detect_device_count
    from auto_round.utils import logger

    model_name = args.model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    logger.info(f"start to quantize {model_name}")
    device_str = detect_device(devices[0])
    torch_dtype = "auto"
    if "hpu" in device_str:
        torch_dtype = torch.bfloat16

    is_glm = bool(re.search("chatglm", model_name.lower()))
    low_cpu_mem_usage = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code)

    model_cls = AutoModel if is_glm else AutoModelForCausalLM

    if args.low_cpu_mem_tmp_dir is None:
        args.low_cpu_mem_tmp_dir = os.path.join(args.output_dir, "low_cpu_mem_tmp")
    if args.low_cpu_mem_mode == 2:
        from auto_round.low_cpu_mem.utils import load_model_with_hooks
        model = load_model_with_hooks(
            model_name,
            model_cls,
            device=device_str,
            clean_weight=True,
            saved_path=args.low_cpu_mem_tmp_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=not args.disable_trust_remote_code
        )
    elif args.low_cpu_mem_mode == 1:
        from auto_round.low_cpu_mem.utils import load_empty_model
        low_cpu_mem_usage = True
        model = load_empty_model(
            model_name,
            model_cls,
            device=device_str,
            saved_path=args.low_cpu_mem_tmp_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=not args.disable_trust_remote_code
        )
    else:
        model = model_cls.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype=torch_dtype,
            trust_remote_code=not args.disable_trust_remote_code, device_map="auto" if use_auto_mapping else None
        )

    from auto_round import AutoRound, AutoRoundAdam

    model = model.eval()
    # align with GPTQ to eval ppl
    seqlen = args.seqlen
    if "opt" in model_name:
        seqlen = model.config.max_position_embeddings
        model.seqlen = model.config.max_position_embeddings
    else:
        seqlen = 2048

    if args.model_dtype != None:
        try:
            if args.model_dtype == "float16" or args.model_dtype == "fp16":
                model = model.to(torch.float16)
            elif args.model_dtype == "bfloat16" or args.model_dtype == "bfp16" or args.model_dtype=="bf16":
                model = model.to(torch.bfloat16)
            elif args.model_dtype=="float32" or args.model_dtype=="fp32":
                model = model.to(torch.float32)
        except:
            logger.error("please use more device to fit the device or just use one device")
            exit()

    if hasattr(tokenizer, "model_max_length"):
        if tokenizer.model_max_length < seqlen:
            logger.info(
                f"change sequence length to {tokenizer.model_max_length} due to the limitation of model_max_length")
            seqlen = min(seqlen, tokenizer.model_max_length)
            args.seqlen = seqlen

    if "bloom" in model_name:
        args.low_gpu_mem_usage = False

    round = AutoRound
    if args.adam:
        round = AutoRoundAdam

    layer_config = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.modeling_utils.Conv1D):
            if m.weight.shape[0] % 32 != 0 or m.weight.shape[1] % 32 != 0:
                layer_config[n] = {"bits": 16}
                logger.info(
                    f"{n} will not be quantized due to its shape not being divisible by 32,"
                    " resulting in an exporting issue to autogptq")

    layer_config = {}
    if args.fp_layers != "":
        fp_layers = args.fp_layers.replace(" ", "").split(",")
        for n, m in model.named_modules():
            if not isinstance(m, (torch.nn.Linear, transformers.modeling_utils.Conv1D)):
                continue
            for fp_layer in fp_layers:
                if fp_layer in n:
                    layer_config[n] = {"bits": 16}
                    logger.info(
                        f"{n} will not be quantized.")
        if len(layer_config) > 0:
            for format in formats:
                if "auto_round" not in format and "fake" not in format:
                    ##TODO gptq, awq could support some mixed precision config
                    logger.warning(f"mixed precision exporting does not support {format} currently")

    lm_head_layer_name = "lm_head"
    for n, _ in model.named_modules():
        lm_head_layer_name = n
    if args.quant_lm_head:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code)
        if config.tie_word_embeddings and hasattr(model, "_tied_weights_keys"):
            tied_keys = model._tied_weights_keys
            for item in tied_keys:
                if lm_head_layer_name in item:  ##TODO extend to encoder-decoder layer, seq classification model
                    args.quant_lm_head = False
                    logger.warning(
                        f"reset `quant_lm_head` to `False` as quantizing lm_head with tied weights has not been "
                        f"supported currently")
                    break
    if args.quant_lm_head:
        layer_config[lm_head_layer_name] = {"bits": args.bits}
        for format in formats:
            if "auto_round" not in format:
                auto_round_formats = [s for s in supported_formats if s.startswith("auto_round")]
                raise ValueError(
                    f"{format} is not supported for lm-head quantization, please change to {auto_round_formats}")

    autoround = round(
        model, tokenizer, args.bits, args.group_size, sym=not args.asym, batch_size=args.batch_size,
        dataset=args.dataset, seqlen=seqlen, nblocks=args.nblocks, iters=args.iters, lr=args.lr,
        minmax_lr=args.minmax_lr, enable_quanted_input=not args.disable_quanted_input,
        device=device_str, amp=not args.disable_amp, nsamples=args.nsamples, seed=args.seed,
        low_gpu_mem_usage=args.low_gpu_mem_usage, scale_dtype=args.scale_dtype,
        gradient_accumulate_steps=args.gradient_accumulate_steps, layer_config=layer_config,
        enable_minmax_tuning=not args.disable_minmax_tuning, act_bits=args.act_bits,
        low_cpu_mem_usage=low_cpu_mem_usage, data_type=args.data_type,
        enable_norm_bias_tuning=args.enable_norm_bias_tuning, not_use_best_mse=args.not_use_best_mse,
        to_quant_block_names=args.to_quant_block_names, enable_torch_compile=args.enable_torch_compile)
    model, _ = autoround.quantize()
    model_name = args.model.rstrip("/")
    if args.low_cpu_mem_mode == 1 or args.low_cpu_mem_mode == 2:
        import shutil
        shutil.rmtree(args.low_cpu_mem_tmp_dir, ignore_errors=True)

    model.eval()
    if "cpu" not in device_str:
        torch.cuda.empty_cache()

    if model_name.split('/')[-1].strip('.') == "":
        export_dir = os.path.join(args.output_dir, f"w{args.bits}g{args.group_size}")
    else:
        export_dir = os.path.join(args.output_dir, model_name.split('/')[-1] + f"-w{args.bits}g{args.group_size}")

    format_list = args.format.replace(' ', '').split(',')
    inplace = False if len(format_list) > 1 else True
    for format_ in format_list:
        save_format_ = format_.replace(":", "-")
        save_format_ = save_format_.replace("_", "-")
        eval_folder = f'{export_dir}-{save_format_}'
        autoround.save_quantized(eval_folder, format=format_, inplace=inplace)

    lm_eval_version = get_library_version("lm-eval")

    if isinstance(tasks, str):
        tasks = tasks.split(',')

    if not args.disable_eval:
        logger.info(f"Using lm-eval version {lm_eval_version}")

        model_args = f"pretrained={eval_folder}"
        model_args = model_args + f",trust_remote_code={not args.disable_trust_remote_code}"
        if args.act_bits <= 8:
            if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                from accelerate.big_modeling import dispatch_model

                dispatch_model(model, model.hf_device_map)
                user_model = model
            else:
                user_model = model.to(device_str)
            if args.eval_bs is None or args.eval_bs == "auto":
                args.eval_bs = 16
            from auto_round.eval.evaluation import simple_evaluate_user_model
            res = simple_evaluate_user_model(user_model, tokenizer, tasks=tasks, batch_size=args.eval_bs)
        else:
            if use_auto_mapping:
                model_args += ",parallelize=True"
            res = simple_evaluate(model="hf", model_args=model_args,
                                  tasks=tasks,
                                  batch_size=args.eval_bs)
        print(make_table(res))


def eval(args):
    import os
    devices = args.device.replace(" ", "").split(',')
    parallelism = False

    if all(s.isdigit() for s in devices):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            current_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            current_visible_devices = current_visible_devices.split(',')
            indices = [int(device) for device in devices]
            try:
                pick_device = [current_visible_devices[i] for i in indices]
            except:
                raise ValueError(
                    "Invalid '--device' value: It must be smaller than the number of available devices. "
                    "For example, with CUDA_VISIBLE_DEVICES=4,5, "
                    "--device 0,1 is valid, but --device 4,5 is not supported.")
            visible_devices = ','.join(pick_device)
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
            args.device = ",".join(map(str, range(len(devices))))
            devices = args.device.replace(" ", "").split(',')
        if len(devices) > 1:
            parallelism = True
        device_str = None
    else:
        device_str = detect_device(args.device.replace(" ", ""))

    from auto_round.eval.evaluation import simple_evaluate

    model_args = f"pretrained={args.model},trust_remote_code={not args.disable_trust_remote_code}"
    if parallelism:
        model_args += ",parallelize=True"
    if isinstance(args.tasks, str):
        tasks = args.tasks.split(',')
    res = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        device=device_str,
        batch_size=args.eval_bs)

    from lm_eval.utils import make_table  # pylint: disable=E0401
    print(make_table(res))
