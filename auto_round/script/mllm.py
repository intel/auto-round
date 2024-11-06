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
import argparse

import torch
import transformers

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoProcessor

from auto_round.utils import detect_device
from auto_round.utils import logger

class BasicArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.add_argument("--model", default="facebook/opt-125m", help="model name or path")

        self.add_argument('--eval', action='store_true',
                        help="whether to use eval only mode.")

        self.add_argument("--bits", default=4, type=int,
                            help="number of  bits")

        self.add_argument("--eval_bs", default=None, type=int,
                            help="eval batch size")

        self.add_argument("--device", default="auto", type=str,
                            help="The device to be used for tuning. The default is set to auto/None,"
                                "allowing for automatic detection."
                                " Currently, device settings support CPU, GPU, and HPU.")
    
        self.add_argument("--asym", action='store_true',
                            help=" asym quantization")

        self.add_argument("--dataset", type=str, default="llava_v1_5_mix665k",
                            help="The dataset for quantization training. It can be a custom one.")

        self.add_argument("--lr", default=None, type=float,
                            help="learning rate, if None, it will be set to 1.0/iters automatically")

        self.add_argument("--minmax_lr", default=None, type=float,
                            help="minmax learning rate, if None,it will beset to be the same with lr")

        self.add_argument("--seed", default=42, type=int,
                            help="seed")

        self.add_argument("--adam", action='store_true',
                            help="adam")
        
        self.add_argument("--gradient_accumulate_steps", default=1, type=int, help="gradient accumulate steps")

        self.add_argument("--nblocks", default=1, type=int, help="num of blocks to tune together")

        self.add_argument("--low_gpu_mem_usage", action='store_true',
                            help="lower gpu memory usage but 50-100% slower")
        
        self.add_argument("--format", default=None, type=str,
                            help="The format in which to save the model. "
                                "The options are 'auto_round', 'auto_round:gptq','auto_round:awq',"
                                " and 'fake'.default to 'auto_round."
                            )

        self.add_argument("--data_type", default='int',
                            help="data type for tuning, 'int', 'mx_fp' and etc.")

        self.add_argument("--scale_dtype", default='fp16',
                            help="which scale data type to use for quantization, 'fp16', 'fp32' or 'bf16'.")

        self.add_argument("--output_dir", default="./tmp_autoround", type=str,
                            help="Where to store the final model.")

        self.add_argument("--disable_amp", action='store_true',
                            help="disable amp")

        self.add_argument("--disable_minmax_tuning", action='store_true',
                            help="whether disable enable weight minmax tuning")

        self.add_argument("--enable_norm_bias_tuning", action='store_true',
                            help="whether enable norm bias tuning")

        self.add_argument("--disable_trust_remote_code", action='store_true',
                            help="Whether to disable trust_remote_code")

        self.add_argument("--disable_quanted_input", action='store_true',
                            help="whether to disuse the output of quantized block to tune the next block")

        self.add_argument("--quant_lm_head", action='store_true',
                            help="quant_lm_head")

        self.add_argument("--low_cpu_mem_mode", default=0, type=int,
                            help="Choose which low cpu memory mode to use. "
                                "Can significantly reduce cpu memory footprint but cost more time."
                                "1 means choose block-wise mode, load the weights of each block"
                                " from disk when tuning and release the memory of the block after tuning."
                                "2 means choose layer-wise mode, load the weights of each layer from disk when tuning,"
                                " minimum memory consumption and also slowest running speed."
                                "others means not use low cpu memory. Default to 0, not use low cpu memory.")

        self.add_argument("--low_cpu_mem_tmp_dir", default=None, type=str,
                            help="temp work space to store the temporary files "
                                "when using low cpu memory mode. Will remove after tuning.")

        self.add_argument("--model_dtype", default=None, type=str,
                            help="force to convert the dtype, some backends supports fp16 dtype better")

        self.add_argument("--act_bits", default=32, type=int,
                            help="activation bits")

        self.add_argument("--fp_layers_list", default="", type=str,
                            help="List of Layers to maintain original data type")
        
        self.add_argument("--not_use_best_mse", action='store_true',
                        help="To determine whether the quantization should handle vision component.")

        ## ======================= VLM =======================
        self.add_argument("--quant_nontext_module", action='store_true',
                            help="To determine whether the quantization should handle vision component.")

        self.add_argument("--extra_data_dir", default=None, type=str,
                            help="Dataset dir for storing images/audio/videos. "
                            "Can be a dir path or multiple dir path with format as "
                            "'image=path_to_image,video=path_to_video,audio=path_to_audio'"
                            "By default, it will search in the relative path, "
                            "and if not find, will automatic download.")
        
        self.add_argument("--template", default=None, type=str,
                                help="The template for building training dataset. It can be a custom one.")
        
        ## ======================= VLM eval=======================
        self.add_argument("--tasks", type=str,
                          default="MMBench_DEV_CN_V11,MMBench_DEV_EN_V11,ScienceQA_VAL,TextVQA_VAL,POPE,MMMU_DEV_VAL,LLaVABench",
                          help="eval tasks for VLMEvalKit.")
        # Args that only apply to Video Dataset
        self.add_argument("--nframe", type=int, default=8,
                          help="The number of frames to sample from a video,"
                          " only applicable to the evaluation of video benchmarks.")
        self.add_argument("--pack", action='store_true', 
                          help="A video may associate with multiple questions, if pack==True,"
                          " will ask all questions for a video in a single")
        self.add_argument("--use-subtitle", action='store_true')
        self.add_argument("--fps", type=float, default=-1)
        # Work Dir
        # Infer + Eval or Infer Only
        self.add_argument("--mode", type=str, default='all', choices=['all', 'infer'],
                          help="When mode set to 'all', will perform both inference and evaluation;"
                          " when set to 'infer' will only perform the inference.")
        self.add_argument('--eval_data_dir', type=str, default=None,
                          help='path for VLMEvalKit to store the eval data. Default will store in ~/LMUData')
        # API Kwargs, Apply to API VLMs and Judge API LLMs
        self.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
        # Explicitly Set the Judge Model
        self.add_argument('--judge', type=str, default=None)
        # Logging Utils
        self.add_argument('--verbose', action='store_true')
        # Configuration for Resume
        # Ignore: will not rerun failed VLM inference
        self.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
        # Rerun: will remove all evaluation temp files
        self.add_argument('--rerun', action='store_true')

def setup_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--batch_size", default=1, type=int,
                        help="train batch size")

    parser.add_argument("--iters", default=200, type=int,
                        help=" iters")

    parser.add_argument("--seqlen", default=2048, type=int,
                        help="sequence length")

    parser.add_argument("--nsamples", default=128, type=int,
                        help="number of samples")

    args = parser.parse_args()
    return args

def tune(args):
    if args.format is None:
        args.format = "auto_round"
        
    model_name = args.model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    logger.info(f"start to quantize {model_name}")

    assert args.dataset is not None, "dataset should not be None."

    device_str = detect_device(args.device)
    torch_dtype = "auto"
    if "hpu" in device_str:
        torch_dtype = torch.bfloat16
    
    # load_model
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code)
    tokenizer.processor = processor
    model_type = config.model_type
    if "qwen2_vl" in model_type:
        from transformers import Qwen2VLForConditionalGeneration
        cls = Qwen2VLForConditionalGeneration
    elif "mllama" in model_type:
        from transformers import MllamaForConditionalGeneration
        cls = MllamaForConditionalGeneration
    else:
        cls = AutoModelForCausalLM
    model = cls.from_pretrained(
        model_name,trust_remote_code=not args.disable_trust_remote_code, torch_dtype=torch_dtype)
    if "cogvlm2" in model_name:
        model.config.model_type = "cogvlm2"

    from auto_round import AutoRoundMLLM

    model = model.eval()
    seqlen = args.seqlen

    if args.model_dtype != None:
        if args.model_dtype == "float16" or args.model_dtype == "fp16":
            model = model.to(torch.float16)
        if args.model_dtype == "bfloat16" or args.model_dtype == "bfp16":
            model = model.to(torch.bfloat16)

    round = AutoRoundMLLM
    layer_config = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.modeling_utils.Conv1D):
            if m.weight.shape[0] % 32 != 0 or m.weight.shape[1] % 32 != 0:
                layer_config[n] = {"bits": 32}
                logger.info(
                   f"{n} will not be quantized due to its shape not being divisible by 32,"
                    " resulting in an exporting issue to autogptq")
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
                    print(
                        f"warning, disable quant_lm_head as quantizing lm_head with tied weights has not been "
                        f"supported currently")
                    break
    if args.quant_lm_head:
        layer_config[lm_head_layer_name] = {"bits": args.bits}
        transformers_version = [int(item) for item in transformers.__version__.split('.')[:2]]
        if transformers_version[0] == 4 and transformers_version[1] < 38:
            error_message = "Please upgrade transformers>=4.38.0 to support lm-head quantization."
            raise EnvironmentError(error_message)

    if args.quant_lm_head and args.low_gpu_mem_usage:
        print(f"warning, low_gpu_mem_usage=False is strongly recommended if the whole model could be loaded to "
              f"gpu")
    
    autoround = round(model, tokenizer, dataset=args.dataset, extra_data_dir=args.extra_data_dir,
                      bits=args.bits, group_size=args.group_size, sym=not args.asym,
                      batch_size=args.batch_size, seqlen=seqlen, nblocks=args.nblocks, iters=args.iters,
                      lr=args.lr, minmax_lr=args.minmax_lr, enable_quanted_input=not args.disable_quanted_input,
                      amp=not args.disable_amp, nsamples=args.nsamples, low_gpu_mem_usage=args.low_gpu_mem_usage,
                      device=device_str, seed=args.seed, gradient_accumulate_steps=args.gradient_accumulate_steps,
                      scale_dtype=args.scale_dtype, layer_config=layer_config, template=args.template,
                      enable_minmax_tuning=not args.disable_minmax_tuning, act_bits=args.act_bits,
                      quant_nontext_module=args.quant_nontext_module, not_use_best_mse=args.not_use_best_mse)
    model, _ = autoround.quantize()

    model.eval()
    if args.device != "cpu":
        torch.cuda.empty_cache()
    
    export_dir = args.output_dir + "/" + model_name.split('/')[-1] + f"-autoround-w{args.bits}g{args.group_size}"

    format_list = args.format.replace(' ', '').split(',')
    inplace = False if len(format_list) > 1 else True
    for format_ in format_list:
        eval_folder = f'{export_dir}-{format_}'
        if not hasattr(processor, "chat_template"):
            processor.chat_template = None
        safe_serialization = True
        if "phi3_v" in model_type:
            safe_serialization = False
        autoround.save_quantized(
            eval_folder, format=format_, inplace=inplace, processor=processor, safe_serialization=safe_serialization)


def eval(args):
    if isinstance(args.tasks, str):
        args.tasks = args.tasks.replace(' ', '').split(',')
    from auto_round.mllm import mllm_eval
    mllm_eval(
        args.model,
        work_dir=args.output_dir,
        data_store_dir=args.eval_data_dir,
        dataset=args.tasks,
        pack=args.pack,
        use_subtitle = args.use_subtitle,
        fps = args.fps,
        nframe = args.nframe,
        rerun = args.rerun,
        judge = args.judge,
        verbose = args.verbose,
        mode = args.mode,
        ignore = args.ignore
        )