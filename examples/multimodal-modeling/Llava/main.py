import argparse
import sys

sys.path.insert(0, '../../..')

parser = argparse.ArgumentParser()
import torch
import os
import transformers

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import set_seed

import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
from PIL import Image
import json
import math
import shortuuid
from torch.utils.data import Dataset, DataLoader
from llava.mm_utils import get_model_name_from_path
from llava.train.train import preprocess, preprocess_multimodal
from llava.model.builder import load_pretrained_model

class CustomDataset(Dataset): # for llava tuning
    # much refer to https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py
    def __init__(self, list_data_dict, image_folder, tokenizer, image_processor, args):
        self.list_data_dict = list_data_dict
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.args = args

    def __getitem__(self, index):
        sources = self.list_data_dict[index]
        # image
        image_file = os.path.basename(sources["image"])
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        sources = preprocess_multimodal(
            copy.deepcopy([sources["conversations"]]), # a list
            self.args,
        )
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[index])
        )
        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
        # image exist in the data
        data_dict['images'] = image
        return data_dict

    def __len__(self):
        return len(self.list_data_dict)


def create_data_loader(dataset, batch_size=1):
    assert batch_size == 1, "batch_size must be 1"
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    return data_loader

if __name__ == '__main__':

    parser.add_argument(
        "--model_name", default="facebook/opt-125m"
    )

    parser.add_argument("--bits", default=4, type=int,
                        help="number of  bits")

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--train_bs", default=1, type=int,
                        help="train batch size")

    parser.add_argument("--eval_bs", default=4, type=int,
                        help="eval batch size")

    parser.add_argument("--device", default="auto", type=str,
                        help="The device to be used for tuning. The default is set to auto/None,"
                             "allowing for automatic detection. Currently, device settings support CPU, GPU, and HPU.")

    parser.add_argument("--sym", action='store_true',
                        help=" sym quantization")

    parser.add_argument("--iters", default=200, type=int,
                        help=" iters")

    parser.add_argument("--lr", default=None, type=float,
                        help="learning rate, if None, it will be set to 1.0/iters automatically")

    parser.add_argument("--minmax_lr", default=None, type=float,
                        help="minmax learning rate, if None,it will beset to be the same with lr")

    parser.add_argument("--seed", default=42, type=int,
                        help="seed")

    parser.add_argument("--eval_fp16_baseline", action='store_true',
                        help="whether to eval FP16 baseline")

    parser.add_argument("--adam", action='store_true',
                        help="adam")

    parser.add_argument("--seqlen", default=2048, type=int,
                        help="sequence length")

    parser.add_argument("--gradient_accumulate_steps", default=1, type=int, help="gradient accumulate steps")

    parser.add_argument("--nblocks", default=1, type=int, help="num of blocks to tune together")

    parser.add_argument("--nsamples", default=512, type=int,
                        help="number of samples")

    parser.add_argument("--low_gpu_mem_usage", action='store_true',
                        help="low_gpu_mem_usage is deprecated")

    parser.add_argument("--deployment_device", default='fake', type=str,
                        help="targeted inference acceleration platform,The options are 'fake', 'cpu', 'gpu' and 'xpu'."
                             "default to 'fake', indicating that it only performs fake quantization and won't be exported to any device.")

    parser.add_argument("--scale_dtype", default='fp16',
                        help="which scale data type to use for quantization, 'fp16', 'fp32' or 'bf16'.")

    parser.add_argument("--tasks",
                        default="lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1," \
                                "truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge,wikitext2,ptb-new,c4-new",
                        help="lm-eval tasks for lm_eval version 0.4")

    parser.add_argument("--output_dir", default="./tmp_autoround", type=str,
                        help="Where to store the final model.")

    parser.add_argument("--disable_eval", action='store_true',
                        help="Whether to do lmeval evaluation.")

    parser.add_argument("--disable_amp", action='store_true',
                        help="disable amp")

    parser.add_argument("--disable_low_gpu_mem_usage", action='store_true',
                        help="disable low_gpu_mem_usage")

    parser.add_argument("--disable_minmax_tuning", action='store_true',
                        help="whether disable  enable weight minmax tuning")

    parser.add_argument("--disable_trust_remote_code", action='store_true',
                        help="Whether to disable trust_remote_code")

    parser.add_argument("--disable_quanted_input", action='store_true',
                        help="whether to disuse the output of quantized block to tune the next block")

    parser.add_argument("--quant_lm_head", action='store_true',
                        help="quant_lm_head")

    parser.add_argument("--model_dtype", default=None, type=str,
                        help="force to convert the dtype, some backends supports fp16 dtype better")
    
    # ========== Calibration Datasets ============= 
    parser.add_argument("--mm-use-im-start-end", type=bool, default=False)
    
    parser.add_argument("--is_multimodal", type=bool, default=False,
                        help="To determine whether the preprocessing should handle multimodal data.")
    
    parser.add_argument("--image_folder", default="coco", type=str,
                        help="The dataset for quantization training. It can be a custom one.")
    
    parser.add_argument("--question_file", default=None, type=str,
                            help="The dataset for quantization training. It can be a custom one.")
    
    # parser.add_argument("--dataset", default=None, type=str,
    #                     help="The dataset for quantization training. It can be a custom one.")
    
    # ================= Evaluation Related =====================
    parser.add_argument("--eval-question-file", type=str, default="tables/question.jsonl")
    
    parser.add_argument("--eval-image-folder", type=str)
    
    parser.add_argument('--eval-result-file', type=str)
    
    parser.add_argument('--eval-annotation-file', type=str)

    args = parser.parse_args()

    set_seed(args.seed)
    tasks = args.tasks

    model_name = args.model_name
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    print(model_name, flush=True)

    from auto_round.utils import detect_device

    device_str = detect_device(args.device)
    torch_dtype = "auto"
    if "hpu" in device_str:
        torch_dtype = torch.bfloat16
    torch_device = torch.device(device_str)
    model_path = args.model_name
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base=None, model_name=model_name,
            torch_dtype=torch_dtype)
    questions = json.load(open(args.question_file, "r"))
    dataset = CustomDataset(questions, args.image_folder, tokenizer, image_processor, args)
    dataloader = create_data_loader(dataset, args.train_bs)

    from auto_round import (AutoRound,
                            AutoAdamRound)

    model = model.eval()

    if args.model_dtype != None:
        if args.model_dtype == "float16" or args.model_dtype == "fp16":
            model = model.to(torch.float16)
        if args.model_dtype == "bfloat16" or args.model_dtype == "bfp16":
            model = model.to(torch.bfloat16)
            
    seqlen = args.seqlen
    if hasattr(tokenizer, "model_max_length"):
        if tokenizer.model_max_length < seqlen:
            print(f"change sequence length to {tokenizer.model_max_length} due to the limitation of model_max_length",
                  flush=True)
            seqlen = min(seqlen, tokenizer.model_max_length)
            args.seqlen = seqlen

    excel_name = f"{model_name}_{args.bits}_{args.group_size}"
    pt_dtype = torch.float16
    if (hasattr(model, 'config') and (model.dtype is torch.bfloat16 or model.config.torch_dtype is torch.bfloat16)):
        dtype = 'bfloat16'
        pt_dtype = torch.bfloat16
    else:
        if str(args.device) != "cpu":
            pt_dtype = torch.float16
            dtype = 'float16'
        else:
            pt_dtype = torch.float32
            dtype = 'float32'

    if args.eval_fp16_baseline:
        print("Evaluating baseline model")
        if args.disable_low_gpu_mem_usage:
            model = model.to(torch_device)
        from mm_evaluation import TextVQAEvaluator
        evaluator = TextVQAEvaluator(
            model,
            tokenizer,
            image_processor,
            args.eval_image_folder,
            args.eval_question_file,
            args.eval_annotation_file,
            model_name = model_name
        )
        evaluator.run_evaluate(result_file = args.eval_result_file)
        evaluator.calculate_accuracy(result_file = args.eval_result_file)
        exit()

    round = AutoRound
    if args.adam:
        round = AutoAdamRound

    weight_config = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.modeling_utils.Conv1D):
            if m.weight.shape[0] % 32 != 0 or m.weight.shape[1] % 32 != 0:
                weight_config[n] = {"data_type": "fp"}
                print(
                    f"{n} will not be quantized due to its shape not being divisible by 32, resulting in an exporting issue to autogptq")
    lm_head_layer_name = "lm_head"
    for n, _ in model.named_modules():
        lm_head_layer_name = n
    if args.quant_lm_head:
        from transformers import AutoConfig

        config = model.config
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
        weight_config[lm_head_layer_name] = {"data_type": "int"}
        transformers_version = [int(item) for item in transformers.__version__.split('.')[:2]]
        if transformers_version[0] == 4 and transformers_version[1] < 38:
            error_message = "Please upgrade transformers>=4.38.0 to support lm-head quantization."
            raise EnvironmentError(error_message)

    if args.quant_lm_head and not args.disable_low_gpu_mem_usage:
        print(f"warning, disable_low_gpu_mem_usage is strongly recommended if the whole model could be loaded to "
              f"gpu")
    deployment_device = args.deployment_device.split(',')
    gpu_format = "auto_gptq"
    if 'gpu' in deployment_device:
        if lm_head_layer_name in weight_config.keys() and weight_config[lm_head_layer_name]["data_type"] == "int":
            gpu_format = "auto_round"

    if "autoround" in deployment_device or "auto-round" in deployment_device or "auto_round" in deployment_device:
        gpu_format = "auto_round"

    autoround = round(model, tokenizer, args.bits, args.group_size, sym=args.sym, batch_size=args.train_bs,
                      dataset=dataloader, seqlen=seqlen, nblocks=args.nblocks, iters=args.iters, lr=args.lr,
                      minmax_lr=args.minmax_lr, enable_quanted_input=not args.disable_quanted_input, device=device_str,
                      amp=not args.disable_amp, nsamples=args.nsamples,
                      low_gpu_mem_usage=not args.disable_low_gpu_mem_usage,
                      seed=args.seed, gradient_accumulate_steps=args.gradient_accumulate_steps,
                      scale_dtype=args.scale_dtype, weight_config=weight_config,
                      enable_minmax_tuning=not args.disable_minmax_tuning, multimodal=True)
    model, _ = autoround.quantize()
    model_name = args.model_name.rstrip("/")

    model.eval()
    if args.device != "cpu":
        torch.cuda.empty_cache()

    export_dir = args.output_dir + "/" + model_name.split('/')[-1] + f"-autoround-w{args.bits}g{args.group_size}"
    output_dir = args.output_dir + "/" + model_name.split('/')[-1] + f"-autoround-w{args.bits}g{args.group_size}-qdq"
    
    inplace = True if len(deployment_device) < 2 else False
    if 'gpu' in deployment_device or "auto_round" in gpu_format or "auto-round" in gpu_format:
        autoround.save_quantized(f'{export_dir}-gpu', format=gpu_format, use_triton=True, inplace=inplace)
    if 'xpu' in deployment_device:
        autoround.save_quantized(f'{export_dir}-xpu', format="itrex_xpu", use_triton=True, inplace=inplace,
                                 compression_dtype=torch.int8, compression_dim=0, use_optimum_format=False,
                                 device="xpu")
    if "cpu" in deployment_device:
        autoround.save_quantized(output_dir=f'{export_dir}-cpu', format='itrex', inplace=inplace)
    if "fake" in deployment_device:
        model = model.to("cpu")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    if not args.disable_eval and "fake" in deployment_device:  ##support autogptq real eval later
        model = model.half()
        model = model.to(torch_device)
        from mm_evaluation import TextVQAEvaluator
        evaluator = TextVQAEvaluator(
            model,
            tokenizer,
            image_processor,
            args.eval_image_folder,
            args.eval_question_file,
            args.eval_annotation_file,
            model_name = model_name
        )
        evaluator.run_evaluate(result_file = args.eval_result_file)
        evaluator.calculate_accuracy(result_file = args.eval_result_file)


