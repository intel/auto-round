import os
import argparse

import warnings
import torch
import transformers
from transformers import set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
sys.path.insert(0, '../../..')

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="Qwen/Qwen-VL"
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

    parser.add_argument("--adam", action='store_true',
                        help="adam")

    parser.add_argument("--seqlen", default=512, type=int,
                        help="sequence length")

    parser.add_argument("--gradient_accumulate_steps", default=8, type=int, help="gradient accumulate steps")

    parser.add_argument("--nblocks", default=1, type=int, help="num of blocks to tune together")

    parser.add_argument("--nsamples", default=512, type=int,
                        help="number of samples")

    parser.add_argument("--low_gpu_mem_usage", action='store_true',
                        help="low_gpu_mem_usage is deprecated")

    parser.add_argument("--deployment_device", default='fake', type=str,
                        help="targeted inference acceleration platform,The options are 'fake', 'cpu', 'gpu' and 'xpu'."
                             "default to 'fake', indicating that it only performs fake quantization and won't be exported to any device.")

    parser.add_argument("--format", default=None, type=str,
                        help="The format in which to save the model. "
                             "The options are 'auto_round', 'auto_gptq', 'auto_awq', 'itrex', 'itrex_xpu' and 'fake'."
                             "default to 'auto_round."
                        )

    parser.add_argument("--scale_dtype", default='fp16',
                        help="which scale data type to use for quantization, 'fp16', 'fp32' or 'bf16'.")

    parser.add_argument("--output_dir", default="./tmp_autoround", type=str,
                        help="Where to store the final model.")

    parser.add_argument("--disable_eval", action='store_true',
                        help="Whether to do lmeval evaluation.")

    parser.add_argument("--disable_amp", action='store_true',
                        help="disable amp")

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
    
    parser.add_argument("--model_max_length", default=2048, type=int,
                        help="")
    
    parser.add_argument("--act_bits", default=32, type=int,
                    help="activation bits")
    
    parser.add_argument("--quant_vision", action='store_true',
                        help="To determine whether the quantization should handle vision component.")
    
    # ========== Calibration Datasets ============= 
    parser.add_argument("--image_folder", default="coco", type=str,
                        help="The dataset for quantization training. It can be a custom one.")
    
    parser.add_argument("--question_file", default=None, type=str,
                            help="The dataset for quantization training. It can be a custom one.")
    
    parser.add_argument("--template", default=None, type=str,
                            help="The template for building training dataset. It can be a custom one.")

    args = parser.parse_args()
    return args


def main():
    args = setup_args()

    set_seed(args.seed)

    if args.act_bits <= 8:
        print(
            "Warning, activation quantization is an experiment feature")
    
    if args.act_bits <= 8 and args.deployment_device != "fake":
        assert False, "only support fake mode for activation quantization currently"
        
    if args.format and args.deployment_device:
        assert False, "please only specify one of format and deployment_device"

    if args.deployment_device is None and args.format is None:
        args.format = "auto_round"

    if args.deployment_device:
        warnings.warn("The deployment_device is deprecated and will be removed in future version."
                      "Please use format instead", DeprecationWarning)

        if "marlin" in args.deployment_device and args.sym == False:
            assert False, "marlin backend only supports sym quantization, please set --sym"

    if args.format:
        if "marlin" in args.format and args.sym == False:
            assert False, "marlin backend only supports sym quantization, please set --sym"
        
    model_name = args.model_name
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    print(model_name, flush=True)

    from auto_round.utils import detect_device

    device_str = detect_device(args.device)
    torch_dtype = "auto"
    if "hpu" in device_str:
        torch_dtype = torch.bfloat16
    
    torch.manual_seed(1234)
    model_name = args.model_name

    # load_model
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoConfig, Qwen2VLForConditionalGeneration
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.processor = processor
    model_type = config.model_type

    model_cls = AutoModelForCausalLM
    if "qwen2" in model_type:
        from transformers import Qwen2VLForConditionalGeneration
        from transformers.data.data_collator import DataCollatorWithPadding
        model_cls = Qwen2VLForConditionalGeneration
    elif "mllama" in model_type:
        from transformers import MllamaForConditionalGeneration
        model_cls = MllamaForConditionalGeneration
    else:
        model_cls = AutoModelForCausalLM
    
    model = model_cls.from_pretrained(model_name, device_map="auto", torch_dtype=torch_dtype, trust_remote_code=not args.disable_trust_remote_code,)

    from auto_round.vlm_dataset import get_dataloader
    if args.template:
        template = args.template
    else:
        template = model_type
    dataloader = get_dataloader(template, tokenizer, question_path=args.question_file, image_path=args.image_folder, seqlen=args.seqlen)

    from auto_round import (AutoRound,
                            AutoAdamRound)
    from auto_round.utils import get_multimodal_block_names

    model = model.eval()
    seqlen = args.seqlen

    round = AutoRound
    if args.adam:
        round = AutoAdamRound

    layer_config = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.modeling_utils.Conv1D):
            if m.weight.shape[0] % 32 != 0 or m.weight.shape[1] % 32 != 0:
                layer_config[n] = {"bits": 32}
                print(
                    f"{n} will not be quantized due to its shape not being divisible by 32, resulting in an exporting issue to autogptq")
    lm_head_layer_name = "lm_head"
    for n, _ in model.named_modules():
        lm_head_layer_name = n
    if args.quant_lm_head:
        from transformers import AutoConfig

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
        
    quant_block_list = get_multimodal_block_names(model, args.quant_vision)
    
    autoround = round(model, tokenizer, args.bits, args.group_size, sym=args.sym, batch_size=args.train_bs,
                      dataset=dataloader, seqlen=seqlen, nblocks=args.nblocks, iters=args.iters, lr=args.lr,
                      minmax_lr=args.minmax_lr, enable_quanted_input=not args.disable_quanted_input,
                      amp=not args.disable_amp, nsamples=args.nsamples,
                      low_gpu_mem_usage=args.low_gpu_mem_usage, device=device_str,
                      seed=args.seed, gradient_accumulate_steps=args.gradient_accumulate_steps,
                      scale_dtype=args.scale_dtype, layer_config=layer_config,
                      enable_minmax_tuning=not args.disable_minmax_tuning, act_bits=args.act_bits,
                      quant_block_list=quant_block_list)
    model, _ = autoround.quantize()
    model_name = args.model_name.rstrip("/")

    model.eval()
    if args.device != "cpu":
        torch.cuda.empty_cache()
    
    export_dir = args.output_dir + "/" + model_name.split('/')[-1] + f"-autoround-w{args.bits}g{args.group_size}"
    output_dir = args.output_dir + "/" + model_name.split('/')[-1] + f"-autoround-w{args.bits}g{args.group_size}-qdq"

    if args.format:
        format_list = args.format.replace(' ', '').split(',')
        inplace = False if len(format_list) > 1 else True
        for format_ in format_list:
            eval_folder = f'{export_dir}-{format_}'
            autoround.save_quantized(eval_folder, format=format_, inplace=inplace)
    else:
        deployment_device = args.deployment_device.split(',')
        gpu_formats = []
        for item in deployment_device:
            if item in ["gpu", "auto_gptq", "auto_round", "auto_awq"]:
                if item == "gpu":
                    if lm_head_layer_name in layer_config.keys() and layer_config[lm_head_layer_name][
                        "data_type"] == "int":
                        gpu_formats.append("auto_round")
                    else:
                        gpu_formats.append("auto_gptq")
                else:
                    gpu_formats.append(item)
        gpu_formats = list(set(gpu_formats))

        inplace = True if len(deployment_device) < 2 else False
        for gpu_format in gpu_formats:
            eval_folder = f'{export_dir}-{gpu_format}'
            autoround.save_quantized(eval_folder, format=gpu_format, inplace=inplace)

        if 'xpu' in deployment_device:
            autoround.save_quantized(f'{export_dir}-xpu', format="itrex_xpu", use_triton=True, inplace=inplace)
        if "cpu" in deployment_device:
            autoround.save_quantized(output_dir=f'{export_dir}-cpu', format='itrex', inplace=inplace)
        if "fake" in deployment_device:
            model = model.to("cpu")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            if eval_folder is None:
                eval_folder = output_dir

    if (not ('gpu' in deployment_device or len(gpu_formats) > 0)) and 'fake' not in deployment_device:
        print('does not support cpu, xpu model evaluation.')
        exit()  ## does not support cpu,xpu model eval
    
    if args.disable_eval:
        exit()

if __name__ == '__main__':
    main()