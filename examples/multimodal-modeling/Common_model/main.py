import argparse
import sys
sys.path.insert(0, '../../..')
parser = argparse.ArgumentParser()
import os
import torch
import transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import set_seed
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from auto_round.utils import convert_dtype_torch2str
from typing import Dict, Optional, List
OLD_IMAGE_TOKEN = '<image>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import inspect
from PIL import Image

def DataFormating(raw_data, image_folder=None, model_type='qwen'):
    for source in raw_data:
        source_inputs = source['conversations']
        for sentence in source_inputs:
            sentence['from'] = sentence['from'].replace('human', 'user')
            sentence['from'] = sentence['from'].replace('gpt', 'assistant')
            if OLD_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(OLD_IMAGE_TOKEN, '').strip()
                sentence['value'] = OLD_IMAGE_TOKEN + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if 'qwen2' in model_type: # for Qwen2-vl
                    replace_token = '<|vision_start|><|image_pad|><|vision_end|>'
                elif 'mllama' in model_type:
                    replace_token = '<|image|>'
                else:
                    replace_img = os.path.join(image_folder, os.path.basename(source["image"]))
                    replace_token = DEFAULT_IM_START_TOKEN + replace_img + DEFAULT_IM_END_TOKEN + '\n'
                sentence["value"] = sentence["value"].replace(OLD_IMAGE_TOKEN, replace_token)
    return raw_data


def common_preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant.",
    model_type='qwen2'
) -> Dict:
    if 'mllama' in model_type:
        roles = {"user": "<|start_header_id|>user<|end_header_id|>\n", "assistant": "<|start_header_id|>assistant<|end_header_id|>\n"}
        im_start = "<|start_header_id|>"
        im_end = "<|end_header_id|>\n"
        im_dot = '<|eot_id|>'
        text_start = '<|begin_of_text|>'
    else :
        roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
    nl_tokens = '\n'
    _system = 'system'

    # Apply prompt templates
    inputs, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        text, target = "", None
        if 'mllama' in model_type:
            system = text_start + im_start + _system + im_end + nl_tokens + system_message + im_dot
        else:
            system = im_start + _system + nl_tokens + system_message + im_end + nl_tokens
        text += system
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if 'mllama' in model_type:
                _text = role + nl_tokens + \
                    sentence["value"] + im_dot
            else:
                _text = role + nl_tokens + \
                    sentence["value"] + im_end + nl_tokens
            text += _text
        token_length = len(tokenizer(text).input_ids)
        if token_length < max_len:
            text += tokenizer.pad_token * (max_len - token_length)
        else:
            text = tokenizer.decode(tokenizer(text).input_ids[:max_len])
            pass
        inputs.append(text)

    return inputs


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
    if 'qwen2' not in model_type:
        im_start = tokenizer.im_start_id
        im_end = tokenizer.im_end_id
    else:
        im_start = tokenizer('<|im_start|>')
        im_end = tokenizer('<|im_end|>')
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer,
                 max_len: int, image_folder=None, model_type='qwen_vl'):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_folder = image_folder
        print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        if 'qwen' == model_type: # for Qwen-VL
            ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
            ret = dict(
                input_ids=ret["input_ids"][0],
                labels=ret["labels"][0],
                attention_mask=ret["attention_mask"][0],
            )
        else: # Qwen2-VL and Llama-3.2 
            texts = common_preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len, model_type=model_type)
            if 'qwen2' in model_type:
                image_path = os.path.join(f"file://{self.image_folder}", os.path.basename(self.raw_data[i]["image"]))
                image = fetch_image({'image':image_path})
            else:
                image = Image.open(os.path.join(self.image_folder, os.path.basename(self.raw_data[i]["image"]))) #.convert('RGB')
            ret = self.tokenizer.processor(
                text=texts,
                images=image,
                padding=True,
                truncation=True,
                return_tensors="pt",
                # videos=None,
            )
            if 'qwen2' in model_type:
                ret = dict(
                    input_ids=ret["input_ids"][0],
                    # labels=ret["labels"][0],
                    attention_mask=ret["attention_mask"][0],
                    image_grid_thw=ret["image_grid_thw"][0],
                    pixel_values=ret["pixel_values"],
                )
            else:
                ret = dict(
                    input_ids=ret["input_ids"][0],
                    attention_mask=ret["attention_mask"][0],
                    aspect_ratio_ids=ret["aspect_ratio_ids"][0],
                    aspect_ratio_mask=ret["aspect_ratio_mask"][0],
                    cross_attention_mask=ret["cross_attention_mask"][0],
                    pixel_values=ret["pixel_values"][0],
                )
        self.cached_data_dict[i] = ret
        return ret
    
    
from transformers.trainer_utils import RemoveColumnsCollator
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

def set_signature_columns_if_needed(model):
    # Inspect model forward signature to keep only the arguments it accepts.
    model_to_inspect = model
    signature = inspect.signature(model_to_inspect.forward)
    signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    signature_columns += list(set(["label", "label_ids", 'labels']))
    return signature_columns
    
def get_collator_with_removed_columns(model, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        signature_columns = set_signature_columns_if_needed(model)

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            description=description,
            model_name=model.__class__.__name__,
        )
        return remove_columns_collator
    
def get_train_dataloader(train_dataset, model, data_collator=default_data_collator,
                         train_batch_size=1, num_workers=0) -> DataLoader:
    """
    Returns the training [`~torch.utils.data.DataLoader`].

    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    if train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")
    
    if data_collator != default_data_collator:
        data_collator = get_collator_with_removed_columns(model, data_collator, description="training")

    dataloader_params = {
        "batch_size": train_batch_size,
        "collate_fn": data_collator,
        "num_workers": num_workers,
    }

    return DataLoader(train_dataset, **dataloader_params)


if __name__ == '__main__':

    parser.add_argument(
        "--model_name", default="Qwen/Qwen-VL"
    )

    parser.add_argument("--bits", default=4, type=int,
                        help="number of  bits")

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--batch_size", "--bs", default=1, type=int,
                        help="train batch size")

    parser.add_argument("--eval_bs", default=4, type=int,
                        help="eval batch size")

    parser.add_argument("--device", default="auto", type=str,
                        help="The device to be used for tuning. The default is set to auto/None,"
                             "allowing for automatic detection. Currently, device settings support CPU, GPU, and HPU.")

    parser.add_argument("--sym", action='store_true',
                        help="sym quantization")

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
    
    parser.add_argument("--not_use_best_mse", action='store_true',
                        help="To determine whether the quantization should handle vision component.")

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
    
    # ================= Evaluation Related =====================
    # parser.add_argument("--eval-path", type=str, default=None)
    
    parser.add_argument("--eval_dataset", type=str, default="textvqa_val,scienceqa_test_img")

    args = parser.parse_args()

    set_seed(args.seed)

    if args.act_bits <= 8:
        print(
            "Warning, activation quantization is an experiment feature")
    
    if args.act_bits <= 8 and args.deployment_device != "fake":
        assert False, "only support fake mode for activation quantization currently"
        
    if "marlin" in args.deployment_device and args.asym == True:
        assert False, "marlin backend only supports sym quantization, please enable --sym"
        
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
    
    torch.manual_seed(1234)
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code,
                                              padding_side="right", use_fast=False)
    seqlen = args.seqlen
    if hasattr(tokenizer, "model_max_length"):
        if tokenizer.model_max_length < seqlen:
            print(f"change sequence length to {tokenizer.model_max_length} due to the limitation of model_max_length",
                  flush=True)
            seqlen = min(seqlen, tokenizer.model_max_length)
            args.seqlen = seqlen
            
    torch_dtype = "auto"
    if args.model_dtype != None:
        if args.model_dtype == "float16" or args.model_dtype == "fp16":
            torch_dtype = torch.float16
        if args.model_dtype == "bfloat16" or args.model_dtype == "bf16":
            torch_dtype = torch.bfloat16
            
    dtype_str = convert_dtype_torch2str(torch_dtype)
    questions = json.load(open(args.question_file, "r"))
    config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code)
    model_type = config.model_type
    processor = None
    if "mllama" in model_type:
        from transformers import MllamaForConditionalGeneration
        model = MllamaForConditionalGeneration.from_pretrained(args.model_name,
                                                               trust_remote_code=not args.disable_trust_remote_code) # torch_dtype=torch.bfloat16
        processor = AutoProcessor.from_pretrained(args.model_name)
        tokenizer.processor = processor
        default_collator = default_data_collator
    elif 'qwen2' not in model_type: # for Qwen-VL/Qwen-VL-Chat
        tokenizer.pad_token_id = tokenizer.eod_id
        config.use_cache = False
        if dtype_str == "bf16":
            model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=not args.disable_trust_remote_code, bf16=True).eval()
        elif dtype_str == "fp16":
            model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=not args.disable_trust_remote_code, fp16=True).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=not args.disable_trust_remote_code).eval()
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
    else: # for Qwen2-VL-instruct
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info, fetch_image
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch_dtype)
        processor = AutoProcessor.from_pretrained(args.model_name)
        tokenizer.processor = processor
        default_collator = default_data_collator
    
    raw_data = DataFormating(questions, args.image_folder, model_type=model_type)
    dataset = LazySupervisedDataset(raw_data, tokenizer,
                                    max_len=min(args.seqlen, tokenizer.model_max_length), image_folder=args.image_folder)
    dataloader = get_train_dataloader(dataset, model, data_collator=default_collator, train_batch_size=args.batch_size)
    
    from auto_round import (AutoRound,
                            AutoRoundAdam)
    from auto_round.utils import get_multimodal_block_names

    model = model.eval()
    seqlen = args.seqlen

    round = AutoRound
    if args.adam:
        round = AutoRoundAdam

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
    
    autoround = round(model, tokenizer, args.bits, args.group_size, sym=args.sym, batch_size=args.batch_size,
                      dataset=dataloader, seqlen=seqlen, nblocks=args.nblocks, iters=args.iters, lr=args.lr,
                      minmax_lr=args.minmax_lr, enable_quanted_input=not args.disable_quanted_input,
                      amp=not args.disable_amp, nsamples=args.nsamples, not_use_best_mse=args.not_use_best_mse,
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

    deployment_device = args.deployment_device.split(',')
    gpu_formats = []
    for item in deployment_device:
        if "gpu" in item or "auto_gptq" in item or "auto_round" in item:
            gpu_formats.append(item)

    if 'gpu' in deployment_device:
        if lm_head_layer_name in layer_config.keys() and layer_config[lm_head_layer_name]["data_type"] == "int":
            gpu_formats.append("auto_round")
        else:
            gpu_formats.append("auto_gptq")
    gpu_formats = list(set(gpu_formats))

    inplace = True if len(deployment_device) < 2 else False
    eval_folder = None
    for gpu_format in gpu_formats:
        if "round" in gpu_format:
            eval_folder = f'{export_dir}-round'
            autoround.save_quantized(eval_folder, format=gpu_format, use_triton=False, inplace=inplace, processor=processor)
        elif "gptq" in gpu_format:
            eval_folder = f'{export_dir}-gpu'
            autoround.save_quantized(eval_folder, format=gpu_format, use_triton=False, inplace=inplace, processor=processor)

    if 'xpu' in deployment_device:
        autoround.save_quantized(f'{export_dir}-xpu', format="itrex_xpu", use_triton=True, inplace=inplace,
                                 compression_dtype=torch.int8, compression_dim=0, use_optimum_format=False,
                                 device="xpu", processor=processor)
    if "cpu" in deployment_device:
        autoround.save_quantized(output_dir=f'{export_dir}-cpu', format='itrex', inplace=inplace, processor=processor)
    if "fake" in deployment_device:
        model = model.to("cpu")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        if eval_folder is None:
            eval_folder = output_dir

    if not args.disable_eval and "fake" in deployment_device and model_type == "qwen":  ## for Qwen-VL
        model = model.half()
        model = model.to(torch_device)
        datasets=args.eval_dataset.split(',')
        for dataset in datasets:
            if 'vqa' in dataset:
                from mm_evaluation.evaluate_vqa import textVQA_evaluation
                evaluator = textVQA_evaluation(
                    model,
                    dataset_name=dataset,
                    # dataset_path=args.eval_path,
                    tokenizer=tokenizer,
                    batch_size=args.eval_bs,
                    device=str(torch_device)
                )
            elif 'scienceqa' in dataset:
                from mm_evaluation.evaluate_multiple_choice import scienceQA_evaluation
                evaluator = scienceQA_evaluation(
                    model,
                    dataset_name=dataset,
                    # dataset_path=args.eval_path,
                    tokenizer=tokenizer,
                    batch_size=args.eval_bs,
                    device=str(torch_device)
                )


