import argparse
import random
import sys

sys.path.insert(0, '../..')
from auto_round import (AutoRound,
                        AutoAdamRound)

parser = argparse.ArgumentParser()
import torch
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from transformers import set_seed

import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_tokenizer_function(tokenizer, seqlen):
    """Returns a default tokenizer function.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.

    Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length of
    seqlen to the "text" field of examples.
    """

    def default_tokenizer_function(examples):
        example = tokenizer(examples, truncation=True, max_length=seqlen, return_tensors="pt")
        # example = tokenizer(examples, return_tensors="pt")
        return example

    return default_tokenizer_function


def get_dataloader(tokenizer, seqlen, data_name="mbpp", split="train", seed=42, bs=4):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    bs: The batch size for the dataloader.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen)

    @torch.no_grad()
    def collate_batch(batch):
        input_ids_new = []
        attention_mask_new = []
        for text in batch:
            token_text = tokenizer_function(text)
            input_ids, attention_mask = token_text["input_ids"], token_text["attention_mask"]
            if input_ids.shape[1] < seqlen:
                continue
            input_ids = input_ids[:seqlen]
            input_ids_list = input_ids.tolist()
            if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
                continue
            attention_mask = attention_mask[:seqlen]
            attention_mask_new.append(attention_mask)
            input_ids_new.append(input_ids)
        if len(input_ids_new) == 0:
            return None
        input_ids_new = torch.vstack(input_ids_new)
        attention_mask_new = torch.vstack(attention_mask_new)
        res = {"input_ids": input_ids_new, "attention_mask": attention_mask_new}
        return res

    samples = []
    splits = ['train', 'validation', 'test']
    for split in splits:
        dataset = load_dataset(data_name, split=split)
        for data in dataset:
            samples.append(data["text"] + data["code"])
    random.Random(seed).shuffle(samples)

    calib_dataloader = DataLoader(samples, batch_size=bs, shuffle=False, collate_fn=collate_batch)
    return calib_dataloader


if __name__ == '__main__':

    parser.add_argument(
        "--model_name", default="/models/opt-125m"
    )

    parser.add_argument("--bits", default=4, type=int,
                        help="number of  bits")

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--train_bs", default=8, type=int,
                        help="train batch size")

    parser.add_argument("--eval_bs", default=4, type=int,
                        help="eval batch size")

    parser.add_argument("--device", default=0, type=str,
                        help="device gpu int number, or 'cpu' ")

    parser.add_argument("--sym", action='store_true',
                        help=" sym quantization")

    parser.add_argument("--iters", default=200, type=int,
                        help=" iters")

    parser.add_argument("--use_quant_input", action='store_true',
                        help="whether to use the output of quantized block to tune the next block")

    parser.add_argument("--lr", default=None, type=float,
                        help="learning rate, if None, it will be set to 1.0/iters automatially")

    parser.add_argument("--minmax_lr", default=None, type=float,
                        help="minmax learning rate, if None,it will beset to be the same with lr")

    parser.add_argument("--seed", default=42, type=int,
                        help="seed")

    parser.add_argument("--amp", action='store_true',
                        help="amp")

    parser.add_argument("--adam", action='store_true',
                        help="adam")

    parser.add_argument("--seqlen", default=2048, type=int,
                        help="sequence length")

    parser.add_argument("--gradient_accumulate_steps", default=1, type=int, help="gradient accumulate steps")

    parser.add_argument("--n_blocks", default=1, type=int, help="num of blocks to tune together")

    parser.add_argument("--n_samples", default=512, type=int,
                        help="number of samples")

    parser.add_argument("--low_gpu_mem_usage", action='store_true',
                        help="low_gpu_mem_usage")

    parser.add_argument("--enable_minmax_tuning", action='store_true',
                        help="whether enable weight minmax tuning")

    parser.add_argument("--deployment_device", default='fake', type=str,
                        help="targeted inference acceleration platform,The options are 'fake', 'cpu' and 'gpu',"
                             "default to 'fake', indicating that it only performs fake quantization and won't be exported to any device.")

    parser.add_argument("--scale_dtype", default='fp32',
                        help="which scale data type to use for quantization, 'fp16', 'fp32' or 'bf16'.")

    parser.add_argument("--tasks",
                        default=['wikitext2', 'ptb-new', 'c4-new', 'lambada_openai', 'hellaswag', 'winogrande', 'piqa',
                                 "hendrycksTest-*", "wikitext", "truthfulqa_mc", "openbookqa", "boolq", "rte",
                                 "arc_easy", "arc_challenge"],
                        help="lm-eval tasks")

    parser.add_argument("--output_dir", default="./tmp_signround", type=str,
                        help="Where to store the final model.")

    args = parser.parse_args()
    set_seed(args.seed)

    model_name = args.model_name
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    print(model_name, flush=True)

    tasks = args.tasks

    if args.device == "cpu":
        device_str = "cpu"
    else:
        device_str = f"cuda:{int(args.device)}"
    torch_device = torch.device(device_str)
    is_glm = bool(re.search("chatglm", model_name.lower()))
    is_llava = bool(re.search("llava", model_name.lower()))
    if is_llava:
        from transformers import LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype="auto")
    elif is_glm:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype="auto", trust_remote_code=True
        )
    model = model.eval()
    # align wigh GPTQ to eval ppl
    if "opt" in model_name:
        seqlen = model.config.max_position_embeddings
        model.seqlen = model.config.max_position_embeddings
    else:
        seqlen = 2048
        model.seqlen = seqlen
    seqlen = args.seqlen

    if "llama" in model_name:
        from transformers import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(tokenizer, "model_max_length"):
        if tokenizer.model_max_length < seqlen:
            print(f"change sequence length to {tokenizer.model_max_length} due to the limitation of model_max_length",
                  flush=True)
            seqlen = min(seqlen, tokenizer.model_max_length)
            args.seqlen = seqlen

    get_dataloader(tokenizer, args.seqlen)
    excel_name = f"{model_name}_{args.bits}_{args.group_size}"

    if not args.low_gpu_mem_usage:
        model = model.to(torch_device)

    scheme = "asym"
    if args.sym:
        scheme = "sym"
    round = AutoRound
    if args.adam:
        round = AutoAdamRound
    dataloader = get_dataloader(tokenizer, args.seqlen)
    autoround = round(model, tokenizer, args.bits, args.group_size, scheme, dataloader=dataloader, bs=args.train_bs,
                      seqlen=seqlen, n_blocks=args.n_blocks, iters=args.iters, lr=args.lr,
                      minmax_lr=args.minmax_lr, use_quant_input=args.use_quant_input, device=device_str,
                      amp=args.amp, n_samples=args.n_samples, low_gpu_mem_usage=args.low_gpu_mem_usage,
                      seed=args.seed, gradient_accumulate_steps=args.gradient_accumulate_steps,
                      scale_dtype=args.scale_dtype)  ##TODO args pass
    model, q_config = autoround.quantize()
    model_name = args.model_name.rstrip("/")
    export_dir = args.output_dir + "/compressed_" + model_name.split('/')[-1] + "/"
    if args.deployment_device == 'cpu':
        autoround.export(output_dir=export_dir)
        del q_config
    elif args.deployment_device == 'gpu':
        autoround.export(export_dir, target="auto_gptq", use_triton=True)
    if args.device != "cpu":
        torch.cuda.empty_cache()
    model.eval()
    output_dir = args.output_dir + "_" + model_name.split('/')[-1] + f"_w{args.bits}_g{args.group_size}"

    import shutil

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if (hasattr(model, 'config') and model.config.torch_dtype is torch.bfloat16):
        dtype = 'bfloat16'
        pt_dtype = torch.bfloat16
    else:
        pt_dtype = torch.float16

    model = model.to(pt_dtype)
    model = model.to("cpu")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
