

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from transformers.utils.import_utils import clear_import_cache

# model = AutoModel.from_pretrained("bert-base-uncased")
# modifications to model code
# clear cache to reload modified code
clear_import_cache()
model_name = "Kimi-K2-Instruct-BF16"
model_name = "/models/Qwen3-30B-A3B"
model_name = "facebook/opt-125m"
model_name = "/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "/models/DeepSeek-V2-Lite-Chat/"
model_name = "/data5/yliu7/HF_HOME/unsloth/gpt-oss-20b-BF16"
model_name = "/data4/yliu/unsloth/gpt-oss-120b-BF16"
model_name = "/storage/yiliu7/unsloth/gpt-oss-20b-BF16/"
model_name = "/storage/yiliu7/unsloth/gpt-oss-120b-BF16"
model_name = "/storage/yiliu7/deepseek-ai/DeepSeek-V2-Lite-Chat/"
model_name = "/storage/yiliu7/deepseek-ai/DeepSeek-V2-Lite-Chat/"
model_name = "/storage/yiliu7/unsloth/DeepSeek-R1-BF16/"
model_name = "/mnt/disk8/deepseek-ai/DeepSeek-V2-Lite-Chat"
model_name = "/mnt/disk5/unsloth/DeepSeek-R1-BF16"
model_name = "/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/"
model_name = "/mnt/disk3/hf_models/DeepSeek-V3.1-Terminus"
model_name = "/mnt/disk8/yiliu7/deepseek-ai/DeepSeek-V3.2-Exp"
model_name = "/storage/yiliu7/deepseek-ai/DeepSeek-V3.2"
model_name = "/storage/yiliu7/deepseek-ai/DeepSeek-V3.2/"
model_name = "/storage/yiliu7/DeepSeek-V3.2-4layers/"
# model_name = "/storage/yiliu7/DeepSeek-V3.2-fp8-w4a16/"
# model_name = "/mnt/disk6/hf_models/DeepSeek-V3.1"
# !THIS ONE IS BETTER FOR DS V32
# https://github.com/huggingface/transformers/pull/42767 
# model_name = "/storage/yiliu7/deepseek-ai/DeepSeek-R1/"
# model_name = "/storage/yiliu7/tflsxyy/DeepSeek-V3-bf16-4layers/"
# Hello, my dog isorrionic cannonballoonshak Sovythobiaaugnil admissions navigatorically excessescribed spiral incapac
# Hello, my dog isorrionicALLY casiferatively striking resemblustring tactfully minority553älighthmares Pearkerdor
device = "cpu"


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from transformers.conversion_mapping import register_checkpoint_conversion_mapping

# register_checkpoint_conversion_mapping("deepseek_v3", [], overwrite=True)
# from ds_v2 import *
from ds_v47 import *
# register_checkpoint_conversion_mapping("deepseek_v2", [], overwrite=True)
# 
# from ds_v2 import apply_ds_v2_fixes, apply_ds_v3_fixes
# apply_ds_v2_fixes()
# apply_ds_v3_fixes()
# from ds_v3 import apply_ds_v3_fixes
# apply_ds_v3_fixes()

# <｜User｜>hello<｜Assistant｜></think>Hello! I am DeepSeek.<｜User｜>1+1=?<｜Assistant｜><think>

# </think>

# Okay, I need to the user
# Okay, I need to generate answer my name</think>
# Okay, I need help! It seems like I'm here. Okay, let meed


def fixed_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

fixed_seed(42)



def quant_ar(model, tokenizer, output_dir):

    from auto_round import AutoRound
    # model_name = args.model
    # scheme = "FP8_STATIC"
    scheme = "W4A16"
    autoround = AutoRound(
        model,
        tokenizer,
        scheme=scheme,
        enable_torch_compile=False,
        iters=0,
        low_gpu_mem_usage=True,
        disable_opt_rtn=True,
        ignore_layers="indexer",
        # nsamples=16,
        # static_kv_dtype="fp8",
        # device="hpu",
    )
    model_base_name = model_name.rstrip("/").split("/")[-1]
    # output_dir = args.output_dir
    # if output_dir is None:
    #     output_dir = "/mnt/disk5/hf_models/" + model_base_name + "-" + scheme + "-fp8-kv-2-test"
    print(f"Output dir: {output_dir}")

    model, save_folder = autoround.quantize_and_save(
        output_dir=output_dir,
        # format="llm_compressor",
    )

def check_meta_module(model):
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            if param.device.type == "meta":
                print("Found meta parameter:", pname, "in module:", name, "shape:", param.shape)
                breakpoint()
                raise RuntimeError(
                    f"The model contains some parameters on the meta device (found in module {name}, parameter {name}). "
                )

def main():
    with torch.no_grad():
        trust_remote_code = False
        # trust_remote_code = True
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=trust_remote_code,
            #   _experts_implementation="eager",
            device_map="cpu",\
            # device_map="auto",
        )
        # generate some text
        # Create a four layers model and save it to disk
        # tiny_model = model.model.layers[:4]
        # model.model.layers = tiny_model
        # model.config.num_hidden_layers = 4
        # model.save_pretrained(")
        # exit(0)
        msg = "Hello, AI is "
        msg = "The capital of France is"
        # msg = "<｜begin▁of▁sentence｜><｜User｜>hello, 1 + 1 = ?<｜Assistant｜>"
        # breakpoint()
        # breakpoint()
        model.eval()
        print(model)
        inputs = tokenizer(msg, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=32)
        # encode = tokenizer.encode(msg, return_tensors="pt")
        # outputs = model.generate(encode, max_new_tokens=32)
        
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # output_dir = "/mnt/disk9/hf_models/test-deepseek-r1-fp8-static"
        # output_dir = "/mnt/disk9/hf_models/test-DeepSeek-V2-Lite-Chat"
        output_dir = f"/storage/yiliu7/{model_name.rstrip('/').split('/')[-1]}-fp8-w4a16-4layers"
        # output_dir = f"/storage/yiliu7/{model_name.rstrip('/').split('/')[-1]}-fp8-w4a16"
        # check_meta_module(model)
        quant_ar(model, tokenizer, output_dir=output_dir)
main()


# The capital of France is Paris. Paris is the most populous city in France, with a population of over 12 million people in the metropolitan area. Paris is located in the north-central part of the country, on the River


# The capital of France is Debateilic Eugensonrails RenewedAuthority Cd514-opakosdeckosm油漆的培养 Chargeafelimpseskopanning以降 Steele的主观性地又好oblotstertainment RAShtaNaj灾区 WillisbursEducator分流
# The capital of France is Debateilic Eugensonrails RenewedAuthority Cd514-opakosdeckosm油漆的培养 Chargeafelimpseskopanning以降 Steele的主观性地又好oblotstertainment RAShtaNaj灾区 WillisbursEducator分流
# 'Bearbeiten starredauriapiaarentlyysk Antar的种类繁多deo recreationally Barg587/ui'