import torch
import sys
sys.path.insert(0, "../..")

from auto_round.auto_quantizer import AutoRoundConfig

from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation import simple_evaluate_user_model
from lm_eval.utils import make_table

# torch.nn.Linear = QuantLinear

if __name__ == "__main__":
    # maybe need to modify transformers/modeling_utils, add "F8_E4M3": torch.float8_e4m3fn to str_to_torch_dtype
    # need to change quant_method to intel/auto-round
    model_name = "opt-125m-w4g128-auto-gptq/"
    batch_size = 16
    tasks = "piqa"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto" 
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    result = simple_evaluate_user_model(model, tokenizer, batch_size=batch_size, tasks=tasks) 
    print(make_table(result))


