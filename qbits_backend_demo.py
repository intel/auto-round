from auto_round import AutoRound
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m"
path = "/home/wangzhe/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)


bits, group_size, sym = 4, 128, False
# device:Optional["auto", None, "hpu", "cpu", "cuda"]
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size,
                      sym=sym, nsamples=32, seqlen=32, device="cpu", iters=1, scale_dtype="fp32")
autoround.quantize()
output_dir = "./tmp_autoround"
autoround.save_quantized(output_dir, format="auto_round")


from auto_round.auto_quantizer import *
from auto_round.qlinear_qbits import *

quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(
    quantized_model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, use_fast=True)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
