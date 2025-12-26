import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


def is_hpex_available():
    try:
        import habana_frameworks.torch.core as htcore  # pylint: disable=E0401
    except ImportError as e:
        return False
    return True


# TODO: This test case is temporarily commented out since it not tested for a long time. We need to add it back and change it into pytest format.

# class TestAutoRound:
#     @classmethod
#     def setup_class(self):
#         model_name = "facebook/opt-125m"
#         self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

#     @classmethod
#     def teardown_class(self):
#         shutil.rmtree("./saved", ignore_errors=True)
#         shutil.rmtree("runs", ignore_errors=True)

#     def test_autogptq_format_hpu_inference(self):
#         if not is_hpex_available():
#             return
#         try:
#             import auto_gptq
#         except:
#             return
#         bits, group_size, sym = 4, 128, False
#         autoround = AutoRound(
#             self.model,
#             self.tokenizer,
#             bits=bits,
#             group_size=group_size,
#             sym=sym,
#             iters=2,
#             seqlen=2,
#             dataset=dataloader,
#         )
#         autoround.quantize()
#         quantized_model_path = "./saved"

#         autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_gptq")
#         model = (
#             AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
#             .to("hpu")
#             .to(torch.float32)
#         )
#         tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
#         text = "There is a girl who likes adventure,"
#         inputs = tokenizer(text, return_tensors="pt").to(model.device)
#         print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
#         shutil.rmtree("./saved", ignore_errors=True)

#     def test_autoround_format_hpu_inference(self):
#         if not is_hpex_available():
#             return
#         bits, group_size, sym = 4, 128, False
#         autoround = AutoRound(
#             self.model,
#             self.tokenizer,
#             bits=bits,
#             group_size=group_size,
#             sym=sym,
#             iters=2,
#             seqlen=2,
#             dataset=dataloader,
#         )
#         autoround.quantize()
#         quantized_model_path = "./saved"

#         autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")

#         model = (
#             AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto").to("hpu").to(torch.float32)
#         )
#         tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
#         text = "There is a girl who likes adventure,"
#         inputs = tokenizer(text, return_tensors="pt").to(model.device)
#         print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
#         shutil.rmtree("./saved", ignore_errors=True)
