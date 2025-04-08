import copy
import shutil
import sys
import unittest

from auto_round.eval.evaluation import simple_evaluate_user_model

sys.path.insert(0, "..")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round import AutoRoundConfig


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "/models/opt-125m"
        ##self.model_name = "/data5/wenhuach/Meta-Llama-3.1-8B-Instruct-int4-sym-inc"

        self.llm_dataloader = LLMDataLoader()
        self.save_folder = "./saved"

    def model_infer(self,model,tokenizer):
        prompts = [
            "Hello,my name is",
            # "The president of the United States is",
            # "The capital of France is",
            # "The future of AI is",
        ]

        ##texts = []
        # for prompt in prompts:
        #     messages = [
        #         {"role": "user", "content": prompt}
        #     ]
        #     text = tokenizer.apply_chat_template(
        #         messages,
        #         tokenize=False,
        #         add_generation_prompt=True
        #     )
        #     texts.append(text)

        inputs = tokenizer(prompts, return_tensors="pt", padding=False, truncation=True)

        outputs = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            do_sample=False,  ## change this to follow official usage
            max_new_tokens=5
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], outputs)
        ]

        decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i, prompt in enumerate(prompts):
            print(f"Prompt: {prompt}")
            print(f"Generated: {decoded_outputs[i]}")
            print("-" * 50)


    @classmethod
    def tearDownClass(self):
        return
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_device_map(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "/data5/wenhuach/Meta-Llama-3.1-8B-Instruct-int4-sym-inc"

        device_map = {}
        for i in range(0, 32):
            key = f"model.layers.{str(i)}"
            device_map[key] = "cuda"
        device_map["model.layers.1"] = "cpu"
        device_map["model.layers.2"] = "cpu"
        device_map["model.layers.3"] = "cpu"
        device_map["model.layers.4"] = "cpu"
        device_map["model.layers.5"] = "cpu"
        device_map["model.layers.6"] = "cpu"
        device_map["model.layers.27"] = "cpu"
        device_map["model.layers.28"] = "cpu"
        device_map["model.layers.30"] = "cpu"
        device_map["model.layers.31"] = "cpu"
        device_map["lm_head"] = "cuda"
        device_map["model.norm"] = "cuda"
        device_map["model.rotary_emb"] = "cuda"
        device_map["model.embed_tokens"] = "cuda"

        device_map1 = {}
        for i in range(0, 32):
            key = f"model.layers.{str(i)}"
            device_map1[key] = "cuda"
        device_map1["model.layers.1"] = "cuda:1"
        device_map1["model.layers.2"] = "cuda:1"
        device_map1["model.layers.3"] = "cuda:1"
        device_map1["model.layers.4"] = "cuda:1"
        device_map1["model.layers.5"] = "cuda:1"
        device_map1["model.layers.6"] = "cuda:1"
        device_map1["model.layers.27"] = "cuda:1"
        device_map1["model.layers.28"] = "cuda:1"
        device_map1["model.layers.30"] = "cuda:1"
        device_map1["model.layers.31"] = "cuda:1"
        device_map1["lm_head"] = "cuda:1"
        device_map1["model.norm"] = "cuda"
        device_map1["model.rotary_emb"] = "cuda"
        device_map1["model.embed_tokens"] = "cuda"

        # for tmp_device_map in [device_map1,device_map,None, 0, "balanced", "balanced_low_0", "sequential", "cpu", "cuda:0", "cuda", "auto",
        #                        ]:
        for tmp_device_map in [device_map]:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=tmp_device_map
                # max_memory={0: "5GB", "cpu": "32GB"},
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            prompts = [
                "Hello,my name is",
                # "The president of the United States is",
                # "The capital of France is",
                # "The future of AI is",
            ]

            texts = []
            for prompt in prompts:
                messages = [
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)

            inputs = tokenizer(texts, return_tensors="pt", padding=False, truncation=True)

            outputs = model.generate(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                do_sample=False,  ## change this to follow official usage
                max_new_tokens=5
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], outputs)
            ]

            decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i, prompt in enumerate(prompts):
                print(f"Prompt: {prompt}")
                print(f"Generated: {decoded_outputs[i]}")
                print("-" * 50)
            model = None
            del model
            torch.cuda.empty_cache()

    #
    # def test_awq_backend(self):
    #     model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
    #     tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
    #     bits, group_size, sym = 4, 128, True
    #     autoround = AutoRound(
    #         model,
    #         tokenizer,
    #         bits=bits,
    #         group_size=group_size,
    #         iters=1,
    #         nsamples=1,
    #         sym=sym,
    #     )
    #     quantized_model_path = self.save_folder
    #     autoround.quantize_and_save(output_dir=quantized_model_path,format="auto_round:auto_awq")
    #
    #     quantization_config = AutoRoundConfig(backend="auto")
    #     # model = AutoModelForCausalLM.from_pretrained(
    #     #     self.save_folder,
    #     #     torch_dtype=torch.float16,
    #     #     device_map="auto",
    #     #     quantization_config=quantization_config
    #     # )
    #     #
    #     # tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
    #     # self.model_infer(model, tokenizer)
    #     # result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
    #     # print(result['results']['lambada_openai']['acc,none'])
    #     # self.assertGreater(result['results']['lambada_openai']['acc,none'], 0.18)
    #     torch.cuda.empty_cache()
    #
    #     model = AutoModelForCausalLM.from_pretrained(
    #         self.save_folder,
    #         torch_dtype=torch.bfloat16,
    #         device_map="auto",
    #         quantization_config=quantization_config
    #     )
    #
    #     tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
    #     self.model_infer(model, tokenizer)


    # def test_tritonv2_bf16(self):
    #     model_name = "/data5/wenhuach/Meta-Llama-3.1-8B-Instruct-int4-sym-inc"
    #     quantization_config = AutoRoundConfig(backend="gptq:tritonv2")
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         torch_dtype=torch.bfloat16,
    #         device_map="auto",
    #         quantization_config=quantization_config
    #     )
    #
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     self.model_infer(model,tokenizer)
    #
    #     torch.cuda.empty_cache()

    #
    # def test_tritonv2_fp16(self):
    #     model_name = "/data5/wenhuach/Meta-Llama-3.1-8B-Instruct-int4-sym-inc"
    #     quantization_config = AutoRoundConfig(backend="tritonv2")
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #         quantization_config=quantization_config
    #     )
    #
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     self.model_infer(model,tokenizer)
    #
    #     torch.cuda.empty_cache()



    ##TODO add asym later
    #
    # def test_autoround_gptq_sym_format(self):
    #     model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
    #     tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
    #     bits, group_size, sym = 4, 128, True
    #     autoround = AutoRound(
    #         model,
    #         tokenizer,
    #         bits=bits,
    #         group_size=group_size,
    #         sym=sym,
    #         iters=2,
    #         seqlen=2,
    #         dataset=self.llm_dataloader,
    #     )
    #     quantized_model_path = "./saved"
    #
    #     autoround.quantize_and_save(output_dir=quantized_model_path)
    #
    #     from auto_round import AutoRoundConfig
    #     quantization_config = AutoRoundConfig(backend="ipex_gptq")
    #
    #     model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu", trust_remote_code=True,
    #                                                  quantization_config=quantization_config)
    #     tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    #     text = "There is a girl who likes adventure,"
    #     inputs = tokenizer(text, return_tensors="pt").to(model.device)
    #     res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
    #     print(res)
    #     assert ("!!!" not in res)
    #
    #
    #     model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
    #     tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    #     text = "There is a girl who likes adventure,"
    #     inputs = tokenizer(text, return_tensors="pt").to(model.device)
    #     res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
    #     print(res)
    #     assert ("!!!" not in res)
    #
    #     model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu", trust_remote_code=True,
    #                                                  quantization_config=quantization_config)
    #     tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    #     text = "There is a girl who likes adventure,"
    #     inputs = tokenizer(text, return_tensors="pt").to(model.device)
    #     res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
    #     print(res)
    #     assert ("!!!" not in res)
    #
    #
    #
    #     shutil.rmtree("./saved", ignore_errors=True)
    #
    #
    # def test_autoround_awq_sym_format(self):
    #     model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
    #     tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
    #     bits, group_size, sym = 4, 128, True
    #     autoround = AutoRound(
    #         model,
    #         tokenizer,
    #         bits=bits,
    #         group_size=group_size,
    #         sym=sym,
    #         iters=2,
    #         seqlen=2,
    #         dataset=self.llm_dataloader,
    #     )
    #     quantized_model_path = "./saved"
    #
    #     autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:auto_awq")
    #
    #     model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
    #     tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    #     text = "There is a girl who likes adventure,"
    #     inputs = tokenizer(text, return_tensors="pt").to(model.device)
    #     res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
    #     print(res)
    #     assert ("!!!" not in res)
    #
    #
    #     model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu", trust_remote_code=True,
    #                                                  torch_dtype=torch.bfloat16)
    #     tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    #     text = "There is a girl who likes adventure,"
    #     inputs = tokenizer(text, return_tensors="pt").to(model.device)
    #     res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
    #     print(res)
    #     assert ("!!!" not in res)
    #
    #     shutil.rmtree("./saved", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
