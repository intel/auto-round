import shutil
import sys
import unittest

sys.path.insert(0, "../..")

from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer


class TestAutoRound(unittest.TestCase):
    def model_infer(self, model, tokenizer):
        prompts = [
            "Hello,my name is",
            # "The president of the United States is",
            # "The capital of France is",
            # "The future of AI is",
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=False, truncation=True)

        outputs = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            do_sample=False,  ## change this to follow official usage
            max_new_tokens=5,
        )
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], outputs)]

        decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i, prompt in enumerate(prompts):
            print(f"Prompt: {prompt}")
            print(f"Generated: {decoded_outputs[i]}")
            print("-" * 50)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("runs", ignore_errors=True)

    def test_load_gptq_no_dummy_gidx_model(self):
        model_name = "/tf_dataset/auto_round/models/ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"
        quantization_config = AutoRoundConfig()
        with self.assertRaises(NotImplementedError) as cm:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype="auto",
                trust_remote_code=True,
                device_map="cpu",
                quantization_config=quantization_config,
            )

    def test_load_awq(self):
        model_name = "/tf_dataset/auto_round/models/casperhansen/opt-125m-awq"
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            trust_remote_code=True,
            device_map="cpu",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model_infer(model, tokenizer)
