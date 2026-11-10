from transformers import AutoTokenizer

from auto_round import AutoRound


class TestCustomizedData:

    def test_list_batch_encoding(self, tiny_qwen_model_path):
        model_name = tiny_qwen_model_path
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        texts = [
            "There is a girl who likes adventure,",
            "Tell me a story about a brave robot,",
            "Explain why the sky is blue,",
        ]
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=9, return_tensors="pt")

        ar = AutoRound(model_name, dataset=[inputs], seqlen=9)
        ar.quantize()
