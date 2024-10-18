from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoConfig

def load_mllm(pretrained_model_name_or_path, trust_remote_code=False, **kwargs):
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
    tokenizer.processor = processor
    model_type = config.model_type

    if "qwen2_vl" in model_type:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, **kwargs)
    elif "mllama" in model_type:
        from transformers import MllamaForConditionalGeneration
        model = MllamaForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, attn_implementation="eager", **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
    return model, tokenizer, processor