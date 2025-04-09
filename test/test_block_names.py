import copy
import shutil
import sys
import unittest
sys.path.insert(0, ".")
sys.path.insert(0, "..")
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

class LLMDataLoader:
    def __init__(self, input_size=10):
        self.batch_size = 1
        self.input_size = input_size
    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, self.input_size], dtype=torch.long)
            


# ================= simple multimodal model =================
class TextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TextEncoder, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        
    def forward(self, x):
        return self.fc(x)

class VisionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VisionEncoder, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        
    def forward(self, x):
        return self.fc(x)

class TextEncoderModuleList(nn.ModuleList):
    pass

class VisionEncoderModuleList(nn.ModuleList):
    pass

class SimpleMultimodalModel(nn.Module):
    def __init__(self, text_input_size, image_input_size, hidden_size, num_text_encoders, num_image_encoders):
        super(SimpleMultimodalModel, self).__init__()
        self.text_encoders = TextEncoderModuleList([TextEncoder(text_input_size, hidden_size) for _ in range(num_text_encoders)])
        self.vision_encoders = VisionEncoderModuleList([VisionEncoder(image_input_size, hidden_size) for _ in range(num_image_encoders)])
        self.classifier = nn.Linear(hidden_size * 2, 1)  # Binary classification task

    def forward(self, text_input, image_input):
        text_features = [encoder(text_input) for encoder in self.text_encoders]
        image_features = [encoder(image_input) for encoder in self.vision_encoders]
        
        # Summing the outputs of all text encoders and image encoders
        combined_text_features = torch.stack(text_features, dim=1).sum(dim=1)
        combined_image_features = torch.stack(image_features, dim=1).sum(dim=1)
        
        combined_features = torch.cat((combined_text_features, combined_image_features), dim=1)
        output = self.classifier(combined_features)
        return output


# ================= simple MoE model =================
class Expert(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        
    def forward(self, x):
        return self.fc(x)

class ExpertGroupModuleList(nn.ModuleList):
    pass

class ExpertModuleList(nn.ModuleList):
    pass

class NestedMoEModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, experts_per_group):
        super(NestedMoEModel, self).__init__()
        self.expert_groups = ExpertModuleList([
            ExpertGroupModuleList([Expert(input_size, hidden_size) for _ in range(experts_per_group)])
            for _ in range(num_groups)
        ])
        self.gate = nn.Linear(input_size, num_groups * experts_per_group)

    def forward(self, x):
        gate_outputs = torch.softmax(self.gate(x), dim=1)
        expert_outputs = []

        for group in self.expert_groups:
            group_outputs = [expert(x) for expert in group]
            group_output_sum = torch.stack(group_outputs, dim=1).sum(dim=1)
            expert_outputs.append(group_output_sum)

        expert_outputs = torch.stack(expert_outputs, dim=1).view(x.size(0), -1)
        output = torch.sum(gate_outputs.unsqueeze(2) * expert_outputs.unsqueeze(1), dim=1)
        return output


class TestQuantizationBlocks(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "MBZUAI/LaMini-GPT-124M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)


    def test_moe_quant(self):
        input_size = 10
        hidden_size = 10
        num_groups = 2
        experts_per_group = 2
        self.llm_dataloader = LLMDataLoader(input_size)
        self.model = NestedMoEModel(input_size, hidden_size, num_groups, experts_per_group)
        from auto_round.utils import get_multimodal_block_names, get_block_names
        llm_block_names = get_block_names(self.model)
        all_block_names = []
        try:
            all_block_names = get_multimodal_block_names(self.model, quant_vision=True)
        except:
            pass
        assert len(llm_block_names) != len(all_block_names)
        

    def test_multimodal_quant(self):
        num_text_encoders = 1
        num_image_encoders = 1
        image_input_size = 10
        text_input_size = 10
        hidden_size = 10
        self.model = SimpleMultimodalModel(text_input_size, image_input_size, hidden_size, num_text_encoders, num_image_encoders)
        from auto_round.utils import get_multimodal_block_names, get_block_names
        llm_block_names = get_block_names(self.model)
        block_names_wo_vision = get_multimodal_block_names(self.model, quant_vision=False)
        block_names_with_vision = get_multimodal_block_names(self.model, quant_vision=True)
        assert block_names_wo_vision == llm_block_names
        assert len(block_names_wo_vision) != (block_names_with_vision)
        
    
    def test_block_name_quant(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        from auto_round.utils import get_block_names, validate_modules
        llm_block_names = get_block_names(self.model)
        validate_modules(llm_block_names)
        bits, group_size, sym, batch_size = 4, 128, False, 20
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            batch_size=batch_size,
            dataset=self.llm_dataloader,
            to_quant_block_names=llm_block_names
        )
        autoround.quantize()
        try:
            import auto_gptq
        except:
            return
        if not torch.cuda.is_available():
            return
        quantized_model_path = "./saved"
        autoround.save_quantized(quantized_model_path, inplace=False, safe_serialization=False, format="auto_round")
        
        from auto_round.auto_quantizer import AutoHfQuantizer
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree("./saved", ignore_errors=True)
        quant_config = model.config.quantization_config
        assert quant_config.to_quant_block_names is not None
        
        
        

if __name__ == "__main__":
    unittest.main()




