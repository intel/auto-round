import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
from dataloader import Dataloader
from auto_round import AutoRoundMLLM, AutoRound

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16

model_id = "openai/whisper-large-v3"
# model_id = "openai/whisper-large-v3-turbo"
bits, group_size, sym, act_bits = 8, -1, True, 8


model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
dataloader = Dataloader(processor, model, n_samples=20)




## quantize the model
autoround = AutoRoundMLLM(model, tokenizer, processor,
                        bits=bits, group_size=group_size, sym=sym, act_bits=act_bits,
                        iters=1,
                        dataset=dataloader,
                        batch_size=dataloader.batch_size,
                        nsamples=dataloader.n_samples,
                        layer_config={
                            "proj_out": {
                                "bits": bits, 
                                "group_size": group_size, 
                                "sym": sym, 
                                "act_bits": act_bits,
                            }
                        },
                    )
autoround.quantize()
print(autoround.model)
# breakpoint()

# save the quantized model, set format='auto_gptq' or 'auto_awq' to use other formats
output_dir = "./atrd_whisper_large_v3"
autoround.save_quantized(output_dir, format='llmcompressor', inplace=True)
print(autoround.model)
