import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "/mengni/Llama-4-Maverick-17B-128E-Instruct"
model_name = "/models/Llama-4-Maverick-17B-128E-Instruct"
# model_name = "/nvme1n1/Llama-4-Scout/"
from auto_round.utils import mllm_load_model

model, processor, tokenizer, image_processor = mllm_load_model(
    model_name,
)

device_map = {}

moe = list(range(1, 48, 2))

# for layer in moe:
#     for idx in list(range(0,32)):
#         device_map[f"language_model.model.layers.{layer}.feed_forward.experts.{idx}.gate_proj"] = "cuda:1"
#         device_map[f"language_model.model.layers.{layer}.feed_forward.experts.{idx}.up_proj"] = "cuda:1"
#         device_map[f"language_model.model.layers.{layer}.feed_forward.experts.{idx}.down_proj"] = "cuda:1"

#     for idx in list(range(32,64)):
#         device_map[f"language_model.model.layers.{layer}.feed_forward.experts.{idx}.gate_proj"] = "cuda:2"
#         device_map[f"language_model.model.layers.{layer}.feed_forward.experts.{idx}.up_proj"] = "cuda:2"
#         device_map[f"language_model.model.layers.{layer}.feed_forward.experts.{idx}.down_proj"] = "cuda:2"

#     for idx in list(range(64,96)):
#         device_map[f"language_model.model.layers.{layer}.feed_forward.experts.{idx}.gate_proj"] = "cuda:3"
#         device_map[f"language_model.model.layers.{layer}.feed_forward.experts.{idx}.up_proj"] = "cuda:3"
#         device_map[f"language_model.model.layers.{layer}.feed_forward.experts.{idx}.down_proj"] = "cuda:3"

#     for idx in list(range(96,128)):
#         device_map[f"language_model.model.layers.{layer}.feed_forward.experts.{idx}.gate_proj"] = "cuda:4"
#         device_map[f"language_model.model.layers.{layer}.feed_forward.experts.{idx}.up_proj"] = "cuda:4"
#         device_map[f"language_model.model.layers.{layer}.feed_forward.experts.{idx}.down_proj"] = "cuda:4"

layer_config = {}
fp_layers = [
    "router",
    "shared_expert",
    "feed_forward.down_proj",
    "feed_forward.gate_proj",
    "feed_forward.up_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "v_proj",
    "lm_head",
    "vision_mdoel",
]
for n, m in model.named_modules():
    if not isinstance(m, (torch.nn.Linear)):
        continue
    for name in fp_layers:
        if name in n:
            layer_config[n] = {"bits": 16, "act_bits": 16}
            break

from auto_round import AutoRoundMLLM

scheme = "FP8_STATIC"
output_dir = f"/data2/yiliu4/{model_name.split('/')[-1]}-{scheme}"
autoround = AutoRoundMLLM(
    model=model,
    tokenizer=tokenizer,
    # device_map=device_map,
    # nsamples=512,
    # batch_size=1,
    # low_gpu_mem_usage=True,
    # seqlen=512,
    # iters=10,
    # data_type="nv_fp4",
    # group_size=16,
    # bits=4,
    # gradient_accumulate_steps=4,
    # act_bits=4,
    # act_data_type="nv_fp4_with_static_gs",
    scheme=scheme,
    layer_config=layer_config,
    processor=processor,
    image_processor=image_processor,
    iters=0,
)

autoround.quantize_and_save(format="auto_round", output_dir=output_dir)
