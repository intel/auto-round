import os

os.environ['HF_HOME'] = '/models/huggingface'
os.environ['http_proxy'] = 'http://child-jf.intel.com:912'
os.environ['https_proxy'] = 'http://child-jf.intel.com:912'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYDEVD_UNBLOCK_THREADS_TIMEOUT"] = "30"
import torch
import time
data = torch.randn((4096,4096)).to(torch.bfloat16)
data = data.reshape(-1,128)

for i in range(1000):
    res = torch.max(data, dim=-1)
iters = 10000
torch.cuda.synchronize()
start_time = time.time()
for i in range(iters):
    torch.max(data, dim=0)

torch.cuda.synchronize()
end_time = time.time()
print((end_time - start_time)/iters)
exit()
# import random
# data = [0.6, 0.3,-5.6,1.2,-0.8,1.0,0.2,-1.9,1.9,0.7,-6.4,-2.3,-0.2,0.3,0.5,-0.9,2.4,1.3,0.1,-0.6]
#
# wmax=max(data)
# wmin = min(data)
# maxq = 15
# scale = ((wmax - wmin) / maxq)
# zp = -wmin / scale
# int_w = [d / scale +zp+random.random()  for d in data]
# int_w = [min(max(d,0),maxq) for d in int_w]
# res = [scale*(d-zp) for d in int_w]
#
# tmp = 1

# import torch
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
# from transformers import set_seed
#
# import json
# from torch.nn.functional import pad
# import re
# from collections import Counter
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.cluster import KMeans
# import numpy as np

# import torch
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
# from transformers import set_seed
#
# import json
# from torch.nn.functional import pad
# import re
# from collections import Counter
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.cluster import KMeans
# import numpy as np

# model_name = "/models/Qwen2-7B"
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
# # model_name = "/home/lyt/lyt_models/Baichuan2-7B-Chat"
# embedding_model = model.base_model.embed_tokens
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# file_path = '/home/lyt/ChineseLLM_quant/auto-round/examples/language-modeling/content_Qwen2-7B.json'
# with open(file_path, 'r') as file:
#     generated_content = json.load(file)
#
#
# def most_common(lst):
#     return max(set(lst), key=lst.count)
#
#
# embed_feature = {}
# embedding_model = embedding_model.cuda()
# content_feature = []
# for key in generated_content:
#     content = generated_content[key]
#     input = tokenizer(content, return_tensors='pt')
#     input_list = input["input_ids"][0].tolist()
#     most_token = most_common(input_list)
#     if input_list.count(most_token) >= len(input_list) // 3:
#         continue
#
#     # input_cuda = {}
#     # for key in input:
#     #     input_cuda[key] =input[key].cuda()
#     res = embedding_model(input["input_ids"][:, :32].cuda())
#     res = res.view(-1, res.shape[-1])
#     emb_mean = torch.mean(res, dim=0)
#     content_feature.append([content, emb_mean])
#
# # content_feature = sorted(content_feature, key=lambda x: x[1])
# kmeans = KMeans(n_clusters=512, random_state=0)
# scores = np.array([x[1].detach().cpu() for x in content_feature])
# clusters = kmeans.fit_predict(scores)
# cluster_doc = [[] for i in range(512)]
# for i, c in enumerate(clusters):
#     cluster_doc[c].append(content_feature[i][0])
#
#
# choose_doc = [docs[0] for docs in cluster_doc]
# choose_doc_content = {f"content{i}": doc for i, doc in enumerate(choose_doc)}
#
#     # with open('contentYi_clusterFull.json', 'w', encoding="utf-8") as file:
#     #     json.dump(cluster_doc_json, file, ensure_ascii=False, indent=4)
#
# with open('result.json', 'w', encoding="utf-8") as file:
#     json.dump(choose_doc_content, file, ensure_ascii=False, indent=4)
#
# # with open('contentYi_clusterCat.json', 'w', encoding="utf-8") as file:
# #     json.dump(cluster_doc_cat_json, file, ensure_ascii=False, indent=4)
#
# print("saved and finished")
# exit()

import sys
# sys.path.insert(0, '/home/wenhuach/transformers/src')
# from transformers.quantizers.auto import AutoRoundQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import sys
from auto_round.auto_quantizer import AutoHfQuantizer

sys.path.insert(0, './')

from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

quantized_model_dir = "/dataset/falcon-7b-int4-inc"
quantized_model_dir = "/data5/wenhuach/llama3-test/llama3-autoround-w4g128-gpu"
quantized_model_dir = "/data5/wenhuach/tmp-llama3-quant-lm-head/llama3-autoround-w4g128-gpu"
# quantized_model_dir = "/models/gemma-2b"
quantized_model_dir = "/data5/qdq_llama3/llama3_True-0.01-200/"
quantized_model_dir = "/data5/llama3_8b_instruct-chat"
quantized_model_dir = "/home/wenhuach/auto-round/examples/language-modeling/tmp_autoround/opt-125m-autoround-w8g128-gpu"
quantized_model_dir = "/data5/wenhuach/Llama3-8b-hf-iter1000-with-lm-head-use-quant-input-w4g128/llama3_8b_instruct-chat-autoround-w4g128-gpu"
quantized_model_dir = "/data2/lyt/ChineseLLM_quant/output/Yi/tmp_Yi5/Yi-6B-Chat-autoround-w4g128-gpu"
quantized_model_dir = "/data5/wenhuach/test/opt-125m-autoround-w4g128-gpu"
quantized_model_dir = "/data5/wenhuach/test/Meta-Llama-3-8B-Instruct-autoround-w4g128-gpu"
quantized_model_dir = "/home/wenhuach/auto-round/examples/language-modeling/tmp_autoround/opt-125m-autoround-w4g128-round"
quantized_model_dir = "/home/wenhuach/auto-round/examples/language-modeling/tmp_autoround/opt-125m-autoround-w4g128-gpu"
quantized_model_dir = "/data5/wenhuach/marlin/opt-125m-autoround-w4g128-round"
# quantized_model_dir = "/data5/wenhuach/test/Qwen2-7B-autoround-w4g128-gpu"
quantized_model_dir = "/data5/wenhuach/marlin/Meta-Llama-3-8B-Instruct-autoround-w4g128-round"
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)
model = AutoModelForCausalLM.from_pretrained(quantized_model_dir,
                                             device_map="auto"
                                             )
# model = model.to(torch.float16)
# model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_safetensors=True)
# model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_safetensors=True, use_marlin=True)
# tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)


# text = "Help complete the following passage: \n It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I have not seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around. If pasta from other team are around, though, they get locked down in their attempt, unable to conquer the plate, until one of them die (think Battlefield's standard 'Conquest' mode). You get points every second for every plate you own.,"
# text = "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun."
text = "There is a girl who likes adventure,"
##based on the following passage:
# conversation = [ {'role': 'user', 'content': {text}} ]
# prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
exit()

model_name = "/data5/wenhuach/Llama3-8b-hf-iter1000-with-lm-head-use-quant-input-w4g128/llama3_8b_instruct-chat-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/llama3-test/llama3_8b_instruct-chat-autoround-w4g128-gpu"
model_name = "/data4/wenhuach/Mistral-7B-v0.1_iter200_lm_head/Mistral-7B-v0.1-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/gemma-2b-iter400-fp16/gemma-2b-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/phi2-iter1000-nolmhead-disable_no_quanted_input/phi-2-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/falcon-7b-iter1000-w4g64-disable_quanted_input/falcon-7b-autoround-w4g64-gpu"
model_name = "/models/falcon-7b"
model_name = "/data5/wenhuach/opt125m-test/Mistral-7B-v0.1-autoround-w2g128-gpu"
model_name = "/home/wenhuach/auto-round/examples/language-modeling/tmp_autoround/Mistral-7B-v0.1-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/opt125m-test/llama3_8b_instruct-chat-autoround-w4g128-gpu"
model_name = "/home/wenhuach/auto-round/examples/language-modeling/tmp_autoround/llama3_8b_instruct-chat-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/llama3-iter2-w2g32-lmhead-w4g32-asym-autoround/Meta-Llama-3-8B-Instruct-autoround-w2g32-gpu"
model_name = "/data5/wenhuach/llama3-iter1000-lmhead-w4g32-asym-autoround/Meta-Llama-3-8B-Instruct-autoround-w2g32-gpu"
model_name = "/data5/wenhuach/llama3-iter200-w2g32-lmhead-w4g32-asym-autoround/Meta-Llama-3-8B-Instruct-autoround-w2g32-gpu"
model_name = "/data5/wenhuach/llama3-iter200-w2g32-lmhead-w4g32-asym-autoround-disable-quanted-input/Meta-Llama-3-8B-Instruct-autoround-w2g32-gpu"
model_name = "/data5/zww/test_qwen/Qwen2-1.5B_1e-3_quanted_input/Qwen2-1.5B-autoround-w4g32-gpu"
model_name = "/data5/wenhuach/Qwen7B-sym-iter200/Qwen2-7B-autoround-w4g32-gpu"
model_name = "/data2/lyt/ChineseLLM_quant/output/Qwen2/7B/tmp4/Qwen2-7B-autoround-w4g32-gpu"
model_name = "/data5/zww/test_qwen/Qwen2-1.5B_1e-3_quanted_input/Qwen2-1.5B-autoround-w4g32-gpu"
model_name = "/data2/lyt/ChineseLLM_quant/output/Qwen2/0.5B/tmp1/Qwen2-0.5B-Instruct-autoround-w4g32-gpu"
model_name = "/models/Qwen2-0.5B-Instruct"
model_name = "/data5/wenhuach/Qwen2-0.5B-instruct-sym-iter1000-minmaxlr-2e-3/Qwen2-0.5B-Instruct-autoround-w4g32-gpu"
model_name = "/data5/zww/test_qwen/Qwen2-1.5B-Instruct_1e-3_quanted_input/Qwen2-1.5B-Instruct-autoround-w4g32-gpu"
model_name = "/data5/wenhuach/Qwen0.5B-instruct-sym-iter1000-minmaxlr-2e-3--trainbs16-nsamples-1024/Qwen2-0.5B-Instruct-autoround-w4g32-gpu"
model_name = "/data5/wenhuach/Qwen0.5B-instruct-sym-iter1000-minmaxlr-2e-3--mixed/Qwen2-0.5B-Instruct-autoround-w4g32-gpu"
model_name = "/data5/wenhuach/Qwen2-7B-instruct-w4g128-sym/Qwen2-7B-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/Qwen2-0.5B-instruct-W4G128asym-iter1000-seqlen4096-ar/Qwen2-0.5B-Instruct-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/Qwen2-7B-instruct-w4g128-asym/Qwen2-7B-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/Qwen2-1.5B-instruct-W4G128asym-iter1000-seqlen4096-ar/Qwen2-1.5B-Instruct-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/Qwen2-0.5B-instruct-W4G128asym-iter1000-seqlen4096/Qwen2-0.5B-Instruct-autoround-w4g128-gpu"  ##prompt不行
model_name = "/data5/wenhuach/Qwen2-0.5B-instruct-W4G128asym-iter1000-autoround/Qwen2-0.5B-Instruct-autoround-w4g128-gpu"  ##prompt可以
model_name = "/data5/wenhuach/Qwen2-7B-W4G128asym-iter200-quantlmheadg128-ar/Qwen2-7B-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/Qwen2-7B-int4-inc-lmhead"
model_name = "/data5/wenhuach/Qwen2-7B-W4G128asym-iter1000-ar-float16/Qwen2-7B-autoround-w4g128-gpu"
# model_name = "/data5/wenhuach/Qwen2-7B-W4G128asym-iter1000-float16/Qwen2-7B-autoround-w4g128-gpu"
# model_name = "/models/Qwen2-7B"
# model_name = "/data5/wenhuach/Qwen1.5-sym-iter200/Qwen2-1.5B-autoround-w4g32-gpu"
# model_name = "/models/Mistral-7B-v0.1"
# model_name = "/models/Qwen1.5-7B-Chat"
# model_name = "/data5/wenhuach/Mistral-7B-v0.1-int4-inc"
# model = AutoRoundModelForCausalLM.from_pretrained(model_name, device_map="auto")
model_name = "/models/Qwen2-1.5B-Instruct"
model_name = "/data5/wenhuach/Qwen2-7B-Instruct-iter200/Qwen2-7B-Instruct-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/Qwen2-7B-Instruct-iter200-lmhead/Qwen2-7B-Instruct-autoround-w4g128-gpu"
model_name = "/data5/wenhuach/Qwen2-7B-int4-inc-lmhead-private"
model_name = "/data5/wenhuach/test_iter1000_lr_5e-3/Qwen2-7B-autoround-w4g32-gpu"
model_name = "/data5/wenhuach/marlin/Meta-Llama-3-8B-Instruct-autoround-w4g128-round"
# model_name = "/data5/wenhuach/opt-125m-awq"
# model_name = "Intel/Mistral-7B-v0.1-int4-inc-lmhead"
# model_name = "/data5/wenhuach/Mistral-7B-v0.1-int4-inc-lmhead"
import torch

prompt = "下面我来介绍一下阿里巴巴公司，"
# prompt = "88+99等于多少?"
prompt = "Once upon a time,"
# prompt = "There is a girl who likes adventure,"
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# device = "cuda:0"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map=device
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(device)
#
# generated_ids = model.generate(
#     model_inputs.input_ids,
#     max_new_tokens=50,
#     do_sample=True
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]
#
#
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)
# exit()
# from transformers import AutoRoundConfig
# quantization_config = AutoRoundConfig(
#    backend="cpu"
# )

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# text = "下面我来介绍一下阿里巴巴公司，"
# text = "88+99等于多少?"
text = "Once upon a time,"
# text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
exit()

# model_name = "/home/lyt/lyt_models/Baichuan2-7B-Chat"
embedding_model = model.base_model.embed_tokens
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
file_path = 'content_filter.json'
with open(file_path, 'r') as file:
    generated_content = json.load(file)

embed_feature = {}
embedding_model = embedding_model.cuda()
content_feature = []
for key in generated_content:
    content = generated_content[key]
    input = tokenizer(content, return_tensors='pt')
    # input_cuda = {}
    # for key in input:
    #     input_cuda[key] =input[key].cuda()
    res = embedding_model(input["input_ids"][:, :32].cuda())
    res = res.view(-1, res.shape[-1])
    emb_mean = torch.mean(res, dim=0)
    content_feature.append([content, emb_mean])

# content_feature = sorted(content_feature, key=lambda x: x[1])
kmeans = KMeans(n_clusters=512, random_state=0)
scores = np.array([x[1].detach().cpu() for x in content_feature])
clusters = kmeans.fit_predict(scores)
cluster_doc = [[] for i in range(512)]
for i, c in enumerate(clusters):
    cluster_doc[c].append(content_feature[i][0])
tmp = 1

# 打印聚类结果
# for doc, cluster in zip(documents, clusters):
#     print(f"Document: {doc} - Cluster: {cluster}")
# tmp = 1


##save content
# file_path = 'content.json'
# with open(file_path, 'r') as file:
#     generated_content = json.load(file)
# content_new, seen_content = {}, set()
# for key in generated_content:
#     content = generated_content[key].strip()
#     content = re.sub(r'[ \n]{2,}', ' ', content) #去除连续的空格、换行
#     if not content:
#         continue
#     # pattern_cn = r'(\b[\u4e00-\u9fa5]+[，。])\1+' #中文去重
#     # re.sub(pattern_cn, r'\1', content)
#     # pattern = r'(?i)(\b[A-Za-z\s]+[.!?])\s+\1+' #英文去重
#     # re.sub(pattern, r'\1', content)
#     input = tokenizer(content, return_tensors='pt')
#     input_ids, attention_mask = input['input_ids'], input['attention_mask'].tolist()[0]
#     L = input_ids.size()[1]
#     count_info = Counter(content)
#     most_common_char, count = count_info.most_common(1)[0]
#     if L < 32 or count >= len(content) // 2:
#         continue
#     if content in seen_content:
#         print(f'content seen: {key}, {input_ids.shape}')
#         continue
#     seen_content.add(content)
#     # # pad_len = 2048 - input_ids.shape[1]
#     # # input_ids = pad(input_ids, (0, pad_len), value=1).tolist()[0]
#     # input_ids = input_ids.tolist()[0]
#     content_new[key] = content
#     print(f"dataset len: {key} {L}")
#
# file_name = 'content_filter.json'
# with open(file_name, 'w', encoding="utf-8") as file:
#     json.dump(content_new, file, ensure_ascii=False, indent=4)
# print('saved  ', len(content_new))


# # from llava.model.builder import load_pretrained_model
# # from llava.mm_utils import get_model_name_from_path
# # from llava.eval.run_llava import eval_model
# #
# # model_path = "/models/llava-v1.5-7b"
# #
# # tokenizer, model, image_processor, context_len = load_pretrained_model(
# #     model_path=model_path,
# #     model_base=None,
# #     model_name=get_model_name_from_path(model_path)
# # )
# #
# #
# # prompt = "What are the things I should be cautious about when I visit here?"
# # image_file = "https://llava-vl.github.io/static/images/view.jpg"
# #
# # args = type('Args', (), {
# #     "model_path": model_path,
# #     "model_base": None,
# #     "model_name": get_model_name_from_path(model_path),
# #     "query": prompt,
# #     "conv_mode": None,
# #     "image_file": image_file,
# #     "sep": ",",
# #     "temperature": 0,
# #     "top_p": None,
# #     "num_beams": 1,
# #     "max_new_tokens": 512
# # })()
# #
# # eval_model(args)
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# quantized_model_dir = "/models/Qwen1.5-7B-Chat"
# tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, padding_side='left')
#
# model = AutoModelForCausalLM.from_pretrained(quantized_model_dir,
#                                              device_map="auto"
#                                              )
# tmp = 1
# json_content = {}
# for i in range(10240):
#     key = "content" + str(i)
#     output = model.generate(input_ids=None, max_new_tokens=2560, do_sample=True)
#     res = tokenizer.decode(output[0], skip_special_tokens=True)
#     json_content[key] = res
#     print(res, flush=True)
#     import json
#
#     if i % 100 == 0 and i > 0:
#         with open("content.json", "w+", encoding="utf-8") as f:
#             json.dump(json_content, f, ensure_ascii=False, indent=4)
#
#     # prompt = "请优化以下内容:" + res
#     # messages = [
#     #     {"role": "system", "content": "You are a helpful assistant."},
#     #     {"role": "user", "content": prompt}
#     # ]
#     # text = tokenizer.apply_chat_template(
#     #     messages,
#     #     tokenize=False,
#     #     add_generation_prompt=True
#     # )
#     # model_inputs = tokenizer([text], return_tensors="pt").to("cuda:0")
#     #
#     # generated_ids = model.generate(
#     #     model_inputs.input_ids,
#     #     max_new_tokens=2048,
#     #     min_new_tokens=2048
#     # )
#     # generated_ids = [
#     #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     # ]
#     #
#     # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     #
#     # print(response)
#
