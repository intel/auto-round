## Model Details

This model is an int4 model with group_size 128 and symmetric quantization of [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) generated by [intel/auto-round](https://github.com/intel/auto-round).  Load the model with `revision="5a6d912"` to use AutoGPTQ format

## How To Use

### INT4 Inference(CPU/HPU/CUDA)

CPU requires auto-round version>0.3.1

```python
from auto_round import AutoRoundConfig ##must import for auto-round format
from transformers import AutoModelForCausalLM,AutoTokenizer
quantized_model_dir = "Intel/Qwen2.5-7B-Instruct-int4-inc"
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)

model = AutoModelForCausalLM.from_pretrained(
    quantized_model_dir,
    torch_dtype='auto',
    device_map="auto",
    ##revision="0b70f95" ##AutoGPTQ format
    ##revision="5a6d912" ##Quantized lm-head version
    
)

##import habana_frameworks.torch.core as htcore ## uncommnet it for HPU
##import habana_frameworks.torch.hpu as hthpu ## uncommnet it for HPU
##model = model.to(torch.bfloat16).to("hpu") ## uncommnet it for HPU

prompt = "There is a girl who likes adventure,"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=200,  ##change this to align with the official usage
    do_sample=False  ##change this to align with the official usage
)
generated_ids = [
output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

prompt = "There is a girl who likes adventure,"
##INT4:
"""That's great! It sounds like this girl has an exciting spirit. How can I help you explore her love for adventure? Are you looking for ideas for adventurous activities, planning a trip, or perhaps looking for ways to support her adventurous nature?
"""

##BF16:
"""That sounds exciting! What kind of adventures does she enjoy? Is there something specific you'd like to plan or discuss related to her love for adventure?
"""

prompt = "9.11和9.8哪个数字大"  
#INT4: 
"""在比较9.11和9.8时，我们从左到右逐位进行比较：

1. 首先比较整数部分：两个数的整数部分都是9，相等。
2. 接下来比较小数部分：
   - 9.11的小数部分是0.11
   - 9.8的小数部分是0.8

将0.11和0.8进行比较，显然0.8大于0.11。

因此，9.8比9.11大。
"""

##BF16: 
"""在比较9.11和9.8这两个数字时，我们可以直接进行比较：

- 9.11 是九点一一
- 9.8 是九点八

显然，9.8 比 9.11 大。这是因为9.8中的十分位是8，而9.11中的十分位是1，8大于1。

所以，9.8 > 9.11。"""


prompt = "Once upon a time,"
##INT4: 
"""Once upon a time, in a land filled with wonder and magic, there lived a young girl named Elara. She had bright eyes that sparkled like the stars on a clear night and hair as golden as the sun-kissed fields of wheat. Elara's home was a cozy cottage nestled at the edge of a vast, enchanted forest, where ancient trees whispered secrets to one another and mystical creatures roamed freely.

Every day, Elara would venture into the forest, exploring its hidden paths and marveling at the wonders it held. One sunny morning, as she wandered deeper into the woods than ever before, she stumbled upon a glade bathed in a soft, ethereal light. In the center of this glade stood an enormous tree, its trunk wider than any building Elara had ever seen, and its branches stretching high into the sky.

As she approached the tree, she noticed a small, shimmering door carved into its bark. Curious, Elara reached out and"""

##BF16:
"""Once upon a time, in a land filled with wonder and magic, there lived a young girl named Elara. She had bright eyes that sparkled like the stars on a clear night and hair as golden as the sun-kissed fields of wheat. Elara's home was a cozy cottage nestled at the edge of a vast, enchanted forest, where mythical creatures roamed freely and ancient trees whispered secrets of old.

Every day, Elara would venture into the forest to gather herbs and flowers for her grandmother, who was a skilled healer. The forest was her sanctuary, a place where she could escape the worries of the world and find solace in nature's embrace. As she walked through the dense underbrush, she often stumbled upon hidden glades and sparkling streams, each more enchanting than the last.

One sunny morning, as Elara was collecting wildflowers near a babbling brook, she heard a faint cry for help. Curious, she followed the sound to a clearing where"""


prompt = "请简短介绍一下阿里巴巴公司"
##INT4:
"""阿里巴巴集团是一家总部位于中国杭州的全球领先电子商务和科技公司，成立于1999年。它由马云创立，旗下拥有包括淘宝、天猫、阿里云等在内的多个知名业务。阿里巴巴致力于构建全球化的商业生态系统，为消费者、企业和政府提供创新的在线解决方案。同时，阿里巴巴也是中国最大的零售市场之一，并在云计算、数字娱乐、物流等领域具有重要影响力。
"""

##BF16:
"""阿里巴巴集团是一家总部位于中国杭州的全球领先电子商务和科技公司，成立于1999年。阿里巴巴旗下拥有淘宝、天猫、阿里云等知名业务，致力于构建全球化的商业生态系统。阿里巴巴愿景是让世界各地的企业都能够平等地进行贸易，并通过技术创新推动社会进步。
"""
```

### Evaluate the model

pip3 install lm-eval==0.4.5

```bash
auto-round --model "Intel/Qwen2.5-7B-Instruct-int4-inc" --eval --eval_bs 16  --tasks leaderboard_ifeval,leaderboard_mmlu_pro,gsm8k,lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,cmmlu,ceval-valid
```

| Metric                                     |  BF16  |  INT4 (5.3G)  | INT4 lm-head (4.5G) |
| :----------------------------------------- | :----: | :----: | ------------------- |
| Avg                                        | 0.6649 | 0.6586 | 0.6577              |
| leaderboard_mmlu_pro 5 shots               | 0.4458 | 0.4436 | 0.4384              |
| leaderboard_ifeval inst_level_strict_acc   | 0.6859 | 0.6715 | 0.6595              |
| leaderboard_ifeval prompt_level_strict_acc | 0.5730 | 0.5508 | 0.5379              |
| mmlu                                       | 0.7174 | 0.7147 | 0.7145              |
| cmmlu                                      | 0.8028 | 0.7888 | 0.7888              |
| ceval-valid                                | 0.7935 | 0.7838 | 0.7741              |
| gsm8k 5 shots                              | 0.7665 | 0.7544 | 0.8006              |
| lambada_openai                             | 0.6949 | 0.6878 | 0.6763              |
| hellaswag                                  | 0.6195 | 0.6139 | 0.6121              |
| winogrande                                 | 0.7119 | 0.7064 | 0.7135              |
| piqa                                       | 0.7938 | 0.7873 | 0.7845              |
| truthfulqa_mc1                             | 0.4786 | 0.4774 | 0.4810              |
| openbookqa                                 | 0.3480 | 0.3580 | 0.3540              |
| boolq                                      | 0.8636 | 0.8602 | 0.8609              |
| arc_easy                                   | 0.8131 | 0.8068 | 0.8081              |
| arc_challenge                              | 0.5282 | 0.5316 | 0.5188              |



### Generate the model

Here is the sample command to generate the model. 

```bash
auto-round \
--model  Qwen/Qwen2.5-7B-Instruct \
--device 0 \
--group_size 128 \
--nsamples 512 \
--bits 4 \
--iter 1000 \
--disable_eval \
--model_dtype "fp16" \
--format 'auto_gptq,auto_round' \
--output_dir "./tmp_autoround" 
```

## Ethical Considerations and Limitations

The model can produce factually incorrect output, and should not be relied on to produce factually accurate information. Because of the limitations of the pretrained model and the finetuning datasets, it is possible that this model could generate lewd, biased or otherwise offensive outputs.

Therefore, before deploying any applications of the model, developers should perform safety testing.

## Caveats and Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

Here are a couple of useful links to learn more about Intel's AI software:

- Intel Neural Compressor [link](https://github.com/intel/neural-compressor)

## Disclaimer

The license on this model does not constitute legal advice. We are not responsible for the actions of third parties who use this model. Please consult an attorney before using this model for commercial purposes.

## Cite

@article{cheng2023optimize, title={Optimize weight rounding via signed gradient descent for the quantization of llms}, author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao and Liu, Yi}, journal={arXiv preprint arXiv:2309.05516}, year={2023} }

[arxiv](https://arxiv.org/abs/2309.05516) [github](https://github.com/intel/auto-round)