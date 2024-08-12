Step-by-Step
============
transformers>=4.41.0
This document presents step-by-step instructions for auto-round.
# Run Quantization on Phi-3-vision Models

In this example, we introduce an straight-forward way to execute quantization on some popular multimodal models such as Phi-3-vision. 

## Download the calibration data

Our calibration process resembles the official visual instruction tuning process.

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip), and unzip the image folder to any directory you desire.


## 2. Run Examples
Enter into the examples folder and install lm-eval to run the evaluation
```bash
pip install -r requirements.txt
```

- **Default Settings:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name microsoft/Phi-3-vision-128k-instruct  --bits 4 --group_size 128
```

- **Speedup the tuning:**

disable_low_gpu_mem_usage(more gpu memory)

reduce the seqlen to 512(potential large accuracy drop)

or combine them

- **Enable quantized lm-head:**

Currently only support in Intel xpu and AutoRound format, however, we found the fake tuning could improve the accuracy is some scenarios. Disable --low_gpu_mem_usage is strongly recommended if the whole model could be loaded to the device, otherwise it will be quite slow to cache the inputs of lm-head. Another way is reducing nsamples,e.g. 128, to alleviate the issue.
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name microsoft/Phi-3-vision-128k-instruct  --bits 4 --group_size 128 --quant_lm_head
```

- **Utilizing the AdamW Optimizer:**

Include the flag `--adam`. Note that AdamW is less effective than sign gradient descent in many scenarios we tested.

- **Running on Intel Gaudi2**
```bash
bash run_autoround_on_gaudi.sh
```


## 3. Environment

PyTorch 1.8 or higher version is needed


## Reference
If you find SignRound useful for your research, please cite our paper:
```bash
@article{cheng2023optimize,
  title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```






