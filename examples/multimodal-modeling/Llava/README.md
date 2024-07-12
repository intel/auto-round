Step-by-Step
============

This document presents step-by-step instructions for auto-round.
# Run Quantization on Multimodal Models

In this example, we introduce an straight-forward way to execute quantization on some popular multimodal models such as LLaVA. 

## Install
If you are not using Linux, do NOT proceed, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Clone this repository and navigate to LLaVA folder
```shell
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install Package
```
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Download the calibration data

Our calibration process resembles the official visual instruction tuning process. To align the official implementation of [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#visual-instruction-tuning)

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip), and unzip the image folder to any directory you desire.

<br />

## 2. Run Examples
Enter into the examples folder and install requirements

```bash
pip install -r requirements.txt
```

- **Default Settings:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name liuhaotian/llava-v1.5-7b  --bits 4 --group_size 128
```

- **Speedup the tuning:**

reduce the seqlen to 512(potential large accuracy drop)

or combine them

- **Enable quantized lm-head:**

Currently only support in Intel xpu and AutoRound format,however, we found the fake tuning could improve the accuracy is some scenarios. low_gpu_mem_usage=False is strongly recommended if the whole model could be loaded to the device, otherwise it will be quite slow to cache the inputs of lm-head. Another way is reducing nsamples,e.g. 128, to alleviate the issue.
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name liuhaotian/llava-v1.5-7b  --bits 4 --group_size 128 --quant_lm_head
```

- **Utilizing the AdamW Optimizer:**

Include the flag `--adam`. Note that AdamW is less effective than sign gradient descent in many scenarios we tested.

- **Running on Intel Gaudi2**
```bash
bash run_autoround_on_gaudi.sh
```

## 4. Results
Using [COCO 2017](https://cocodataset.org/) and [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) datasets for quantization calibration, and TextVQA dataset for evaluation. When the vision components are not involved in quantization, it is able to achieve accuracy loss within 1%. The results for LLava-7b are as follows:
| Model | Config | Precision | Hyperparameter | Accuracy% | Relative drop |
|  :----: | :----: | :----: | :----: | :----: | :----: |
| liuhaotian/llava-v1.5-7b | - | FP16 | - | 58.21 | - |
| liuhaotian/llava-v1.5-7b | W4G128 | FP16 | with vision | 56.39 | -3.13% |
| liuhaotian/llava-v1.5-7b | W4G128 | FP16 | w/o vision | 58.08 | -0.22% |


## 5. Known Issues
* huggingface format model is not support yet, e.g. llava-1.5-7b-hf
* Setting seqlen to 2048 is not working yet.


## 6. Environment

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








