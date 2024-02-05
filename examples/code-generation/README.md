Step-by-Step
============

This document presents step-by-step instructions for auto-round.


## 2. Prepare Dataset

The mbpp in huggingface is adopted as the default calibration data and  will be downloaded automatically from the datasets Hub. To customize a dataset, please kindly follow our dataset code.
See more about loading [huggingface dataset](https://huggingface.co/docs/datasets/loading_datasets.html)

<br />

## 3. Run Examples
Enter into the examples folder
- **Default Settings:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name Salesforce/codegen25-7b-multi --amp --bits 4 --group_size -1 --enable_minmax_tuning --use_quant_input
```
- **Reduced GPU Memory Usage and Adjusted Training Batch Size:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name Salesforce/codegen25-7b-multi --amp --bits 4 --group_size -1 --low_gpu_mem_usage --train_bs 1 --gradient_accumulate_steps 8
```
- **Utilizing the AdamW Optimizer:**
Include the flag `--adam`. Note that AdamW is less effective than Sign gradient descent in many scenarios we tested.

- **Running the Original SignRound:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name Salesforce/codegen25-7b-multi --amp --bits 4 --group_size -1 --iters 400 --lr 0.0025 --minmax_lr 0.0025
```
 `--enable_minmax_tuning` is strongly recommended 


## 4. Evaluation
Please follow https://github.com/bigcode-project/bigcode-evaluation-harness to eval the model, currently we only support fake model evaluation.
<table border="1">
  <tr>
    <th>Model</th>
    <th>Method</th>
    <th>HumanEval top1 t=0.2 n_samples=20</th>
  </tr>
  <tr>
    <th rowspan="2">Salesforce/codegen25-7b-multi</th>
    <th>FP16 </th>
    <th>0.2854</th>
  </tr>
  <tr>
    <th>AutoRound use_quant_input=False</th>
    <th>0.2841</th>
  </tr>
</table>



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

