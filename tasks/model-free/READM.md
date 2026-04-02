Task: run CUDA_VISIBLE_DEVICES=0,1,2,3 python torch_gen.py
notes: This model is using the static quant, we may need to patch the FP8 quantizer(FineGrainedFP8) to support static quant.
Please do out of tree patch as much as possible, not change the source code.

## Reproduce

```bash
# Activate the virtual environment
source /home/yiliu7/workspace/venvs/ar/bin/activate

# Run the script
cd /home/yiliu7/workspace/auto-round
CUDA_VISIBLE_DEVICES=0,1,2,3 python tasks/model-free/torch_gen.py
```

## Environment

- Python: 3.13.12 (`/home/yiliu7/workspace/venvs/ar/bin/python`)
- transformers: 4.52.0
- torch: 2.11.0+cu130
- auto_round: 0.13.0
- GPU: 4x NVIDIA GeForce RTX 5090 D (compute capability 12.0, driver 580.142)