#!/bin/bash
# Evaluate all quantized models with lm-eval
PYTHON=/home/yiliu7/workspace/ar-local/bin/python
TASKS="lambada_openai,piqa,mmlu"
LOG_DIR=/storage/yiliu7/act_cp_verify

for dir in "$LOG_DIR"/*/; do
    name=$(basename "$dir")
    echo "========== Evaluating: $name =========="
    CUDA_VISIBLE_DEVICES=0 $PYTHON -m auto_round eval \
        --model "$dir" \
        --tasks "$TASKS" \
        --eval_bs auto \
        2>&1 | tee "$LOG_DIR/${name}_eval.log"
done
