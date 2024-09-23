command to run autoround with mixbit, will automatic run eval.
```bash
python -m auto_round \
    --model model_name_or_path \
    --device 0 \
    --hybrid_json hybrid_json_path \
    --group_size 32 \
    --seqlen 2048 \
    --nsamples 512 \
    --iters 200 \
    --bits 4 \
    --act_bits 4 \
    --data_type "mx_fp" \
    --tasks piqa,winogrande,hellaswag,lambada_openai,mmlu \
    --format fake \
    --output_dir tmp_dir
```

command to eval saved model
```bash
python -m auto_round --model model_name_or_path \
    --tasks piqa,winogrande,hellaswag,lambada_openai,mmlu \
    --device 0 \
    --eval 
```