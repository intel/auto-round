#!/bin/bash
set -x
device=0

CUDA_VISIBLE_DEVICES=$device \
python3 main.py \
--model_name=liuhaotian/llava-v1.5-7b \
--bits 4 \
--group_size 128 \
--iters 200 \
--deployment_device 'autoround' \
--image_folder /path/to/coco/images/train2017/ \
--question_file=self_made.json \
--eval-path=/path/to/textvqa_data/ \
--output_dir "./tmp_autoround"