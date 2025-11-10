
export AR_LOG_LEVEL=TRACE
model="/storage/yiliu7/Qwen/Qwen3-30B-A3B-Base/"
model="/storage/yiliu7/Qwen/Qwen3-30B-A3B-Base/"
# model="/storage/yiliu7/deepseek-ai/DeepSeek-V2-Lite-Chat"
# model="/storage/yiliu7/Qwen/Qwen3-235B-A22B"
base_name=$(basename ${model})
scheme="MXFP4"
scheme="MXFP8"
qmodel_dir="quantized_models/"
mkdir -p ${qmodel_dir}
output_dir="${qmodel_dir}/${base_name}-${scheme}"
# python quantize.py --model $model --scheme $scheme --output_dir $output_dir --skip_attn --use_autoround_format
python quantize.py --model $model -t qwen_mxfp8 --use_autoround_format
# python quantize.py --model $model -t qwen_mxfp4 --use_autoround_format
# python quantize.py --model $model -t qwen_mxfp8 --use_autoround_format
# python quantize.py --model $model -t ds_mxfp8 --use_autoround_format
# model_name="/storage/yiliu7/Qwen/Qwen3-A3B-Base"

# scheme="MXFP4"

# output_path="./"
# base_name=$(basename ${model_name})
# CUDA_VISIBLE_DEVICES=$device \
# python3 quantize.py \
#      --model ${model} \
#      --scheme ${scheme} \
#      --format llm_compressor \
#      --iters 0 \
#      --enable_torch_compile \
#      --output_dir ${output_path}/${base_name}-${scheme}