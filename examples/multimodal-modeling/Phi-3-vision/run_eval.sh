export https_proxy=http://proxy.ims.intel.com:911
export http_proxy=http://proxy.ims.intel.com:911
export HF_HOME=/home/weiweiz1/.cache/

#  Mistral-7B-Instruct-v0.2
# device=3
# Baichuan2-7B-Chat  Phi-3-mini-4k-instruct
# Llama-2-7b-chat-hf 
# lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu,
# ceval-valid,cmmlu
# dir=/data5/zww/test_faster/
# dir=/models
# for model in Phi-3-mini-4k-instruct Meta-Llama-3-8B-Instruct
# do
#     echo ${model}/default
#     CUDA_VISIBLE_DEVICES=$device \
#     python3 eval_042/evaluation.py --model_name ${dir}${model}_default/$model-autoround-w4g128-gpu \
#     --trust_remote_code \
#     --eval_bs 16 --tasks gsm8k,ceval-valid,cmmlu \
#     2>&1| tee -a /data4/zww/test_faster/rounding_${model}_rtn.txt
#     echo ${model}/rtn
# done&

device=2
dir=/data4/zww/tmp/
# dir=/data5/models/
for model in Phi-3-vision-128k-instruct
do
    echo ${model}
    CUDA_VISIBLE_DEVICES=$device \
    python3 eval_042/evaluation.py --model_name ${dir}/$model-autoround-w4g128-round \
    --trust_remote_code \
    --eval_bs 16 --tasks lambada_openai \
    2>&1| tee -a /data4/zww/test_faster/rounding_${model}.txt
    echo ${model}
done
# dir=/data5/zww/test_faster/
# for model in Phi-3-mini-4k-instruct Mistral-7B-Instruct-v0.2
# do
#     echo ${model}/rtn
#     CUDA_VISIBLE_DEVICES=$device \
#     python3 eval_042/evaluation.py --model_name ${dir}${model}_rtn/$model-autoround-w4g128-gpu \
#     --trust_remote_code \
#     --eval_bs 16 --tasks lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu,gsm8k \
#     2>&1| tee -a /data4/zww/test_faster/rounding_${model}_rtn.txt
#     echo ${model}/rtn
# done

