import os
# os.environ['HF_HOME'] = '/dataset/huggingface'
# os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib/python3.10/dist-packages/habana_frameworks/torch/lib/libhabana_pytorch_backend.so'
# os.environ['LD_PRELOAD'] = '/usr/local/lib/python3.10/dist-packages/habana_frameworks/torch/lib/libhabana_pytorch_backend.so'
os.environ['_DNNL_DISABLE_COMPILER_BACKEND'] = '0'
# os.environ['LOGLEVEL'] = 'DEBUG'


cmd_list = \
    [
        "/usr/bin/python3 ./main.py --model_name /lyt/lyt_models/opt-125m --bits 4 --group_size 128 \
            --nsamples 64 --iters 10 --deployment_device fake,auto_round --disable_eval --seqlen 128 \
            --output_dir './output/opt125m2'",    
        # "/usr/bin/python3 eval_042/evaluation.py --model_name ./output/opt125m/opt-125m-autoround-w4g128-round \
        #     --eval_bs 16 --trust_remote_code --tasks lambada_openai",


    
    ]

for item in cmd_list:
    print(item)
    print(os.environ)
    os.system(item)
    print("\n\n\n\n\n\n\n\n\n")



