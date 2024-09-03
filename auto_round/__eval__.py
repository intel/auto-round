import argparse

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", help="Path of the model to evaluate, oly support auto_round format."
    )

    parser.add_argument("--device", default="auto", type=str,
                        help="The device to be used for tuning. The default is set to auto/None,"
                        "allowing for automatic detection. Currently, device settings support CPU, GPU, and HPU.")

    parser.add_argument("--tasks",
                        default="lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1," \
                                "truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge",
                        help="lm-eval tasks for lm_eval")
    
    parser.add_argument("--eval_bs", default=4, type=int,
                        help="eval batch size")
    
    parser.add_argument("--disable_trust_remote_code", action='store_true',
                        help="Whether to disable trust_remote_code")

    args = parser.parse_args()
    from auto_round.eval.evaluation import simple_evaluate
    model_args = model_args + f",trust_remote_code={not args.disable_trust_remote_code}"
    if isinstance(args.tasks, str):
        tasks = args.tasks.split(',')
    res = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        device=args.device,
        batch_size=args.eval_bs)

    from lm_eval.utils import make_table  # pylint: disable=E0401

    print(make_table(res))


if __name__ == '__main__':
    eval()