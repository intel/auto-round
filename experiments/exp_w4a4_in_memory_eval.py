import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--device-map", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--eval-bs", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gsm8k", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from auto_round import AutoRound
    from auto_round.eval.eval_cli import eval_task_by_task

    quantizer = AutoRound(
        args.model,
        scheme=args.scheme,
        algorithm="rtn",
        group_size=16,
        iters=0,
        disable_opt_rtn=True,
        nsamples=128,
        seqlen=2048,
        batch_size=8,
        device_map=args.device_map,
        seed=42,
        enable_deterministic_algorithms=True,
        format="fake",
        low_cpu_mem_usage=True,
    )
    model, _ = quantizer.quantize()
    model.eval()

    eval_task_by_task(
        model,
        tokenizer=quantizer.tokenizer,
        device=args.device_map,
        tasks=args.tasks,
        batch_size=args.eval_bs,
        limit=args.limit,
        eval_model_dtype="bfloat16",
        num_fewshot=5 if args.gsm8k else None,
        gen_kwargs={"max_gen_toks": 1024, "do_sample": False} if args.gsm8k else None,
    )


if __name__ == "__main__":
    main()