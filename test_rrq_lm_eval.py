#!/usr/bin/env python
"""Run lm-eval accuracy benchmarks on RRQ (base+residual) model at different bit-widths.

Usage:
    python test_rrq_lm_eval.py                          # all tasks, all bit-widths
    python test_rrq_lm_eval.py --tasks hellaswarc       # HellaSwag + ARC only
    python test_rrq_lm_eval.py --bits 4 8              # only 4-bit and 8-bit
    python test_rrq_lm_eval.py --limit 200             # fast: 200 examples per task
    python test_rrq_lm_eval.py --base-dir ./rrq_output/base --residual-dir ./rrq_output/residual

Requires: pip install lm-eval
"""

import argparse
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="RRQ lm-eval accuracy benchmark")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Original HF model name")
    ap.add_argument("--base-dir", default="./rrq_output/base", help="Base model dir")
    ap.add_argument("--residual-dir", default="./rrq_output/residual", help="Residual model dir")
    ap.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge,piqa,boolq",
                    help="Comma-separated lm-eval task names")
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 4, 6, 8],
                    help="Bit-widths to evaluate (2, 4, 6, 8)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max examples per task (None = full dataset)")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size")
    ap.add_argument("--device", default="cpu", help="cpu / cuda / xpu")
    ap.add_argument("--skip-fp", action="store_true", help="Skip original fp model reference")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from auto_round import load_rrq_model
    from auto_round.inference.rrq_linear import set_rrq_bits
    from transformers import AutoTokenizer

    # ── Setup ──────────────────────────────────────────────────────────────
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    base_dir = args.base_dir
    residual_dir = args.residual_dir

    for d in (base_dir, residual_dir):
        if not Path(d).is_dir():
            print(f"Error: {d} does not exist. Run test_rrq_qwen3_06b.py first.")
            return

    print(f"\n{'='*70}")
    print(f"  RRQ lm-eval accuracy benchmark")
    print(f"  Model     : {args.model}")
    print(f"  Base dir  : {base_dir}")
    print(f"  Residual  : {residual_dir}")
    print(f"  Tasks     : {tasks}")
    print(f"  Bits      : {args.bits}")
    print(f"  Limit     : {args.limit or 'full'}")
    print(f"  Device    : {args.device}")
    print(f"{'='*70}\n")

    # ── Helper: run lm-eval on a model ─────────────────────────────────────
    def eval_model(model, label: str) -> dict:
        """Run lm-eval and return {task: acc}."""
        import os
        import lm_eval
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokenizer = AutoTokenizer.from_pretrained(base_dir)

        # Wrap model for lm-eval HFLM interface
        from lm_eval.models.huggingface import HFLM
        hflm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device=args.device,
            trust_remote_code=True,
        )

        # Compatible with both old (lm_eval.evaluator) and new (lm_eval) APIs
        try:
            simple_eval_fn = lm_eval.simple_evaluate
        except AttributeError:
            from lm_eval.evaluator import simple_evaluate as simple_eval_fn

        results = simple_eval_fn(
            model=hflm,
            tasks=tasks,
            limit=args.limit,
            num_fewshot=None,  # use task default
            batch_size=args.batch_size,
            log_samples=False,
            verbosity="ERROR",
        )

        # Extract accuracy per task
        task_accs = {}
        for task in tasks:
            # lm-eval returns results with various metric names; find the primary one
            entry = results.get("results", {}).get(task, {})
            # Common metric names: "acc,none", "acc_norm,none", "acc"
            for metric_key in entry:
                if metric_key.startswith(("acc", "exact_match")):
                    task_accs[task] = entry[metric_key]
                    break
            if task not in task_accs:
                # fallback: first metric
                for metric_key, val in entry.items():
                    if isinstance(val, (int, float)):
                        task_accs[task] = val
                        break

        return task_accs

    # ── Evaluate each bit-width ────────────────────────────────────────────
    results_by_bits = {}
    for bits in args.bits:
        print(f"\n{'─'*70}")
        print(f"  Evaluating {bits}-bit ({bits//2} planes) ...")
        print(f"{'─'*70}")

        t0 = time.time()
        model = load_rrq_model(
            base_model_dir=base_dir,
            residual_model_dir=residual_dir,
            active_bits=bits,
            device=args.device,
        )
        load_time = time.time() - t0
        print(f"  loaded in {load_time:.1f}s")

        t1 = time.time()
        accs = eval_model(model, f"{bits}bit")
        eval_time = time.time() - t1
        print(f"  eval done in {eval_time:.1f}s")

        for task, acc in accs.items():
            print(f"    {task:<25s} {acc:.4%}")

        results_by_bits[bits] = accs
        del model
        torch.cuda.empty_cache() if args.device == "cuda" else None

    # ── Original fp model (reference) ──────────────────────────────────────
    if not args.skip_fp:
        print(f"\n{'─'*70}")
        print(f"  Evaluating original fp model (reference) ...")
        print(f"{'─'*70}")

        from transformers import AutoModelForCausalLM

        t0 = time.time()
        fp_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        fp_model.to(args.device)
        print(f"  loaded in {time.time()-t0:.1f}s")

        t1 = time.time()
        fp_accs = eval_model(fp_model, "fp32")
        print(f"  eval done in {time.time()-t1:.1f}s")
        for task, acc in fp_accs.items():
            print(f"    {task:<25s} {acc:.4%}")
        results_by_bits["fp32"] = fp_accs
        del fp_model
        torch.cuda.empty_cache() if args.device == "cuda" else None

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ACCURACY SUMMARY")
    print(f"{'='*70}")

    # Header
    bits_labels = [str(b) for b in args.bits]
    header = f"{'Task':<25s}" + "".join(f"{label:>12s}" for label in bits_labels)
    if not args.skip_fp:
        header += f"{'fp32':>12s}"
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    # Rows
    for task in tasks:
        row = f"  {task:<25s}"
        for bits in args.bits:
            acc = results_by_bits.get(bits, {}).get(task, 0)
            row += f"{acc:>12.4%}"
        if not args.skip_fp:
            acc = results_by_bits.get("fp32", {}).get(task, 0)
            row += f"{acc:>12.4%}"
        print(row)

    # Mean row
    print(f"  {'-'*len(header)}")
    row = f"  {'Mean':<25s}"
    for bits in args.bits:
        accs = [results_by_bits.get(bits, {}).get(t, 0) for t in tasks]
        row += f"{sum(accs)/len(accs):>12.4%}" if accs else f"{'N/A':>12s}"
    if not args.skip_fp:
        accs = [results_by_bits.get("fp32", {}).get(t, 0) for t in tasks]
        row += f"{sum(accs)/len(accs):>12.4%}" if accs else f"{'N/A':>12s}"
    print(row)

    # Delta vs fp32
    if not args.skip_fp:
        print(f"\n  {'Δ vs fp32':<25s}", end="")
        fp_mean = sum(results_by_bits.get("fp32", {}).get(t, 0) for t in tasks) / len(tasks)
        for bits in args.bits:
            accs = [results_by_bits.get(bits, {}).get(t, 0) for t in tasks]
            mean = sum(accs) / len(accs) if accs else 0
            delta = (mean - fp_mean) * 100  # percentage points
            print(f"{delta:>+11.2f}pp", end="")
        print()

    print(f"\n{'='*70}\n")
    print("  Higher is better. Δ vs fp32 shows accuracy change in percentage points.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
