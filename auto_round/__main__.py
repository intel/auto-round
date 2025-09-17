# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys


def run_eval():
    from auto_round.script.run import setup_eval_parser

    args = setup_eval_parser()
    if args.eval_task_by_task:
        from auto_round.script.run import eval_task_by_task

        eval_task_by_task(
            model=args.model,
            device=args.device,
            tasks=args.tasks,
            batch_size=args.eval_bs,
            trust_remote_code=not args.disable_trust_remote_code,
            eval_model_dtype=args.eval_model_dtype,
        )
    else:
        from auto_round.script.run import eval

        eval(args)


def run():
    if "--eval" in sys.argv:
        sys.argv.remove("--eval")
        run_eval()
    else:
        from auto_round.script.run import setup_parser, tune

        args = setup_parser()
        tune(args)


def run_mllm():
    sys.argv.append("--mllm")
    run()


def run_best():
    from auto_round.script.run import setup_best_parser, tune

    args = setup_best_parser()
    tune(args)


def run_light():
    from auto_round.script.run import setup_light_parser, tune

    args = setup_light_parser()
    tune(args)


def run_fast():
    from auto_round.script.run import setup_fast_parser, tune

    args = setup_fast_parser()
    tune(args)


if __name__ == "__main__":
    run()
