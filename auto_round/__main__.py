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

def run():
    from auto_round.script.llm import setup_parser, tune, eval
    args = setup_parser()
    if args.eval:
        eval(args)
    else:
        tune(args)

def run_best():
    from auto_round.script.llm import setup_best_parser, tune
    args = setup_best_parser()
    tune(args)

def run_fast():
    from auto_round.script.llm import setup_fast_parser, tune
    args = setup_fast_parser()
    tune(args)


def run_mllm():
    from auto_round.script.mllm import setup_parser, tune, eval
    args = setup_parser()
    if args.eval:
        eval(args)
    else:
        tune(args)

def run_lmms():
    from auto_round.script.lmms_eval import setup_lmms_args, eval
    args = setup_lmms_args()
    eval(args)

def switch():
    if "--lmms" in sys.argv:
        sys.argv.remove("--lmms")
        run_lmms()
    elif "--mllm" in sys.argv:
        sys.argv.remove("--mllm")
        run_mllm()
    else:
        run()

if __name__ == '__main__':
    switch()
