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

# Thin shim - all logic lives in auto_round.cli.main.
# This file exists solely to satisfy setup.cfg console_scripts entry points.
from auto_round.cli.main import run, run_best, run_eval, run_light, run_mllm, run_opt_rtn, run_rtn  # noqa: F401

if __name__ == "__main__":
    run()
