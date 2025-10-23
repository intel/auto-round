#!/bin/bash

# Copyright (c) 2025 Intel Corporation
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

# Define the approximate path for the `auto-round` installation
AUTO_ROUND_PATH="auto_round/experimental/vllm_ext/sitecustomize.py"

# Use pip to find the installation path of the `auto-round` package
PIP_LOCATION=$(pip show auto-round 2>/dev/null | grep "Location:" | awk '{print $2}')

if [ -n "$PIP_LOCATION" ]; then
    # Construct the full path to `sitecustomize.py`
    SITE_CUSTOMIZE_PATH="$PIP_LOCATION/$AUTO_ROUND_PATH"

    if [ -f "$SITE_CUSTOMIZE_PATH" ]; then
        echo "Found sitecustomize.py at: $SITE_CUSTOMIZE_PATH"
        export PYTHONPATH=$(dirname "$SITE_CUSTOMIZE_PATH"):$PYTHONPATH
        echo "PYTHONPATH set to: $PYTHONPATH"
    else
        echo "Error: sitecustomize.py not found in the auto-round installation path."
        exit 1
    fi
else
    echo "Error: auto-round package not found via pip."
    exit 1
fi