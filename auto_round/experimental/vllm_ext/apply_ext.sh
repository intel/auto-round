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

# Define the relative path for the `auto-round` installation
AUTO_ROUND_PATH="auto_round/experimental/vllm_ext/sitecustomize.py"

# Try to find the pip installation location
PIP_LOCATION=$(pip show auto-round 2>/dev/null | grep "Location:" | awk '{print $2}')

if [ -n "$PIP_LOCATION" ]; then
    SITE_CUSTOMIZE_PATH="$PIP_LOCATION/$AUTO_ROUND_PATH"
    echo "Checking for sitecustomize.py at: $SITE_CUSTOMIZE_PATH"

    if [ -f "$SITE_CUSTOMIZE_PATH" ]; then
        echo "Found sitecustomize.py at: $SITE_CUSTOMIZE_PATH"
        export PYTHONPATH=$(dirname "$SITE_CUSTOMIZE_PATH"):$PYTHONPATH
        echo "PYTHONPATH set to: $PYTHONPATH"
        return 0 2>/dev/null || true
    fi
fi

# Fallback: check current directory
LOCAL_SITE_CUSTOMIZE="./sitecustomize.py"
if [ -f "$LOCAL_SITE_CUSTOMIZE" ]; then
    echo "Found sitecustomize.py at current directory."
    export PYTHONPATH=$(pwd):$PYTHONPATH
    echo "PYTHONPATH set to: $PYTHONPATH"
    return 0 2>/dev/null || true
fi

echo "Warning: sitecustomize.py not found in pip installation or current directory."
# Do not exit the shell
return 1 2>/dev/null || true