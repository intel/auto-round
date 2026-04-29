# Copyright (c) 2026 Intel Corporation
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

import re
import subprocess
import sys


def check_torch_version():
    try:
        import torch

        m = re.match(r"^(\d+)\.(\d+)", torch.__version__)
        if m:
            major, minor = int(m.group(1)), int(m.group(2))
            if major < 2 or (major == 2 and minor < 10):
                raise RuntimeError(
                    f"Torch version 2.10 or higher is required for oneAPI 2025.3 environment. "
                    f"Found: {torch.__version__}"
                )
            return True
        return False
    except ImportError:
        raise RuntimeError("PyTorch is not installed. Please install torch>=2.10.0 first.")


def main():
    print("Checking environment for oneAPI 2025.3 compatibility...")
    check_torch_version()

    pkg = "auto-round-lib"

    print(f"Environment check passed. Installing {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--upgrade-strategy", "only-if-needed"])


if __name__ == "__main__":
    main()
