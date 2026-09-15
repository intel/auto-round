"""Post-install smoke test for the compatibility pipeline.

Run as a standalone script (``python .azure-pipelines/scripts/compat_smoke_test.py``)
so that ``import auto_round`` resolves to the *installed* package instead of the
source tree at the repository root (``sys.path[0]`` becomes this script's directory,
and the current working directory is not added for script execution).

It validates that:
  * the package imports and exposes ``__version__``;
  * the public ``AutoRound`` entry class is importable;
  * a registered console script (``auto-round``) is installed and runnable.
"""

import subprocess
import sys

import auto_round
from auto_round import AutoRound

print(f"auto_round imported from: {auto_round.__file__}")
print(f"auto_round {auto_round.__version__} imported successfully (AutoRound={AutoRound.__name__})")

# Verify the console_scripts entry point was installed and is runnable.
result = subprocess.run(["auto-round", "--help"], capture_output=True, text=True)
if result.returncode != 0:
    sys.stderr.write(result.stdout)
    sys.stderr.write(result.stderr)
    raise SystemExit(f"`auto-round --help` failed with exit code {result.returncode}")

print("console script `auto-round` is installed and runnable")
