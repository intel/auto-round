#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import requests

RAW_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp"
API_URL = "https://api.github.com/repos/ggml-org/llama.cpp"
DEST = Path(__file__).resolve().parent


def latest_commit() -> str:
    response = requests.get(f"{API_URL}/commits/master", timeout=30)
    response.raise_for_status()
    return response.json()["sha"]


def list_conversion_files(commit: str) -> list[str]:
    response = requests.get(f"{API_URL}/git/trees/{commit}?recursive=1", timeout=30)
    response.raise_for_status()
    tree = response.json()["tree"]
    return [item["path"] for item in tree if item["type"] == "blob" and item["path"].startswith("conversion/")]


def download_file(commit: str, path: str, output: Path) -> None:
    response = requests.get(f"{RAW_URL}/{commit}/{path}", timeout=30)
    response.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(response.text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync llama.cpp conversion sources used by GGUF export.")
    parser.add_argument("--commit", default=None, help="llama.cpp commit SHA; defaults to master HEAD")
    args = parser.parse_args()

    commit = args.commit or latest_commit()
    conversion_dir = DEST / "conversion"
    shutil.rmtree(conversion_dir, ignore_errors=True)

    for path in list_conversion_files(commit):
        download_file(commit, path, DEST / path)

    (DEST / "LLAMA_CPP_CONVERSION_COMMIT").write_text(commit + "\n", encoding="utf-8")
    print(f"Synced llama.cpp conversion at {commit} to {conversion_dir}")


if __name__ == "__main__":
    main()
