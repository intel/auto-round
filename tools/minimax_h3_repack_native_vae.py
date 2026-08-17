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
"""Repack an existing AutoRound MiniMax-H3 export for native Omni loading."""

from __future__ import annotations

import argparse

from minimax_h3_native_vae_layout import repack_native_vae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Existing AutoRound export to update in place")
    parser.add_argument(
        "--official-fl2va",
        required=True,
        help="Original MiniMax-H3 model root or its FL2VA release directory",
    )
    args = parser.parse_args()
    partition = repack_native_vae(args.checkpoint, args.official_fl2va)
    print(f"Created native Omni partition at {partition}")


if __name__ == "__main__":
    main()
