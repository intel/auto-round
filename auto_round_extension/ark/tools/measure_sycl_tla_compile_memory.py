#!/usr/bin/env python3

# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

RSS_PATTERN = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def load_compile_commands(path):
    entries = json.loads(path.read_text(encoding="utf-8"))
    commands = []
    for entry in entries:
        if "arguments" in entry:
            argv = list(entry["arguments"])
        elif "command" in entry:
            argv = shlex.split(entry["command"])
        else:
            raise ValueError("compile_commands entry has neither 'arguments' nor 'command'")
        source = Path(entry["file"])
        if not source.is_absolute():
            source = Path(entry["directory"]) / source
        commands.append(
            {
                "source": source.resolve(),
                "directory": Path(entry["directory"]).resolve(),
                "argv": argv,
            }
        )
    return commands


def measure_command(command, time_binary):
    completed = subprocess.run(
        [time_binary, "-v", *command["argv"]],
        cwd=command["directory"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    match = RSS_PATTERN.search(completed.stderr)
    result = {
        "source": str(command["source"]),
        "returncode": completed.returncode,
        "max_rss_kib": int(match.group(1)) if match else None,
        "max_rss_mib": int(match.group(1)) / 1024 if match else None,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "command": command["argv"],
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Measure peak RSS for each compiler invocation in compile_commands.json."
    )
    parser.add_argument("--build-dir", type=Path, required=True, help="CMake build directory")
    parser.add_argument(
        "--compile-commands",
        type=Path,
        help="compile_commands.json path; defaults to <build-dir>/compile_commands.json",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "auto_round_kernel",
        help="ARK auto_round_kernel source directory",
    )
    parser.add_argument("--source-regex", help="Only measure translation units whose path matches this regex")
    parser.add_argument("--time-binary", default="/usr/bin/time", help="GNU time executable")
    parser.add_argument("--output", type=Path, help="Write detailed JSON results")
    args = parser.parse_args()

    compile_commands = (args.compile_commands or args.build_dir / "compile_commands.json").resolve()
    if not compile_commands.is_file():
        raise SystemExit(f"compile commands not found: {compile_commands}")
    if not Path(args.time_binary).is_file():
        raise SystemExit(f"time executable not found: {args.time_binary}")

    commands = load_compile_commands(compile_commands)
    if args.source_regex:
        source_pattern = re.compile(args.source_regex)
        commands = [command for command in commands if source_pattern.search(str(command["source"]))]
    if not commands:
        raise SystemExit("no compile commands selected")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_sycl_tla_templates import analyze_sources

    analysis = analyze_sources(args.root.resolve(), args.build_dir.resolve())
    analysis_by_source = {str(Path(item["source"]).resolve()): item for item in analysis}

    results = []
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {command['source']}", flush=True)
        result = measure_command(command, args.time_binary)
        result.update(analysis_by_source.get(str(command["source"]), {}))
        direct_dispatches = sum(result.get("direct_dispatch_calls", {}).values())
        result["rss_per_direct_dispatch_mib"] = (
            result["max_rss_mib"] / direct_dispatches
            if result["max_rss_mib"] is not None and direct_dispatches
            else None
        )
        results.append(result)
        rss = result["max_rss_mib"]
        rss_text = f"{rss:.1f} MiB" if rss is not None else "unknown RSS"
        print(f"  returncode={result['returncode']} peak RSS={rss_text}", flush=True)

    measured = [result for result in results if result["max_rss_kib"] is not None]
    measured.sort(key=lambda result: result["max_rss_kib"], reverse=True)
    print("source | peak RSS MiB | RSS/direct MiB | return code | direct dispatch | declarations")
    print("-------|---------------|----------------|-------------|-----------------|-------------")
    for result in measured:
        direct = sum(result.get("direct_dispatch_calls", {}).values())
        per_dispatch = result["rss_per_direct_dispatch_mib"]
        per_dispatch_text = f"{per_dispatch:14.1f}" if per_dispatch is not None else "             n/a"
        print(
            f"{result.get('label', result['source'])} | {result['max_rss_mib']:13.1f} | {per_dispatch_text} | "
            f"{result['returncode']:11d} | {direct:15d} | {result.get('template_declarations', 'n/a'):11}"
        )
    if measured:
        print(
            f"\npeak compiler RSS: {measured[0]['max_rss_mib']:.1f} MiB ({measured[0].get('label', measured[0]['source'])})"
        )
    else:
        print("\nNo GNU time RSS measurements were parsed.")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
