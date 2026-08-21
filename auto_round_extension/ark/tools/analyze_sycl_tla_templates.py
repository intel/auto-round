#!/usr/bin/env python3

# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path

LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
TEMPLATE_DECLARATION = re.compile(r"^\s*template\s*<")

CALL_PATTERNS = {
    "dense_gemm": re.compile(r"\b(?:gemm_cute_store_tile\s*<|run_dense_gemm\s*\()"),
    "s8_gemm": re.compile(r"\b(?:launch_igemm\s*<|launch_igemm_kblock\s*<|run_typed\s*\()"),
    "moe_gemm": re.compile(r"\bmoe_gemm_launcher\s*<"),
    "moe_decode": re.compile(r"\blaunch_(?:fp|int4|int8|int2|fp8)\s*<"),
    "moe_prefill": re.compile(r"\b(?:moe_prefill_[a-z0-9_]+_dispatch|dequant_to_KN|launch_dequant_fp8_slm)\s*<"),
    "sdpa": re.compile(r"\blaunch_(?:sage|prefill|sparse)[a-z0-9_]*\s*<"),
}

ESTIMATED_KERNELS = {
    "dense_gemm": 24,
    "s8_gemm": 96,
    "moe_gemm": 2,
    "moe_decode": 22,
    "moe_prefill": 44,
    "sdpa": 14,
}


def resolve_include(include_name, including_file, files_by_name):
    candidate = including_file.parent / include_name
    if candidate.is_file():
        return candidate.resolve()
    for parent in including_file.parents:
        candidate = parent / "wrapper" / "include" / include_name
        if candidate.is_file():
            return candidate.resolve()
    matches = [path for path in files_by_name.get(Path(include_name).name, []) if path.name == Path(include_name).name]
    return matches[0].resolve() if len(matches) == 1 else None


def reachable_files(source, files_by_name):
    result = []
    pending = deque([source.resolve()])
    visited = set()
    while pending:
        current = pending.popleft()
        if current in visited or not current.is_file():
            continue
        visited.add(current)
        result.append(current)
        for line in current.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = LOCAL_INCLUDE.match(line)
            if match:
                included = resolve_include(match.group(1), current, files_by_name)
                if included is not None:
                    pending.append(included)
    return result


def analyze_file(path):
    declarations = 0
    calls = defaultdict(int)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if TEMPLATE_DECLARATION.match(line):
            declarations += 1
            continue
        for family, pattern in CALL_PATTERNS.items():
            calls[family] += len(pattern.findall(line))
    return declarations, calls


def compile_command_sources(build_dir):
    if build_dir is None:
        return []
    compile_commands = build_dir / "compile_commands.json"
    if not compile_commands.is_file():
        return []
    entries = json.loads(compile_commands.read_text(encoding="utf-8"))
    sources = []
    for entry in entries:
        source = Path(entry["file"])
        if not source.is_absolute():
            source = Path(entry.get("directory", build_dir)) / source
        source = source.resolve()
        if source.is_file() and source.suffix in {".c", ".cc", ".cpp", ".cxx"}:
            sources.append(source)
    return sorted(set(sources))


def source_files(root, build_dir=None):
    sources = set(root.glob("*.cpp"))
    sources.update(compile_command_sources(build_dir))
    return sorted(path.resolve() for path in sources if path.is_file())


def source_label(source, root):
    try:
        return str(source.relative_to(root))
    except ValueError:
        return str(source)


def analyze_sources(root, build_dir=None):
    files_by_name = defaultdict(list)
    indexed_roots = [root]
    if build_dir is not None and (build_dir / "generated").is_dir():
        indexed_roots.append(build_dir / "generated")
    for indexed_root in indexed_roots:
        for path in indexed_root.rglob("*"):
            if path.is_file():
                files_by_name[path.name].append(path)

    results = []
    for source in source_files(root, build_dir):
        reachable = reachable_files(source, files_by_name)
        declarations = 0
        reachable_calls = defaultdict(int)
        for included in reachable:
            file_declarations, file_calls = analyze_file(included)
            declarations += file_declarations
            for family, count in file_calls.items():
                reachable_calls[family] += count
        direct_declarations, direct_calls = analyze_file(source)
        results.append(
            {
                "source": str(source),
                "label": source_label(source, root),
                "reachable_files": len(reachable),
                "template_declarations": declarations,
                "direct_template_declarations": direct_declarations,
                "reachable_dispatch_calls": dict(sorted(reachable_calls.items())),
                "direct_dispatch_calls": dict(sorted(direct_calls.items())),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Statically count SYCL-TLA template dispatch sites.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "auto_round_kernel",
        help="ARK auto_round_kernel source directory",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="CMake build directory; compile_commands.json adds generated translation units",
    )
    parser.add_argument("--json", type=Path, help="Write machine-readable per-translation-unit results")
    args = parser.parse_args()
    root = args.root.resolve()
    build_dir = args.build_dir.resolve() if args.build_dir else None
    results = analyze_sources(root, build_dir)

    print(f"root: {root}")
    print("source | reachable files | declarations | direct dispatch | reachable dispatch")
    print("-------|------------------|--------------|-----------------|-------------------")
    for result in results:
        direct = sum(result["direct_dispatch_calls"].values())
        reachable = sum(result["reachable_dispatch_calls"].values())
        print(
            f"{result['label']} | {result['reachable_files']:16d} | {result['template_declarations']:12d} | "
            f"{direct:15d} | {reachable:19d}"
        )

    print("\nestimated concrete kernel combinations")
    estimated_total = sum(ESTIMATED_KERNELS.values())
    for family, count in ESTIMATED_KERNELS.items():
        print(f"{family}: {count}")
    print(f"estimated total: {estimated_total}")
    max_result = max(results, key=lambda result: sum(result["direct_dispatch_calls"].values()), default=None)
    if max_result:
        print(
            f"max direct-dispatch source: {max_result['label']} "
            f"({sum(max_result['direct_dispatch_calls'].values())} call sites)"
        )
    print(f"translation units analyzed: {len(results)}")
    print(
        "\nReachable declarations include header definitions parsed by each translation unit; direct dispatch counts only calls written in the .cpp file."
    )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
