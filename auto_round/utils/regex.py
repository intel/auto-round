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

from auto_round.logger import logger


def to_standard_regex(pattern: str) -> str:
    """
    Convert a user-specified string into a standardized regex for layer matching.

    Rules:
    - If the pattern already contains regex tokens ('.*', '^', '$', etc.),
      keep them as-is.
    - Otherwise, wrap the pattern with `.*` on both sides to allow substring matching.
    - Always ensure the returned regex is valid (compilable by re).

    Examples:
    >>> to_standard_regex("model.embed_tokens")
    '.*model\\.embed_tokens.*'
    >>> to_standard_regex("mlp.gate")
    '.*mlp\\.gate.*'
    >>> to_standard_regex("mlp.gate$")
    '.*mlp\\.gate$'
    >>> to_standard_regex("mlp.*gate")
    '.*mlp.*gate.*'
    """
    # Heuristic: if pattern contains regex meta characters, assume partial regex
    meta_chars = {"*", "{", "}", "^", "$", "|", "(", ")", "[", "]", "?", "+", "\\"}
    has_regex = any(tok in pattern for tok in meta_chars)
    if not has_regex:
        # Escape literal dots, etc., and wrap with .* for substring matching
        pattern = re.escape(pattern)
        regex = f".*{pattern}.*"
    else:
        # Only escape bare dots that are not already part of regex constructs:
        # keep escape sequences (e.g. "\\.", "\\d") and quantified dots
        # (".*", ".+", ".?", ".{n}") unchanged.
        tmp = []
        i = 0
        while i < len(pattern):
            if pattern[i] == "\\" and i + 1 < len(pattern):
                tmp.append(pattern[i : i + 2])  # keep escape sequence
                i += 2
                continue
            if pattern[i] == ".":
                if i + 1 < len(pattern) and pattern[i + 1] in "*+?{":
                    tmp.append(".")  # keep regex token
                else:
                    tmp.append("\\.")  # escape bare dot
            else:
                tmp.append(pattern[i])
            i += 1
        regex = "".join(tmp)
        # If no anchors are provided, allow substring matching
        if not regex.startswith("^") and not regex.startswith(".*"):
            regex = ".*" + regex
        if not regex.endswith("$") and not regex.endswith(".*"):
            regex = regex + ".*"
    # Validate regex
    try:
        re.compile(regex)
    except re.error as e:
        raise ValueError(f"Invalid regex generated from pattern '{pattern}': {e}")
    return regex


def matches_any_regex(layer_name: str, regex_config: dict[str, dict]) -> bool:
    """
    Check whether `layer_name` matches any regex pattern key in `regex_config`.
    Args:
        layer_name (str): The layer name to test.
        regex_config (dict[str, dict]): A mapping of regex patterns to configs.
    Returns:
        bool: True if any pattern matches `layer_name`, otherwise False.
    """
    if not regex_config:
        return False

    for pattern in regex_config:
        # Strip dynamic prefixes (e.g., "+:" or "-:")
        raw_pattern = pattern[2:] if pattern.startswith(("+:", "-:")) else pattern

        try:
            if re.search(raw_pattern, layer_name):
                return True
        except re.error as e:
            logger.warning("Skipping invalid regex pattern %r: %s", pattern, e)
            continue

    return False


def _numeric_values_to_regex(numbers: list[int]) -> str:
    digit_trie = {"terminal": False, "children": {}}
    for number in sorted(set(numbers)):
        current = digit_trie
        for digit in str(number):
            current = current["children"].setdefault(digit, {"terminal": False, "children": {}})
        current["terminal"] = True

    def render(node):
        grouped = {}
        for digit, child in node["children"].items():
            grouped.setdefault(render(child), []).append(digit)
        alternatives = []
        for suffix, digits in grouped.items():
            digits = sorted(digits)
            ranges = []
            start = previous = digits[0]
            for digit in digits[1:]:
                if ord(digit) == ord(previous) + 1:
                    previous = digit
                    continue
                ranges.append((start, previous))
                start = previous = digit
            ranges.append((start, previous))
            parts = [first if first == last else f"{first}-{last}" for first, last in ranges]
            char_class = re.escape(parts[0]) if len(parts) == 1 and len(parts[0]) == 1 else "[" + "".join(parts) + "]"
            alternatives.append(char_class + suffix)
        if node["terminal"]:
            alternatives.append("")
        if len(alternatives) == 1:
            return alternatives[0]
        return "(?:" + "|".join(alternatives) + ")"

    return render(digit_trie)


def compress_regex_patterns(patterns: list[str]) -> list[str]:
    """Merge regexes that differ only by one numeric dot-separated path segment."""
    current = list(dict.fromkeys(patterns))
    while True:
        buckets = {}
        for pattern in current:
            prefix = "re:" if pattern.startswith("re:") else ""
            body = pattern[len(prefix) :]
            for match_index, match in enumerate(re.finditer(r"(?<=\\\.)(?:0|[1-9][0-9]*)(?=\\\.)", body)):
                marker = f"<NUMBER_{match_index}>"
                template = body[: match.start()] + marker + body[match.end() :]
                buckets.setdefault((prefix, match_index, template), []).append((pattern, int(match.group())))

        candidates = []
        for (prefix, index, template), entries in buckets.items():
            numbers = sorted({number for _, number in entries})
            if len(numbers) < 2:
                continue
            marker = f"<NUMBER_{index}>"
            candidate = prefix + template.replace(marker, _numeric_values_to_regex(numbers))
            originals = {pattern for pattern, _ in entries}
            if candidate not in originals and len(candidate) + 2 < sum(len(pattern) + 3 for pattern in originals):
                candidates.append((sum(len(pattern) for pattern in originals) - len(candidate), originals, candidate))

        if not candidates:
            return current

        _, originals, candidate = max(candidates, key=lambda item: item[0])
        updated = []
        inserted = False
        for pattern in current:
            if pattern in originals:
                if not inserted:
                    updated.append(candidate)
                    inserted = True
            else:
                updated.append(pattern)
        current = updated


def _collapse_expert_overrides(extra_config: dict, exact_layer_names: set[str]) -> tuple[dict, set[str]]:
    """Replace per-expert keys with their ``...experts`` container key when every expert shares one config.

    vLLM INC resolves a FusedMoE layer by its exact container name, and the AutoRound loader re.search-es the key,
    so the container key is only emitted when its re.search match set equals exactly those experts.
    """
    containers = {}
    for name in exact_layer_names:
        parts = name.split(".")
        for idx in range(1, len(parts) - 1):
            if parts[idx] == "experts" and parts[idx + 1].isdigit():
                containers.setdefault(".".join(parts[: idx + 1]), set()).add(name)
                break

    replaced = {}
    for container, children in containers.items():
        if container in extra_config or container in exact_layer_names or len(children) < 2:
            continue
        configs = [extra_config.get(child) for child in children]
        if configs[0] is None or any(config != configs[0] for config in configs):
            continue
        pattern = re.compile(container)
        if any(pattern.search(name) for name in exact_layer_names - children):
            continue
        replaced.update(dict.fromkeys(children, container))

    collapsed = {}
    for name, config in extra_config.items():
        collapsed.setdefault(replaced.get(name, name), config)
    return collapsed, set(replaced.values())


def compress_autoround_extra_config(extra_config: dict, exact_layer_names: set[str]) -> dict:
    """Collapse uniform expert overrides and merge adjacent regex overrides that share a config.

    Other exact layer keys stay literal: vLLM and the AutoRound loader remap, suffix-match and prefix-match them.
    ``exact_layer_names`` must list every quantizable layer, otherwise a collapsed key may cover unlisted experts.
    """
    extra_config, containers = _collapse_expert_overrides(extra_config, exact_layer_names)
    exact_keys = exact_layer_names | containers
    compacted = []
    run = []

    def flush_regex_run():
        if run:
            compacted.extend((name, run[0][1]) for name in compress_regex_patterns([name for name, _ in run]))
            run.clear()

    for name, config in extra_config.items():
        if name in exact_keys:
            flush_regex_run()
            compacted.append((name, config))
            continue
        if run and run[-1][1] != config:
            flush_regex_run()
        run.append((name, config))
    flush_regex_run()
    return dict(compacted)
