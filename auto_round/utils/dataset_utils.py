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
"""Utilities for flexible calibration dataset loading and field extraction.

This module provides:

- **CalibDataset** dataclass for a clean, typed API to describe calibration datasets.
- **Auto-detection** of the text field in a HuggingFace ``Dataset`` by inspecting
  column names and sampling values.
- **Field extraction** with support for single fields, multiple fields, and
  user-defined templates (``{field}`` placeholders).
- **Spec-string building and parsing** for CLI / string-based usage.

Example::

    from auto_round.utils.dataset_utils import CalibDataset

    # Single field
    spec = CalibDataset("my-org/my-dataset", split="train", num=1000, fields="text")

    # Multiple fields concatenated
    spec = CalibDataset("my-org/my-dataset", fields=["question", "answer"], separator="\\n")

    # Template
    spec = CalibDataset("my-org/my-dataset", template="{question} {answer}")

    # Pass directly to AutoRound
    from auto_round import AutoRound
    model = AutoRound(model, dataset=spec)
"""

import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# CalibDataset dataclass
# ---------------------------------------------------------------------------


@dataclass
class CalibDataset:
    """Structured specification for a calibration dataset.

    This is the preferred way to specify calibration data programmatically.
    It can be passed directly to ``AutoRound(dataset=...)`` or converted to a
    spec string via :meth:`to_spec_string`.

    Args:
        name: Dataset name (HuggingFace ID) or local file path.
        split: Split name(s).  A single string or a list of strings.
        num: Number of samples to take from this dataset.
        concat: Whether to concatenate samples into full-length sequences.
        fields: Text field(s) to extract.  A single string for one column,
            or a list of strings for multiple columns (concatenated with
            *separator*).  If ``None``, the text field is auto-detected.
        template: A format string with ``{field_name}`` placeholders.
            Takes precedence over *fields* when both are set.
        separator: Separator used when concatenating multiple fields.
            Defaults to ``"\\n\\n"``.
        apply_chat_template: Whether to apply the tokenizer's chat template.
        system_prompt: Optional system prompt for chat template.
        timeout: Seconds to wait for data collection before falling back to
            whatever has been collected so far (default: 300).

    Example::

        >>> spec = CalibDataset("my-org/my-ds", split="train", num=500, fields="text")
        >>> spec = CalibDataset("my-org/my-ds", fields=["question", "answer"], separator="\\n")
        >>> spec = CalibDataset("my-org/my-ds", template="{question} {answer}")
    """

    name: str
    split: Optional[Union[str, List[str]]] = None
    num: Optional[int] = None
    concat: bool = False
    fields: Optional[Union[str, List[str]]] = None
    template: Optional[str] = None
    separator: str = "\n\n"
    apply_chat_template: bool = False
    system_prompt: Optional[str] = None
    timeout: int = 300

    def to_spec_string(self) -> str:
        """Convert to a spec string (the format used in CLI / ``dataset`` param).

        Returns:
            A string like ``"my/ds:split=train:num=100:fields=q+a"``.
        """
        parts = [self.name]

        if self.split is not None:
            if isinstance(self.split, list):
                parts.append(f"split={'+'.join(self.split)}")
            else:
                parts.append(f"split={self.split}")

        if self.num is not None:
            parts.append(f"num={self.num}")

        if self.concat:
            parts.append("concat=true")

        if self.template is not None:
            parts.append(f"template={self.template}")
        elif self.fields is not None:
            if isinstance(self.fields, str):
                parts.append(f"fields={self.fields}")
            else:
                parts.append(f"fields={'+'.join(self.fields)}")

        # Only include separator if it differs from the default and fields/template is used
        if self.separator != "\n\n" and (self.template is not None or self.fields is not None):
            parts.append(f"separator={self.separator}")

        if self.apply_chat_template:
            parts.append("apply_chat_template=true")

        if self.system_prompt is not None:
            parts.append(f"system_prompt={self.system_prompt}")

        if self.timeout != 300:
            parts.append(f"timeout={self.timeout}")

        return ":".join(parts)

    @classmethod
    def from_spec_string(cls, spec: str) -> "CalibDataset":
        """Parse a spec string into a ``CalibDataset`` instance.

        Args:
            spec: A spec string like ``"my/ds:split=train:num=100:fields=q+a"``.

        Returns:
            A ``CalibDataset`` instance.
        """
        return cls(**parse_dataset_spec(spec))


# ---------------------------------------------------------------------------
# Auto-detection of text fields
# ---------------------------------------------------------------------------

#: Priority-ordered list of column names commonly used for text content in
#: HuggingFace datasets.  The first match (in this order) wins when multiple
#: candidates are present.
TEXT_FIELD_PRIORITY: List[str] = [
    "text",
    "content",
    "prompt",
    "question",
    "instruction",
    "input",
    "code",
    "conversations",
    "messages",
    "dialog",
    "utterance",
    "utterances",
    "sentence",
    "paragraph",
    "article",
    "body",
    "abstract",
    "title",
    "caption",
    "description",
    "label",
    "answer",
    "response",
    "output",
    "completion",
    "translation",
    "summary",
    "context",
    "passage",
    "document",
    "doc",
    "raw_text",
    "raw",
]

#: Regex that matches ``{field_name}`` placeholders in a template string.
_TEMPLATE_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _is_text_value(value: Any) -> bool:
    """Return ``True`` if *value* looks like a text string (non-trivial length)."""
    if isinstance(value, str):
        return len(value.strip()) > 0
    return False


def _is_text_list(value: Any) -> bool:
    """Return ``True`` if *value* is a list of non-empty strings (e.g. messages)."""
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) == 0:
        return False
    # At least half of the elements should be non-empty strings
    text_count = sum(1 for v in value if isinstance(v, str) and len(v.strip()) > 0)
    return text_count >= len(value) // 2


def auto_detect_text_field(dataset, sample_size: int = 10) -> str:
    """Auto-detect the most likely text field in a HuggingFace ``Dataset``.

    Strategy (in order of priority):

    1. If the dataset has exactly **one** column, use it.
    2. Check for well-known text field names (``text``, ``content``, ``prompt``,
       ``question``, …) in :data:`TEXT_FIELD_PRIORITY` order.
    3. Among all remaining string columns, pick the one with the longest average
       text length across the first *sample_size* rows.

    Args:
        dataset: A HuggingFace ``Dataset`` or ``IterableDataset``.
        sample_size: Number of rows to sample for heuristic scoring.  Defaults
            to 10.

    Returns:
        The name of the detected text field.

    Raises:
        ValueError: If no suitable text field can be found.
    """
    # --- 1. Single-column shortcut ------------------------------------------
    if hasattr(dataset, "column_names"):
        columns = list(dataset.column_names)
    elif hasattr(dataset, "features"):
        columns = list(dataset.features.keys())
    else:
        # IterableDataset without column_names – grab one sample
        columns = list(next(iter(dataset)).keys())

    if len(columns) == 1:
        return columns[0]

    # --- 2. Well-known names ------------------------------------------------
    for candidate in TEXT_FIELD_PRIORITY:
        if candidate in columns:
            return candidate

    # --- 3. Heuristic: longest average string length ------------------------
    # Sample a few rows to estimate average text length per column.
    samples = []
    for i, row in enumerate(dataset):
        if i >= sample_size:
            break
        samples.append(row)

    if not samples:
        raise ValueError(f"Cannot auto-detect text field: dataset is empty. " f"Available columns: {columns}")

    best_field = None
    best_avg_len = 0.0
    for col in columns:
        total_len = 0
        count = 0
        for row in samples:
            val = row.get(col)
            if isinstance(val, str):
                total_len += len(val)
                count += 1
            elif isinstance(val, (list, tuple)):
                # Sum lengths of string elements (e.g. messages / conversations)
                for item in val:
                    if isinstance(item, str):
                        total_len += len(item)
                    elif isinstance(item, dict) and "content" in item:
                        total_len += len(str(item["content"]))
                count += 1
        if count > 0:
            avg_len = total_len / count
            if avg_len > best_avg_len:
                best_avg_len = avg_len
                best_field = col

    if best_field is None:
        raise ValueError(
            f"Cannot auto-detect a text field in dataset. "
            f"Available columns: {columns}. "
            f"Please specify the field explicitly via the 'field' or 'fields' parameter."
        )
    return best_field


def auto_detect_text_field_from_sample(sample: Dict[str, Any]) -> str:
    """Auto-detect the most likely text field from a single sample dict.

    Uses the same priority logic as :func:`auto_detect_text_field` but operates
    on a single row instead of a full dataset.  Useful for streaming /
    ``IterableDataset`` where column names may not be available upfront.

    Args:
        sample: A single row (dict) from the dataset.

    Returns:
        The name of the detected text field.

    Raises:
        ValueError: If no suitable text field can be found.
    """
    if len(sample) == 1:
        return list(sample.keys())[0]

    for candidate in TEXT_FIELD_PRIORITY:
        if candidate in sample:
            return candidate

    best_field = None
    best_len = 0
    for key, val in sample.items():
        if isinstance(val, str):
            if len(val) > best_len:
                best_len = len(val)
                best_field = key
        elif isinstance(val, (list, tuple)):
            total_len = sum(len(str(v)) for v in val if isinstance(v, str))
            if total_len > best_len:
                best_len = total_len
                best_field = key

    if best_field is None:
        raise ValueError(
            f"Cannot auto-detect a text field from sample. "
            f"Available fields: {list(sample.keys())}. "
            f"Please specify the field explicitly via the 'field' or 'fields' parameter."
        )
    return best_field


# ---------------------------------------------------------------------------
# Text extraction from a single sample
# ---------------------------------------------------------------------------


def extract_text_from_sample(
    sample: Dict[str, Any],
    fields: Optional[Union[str, List[str]]] = None,
    template: Optional[str] = None,
    separator: str = "\n\n",
) -> str:
    """Extract a text string from a single dataset sample.

    Three modes are supported (checked in order):

    1. **Template** – if *template* is given, ``{field_name}`` placeholders are
       replaced with the corresponding values from *sample*.
    2. **Multiple fields** – if *fields* is a list with more than one entry,
       the values are concatenated with *separator*.
    3. **Single field** – if *fields* is a string (or a single-element list),
       the value of that column is returned as-is.

    If none of the above are provided, the first string-valued column is used.

    Args:
        sample: A single row (dict) from the dataset.
        fields: A single field name (str) or a list of field names to
            concatenate.  If ``None``, auto-detection is used.
        template: A format string with ``{field_name}`` placeholders.
            Takes precedence over *fields*.
        separator: Separator used when concatenating multiple fields.
            Defaults to ``"\\n\\n"``.

    Returns:
        The extracted text string.

    Raises:
        KeyError: If a referenced field is not present in *sample*.
        ValueError: If no field can be determined.
    """
    # --- Template mode ------------------------------------------------------
    if template is not None:

        def _replace(match: re.Match) -> str:
            key = match.group(1)
            if key not in sample:
                raise KeyError(
                    f"Template references field '{key}' which is not in the sample. "
                    f"Available fields: {list(sample.keys())}"
                )
            val = sample[key]
            if isinstance(val, (list, tuple)):
                # Flatten list-of-dicts (e.g. messages) or list-of-strings
                parts = []
                for item in val:
                    if isinstance(item, dict):
                        parts.append(str(item.get("content", item)))
                    else:
                        parts.append(str(item))
                return separator.join(parts)
            return str(val)

        return _TEMPLATE_PLACEHOLDER_RE.sub(_replace, template)

    # --- Normalize fields to a list ------------------------------------------
    if fields is None:
        field_list: Optional[List[str]] = None
    elif isinstance(fields, str):
        field_list = [fields]
    else:
        field_list = list(fields)

    # --- Multi-field mode ---------------------------------------------------
    if field_list is not None and len(field_list) > 1:
        parts = []
        for f in field_list:
            if f not in sample:
                raise KeyError(f"Field '{f}' not found in sample. Available fields: {list(sample.keys())}")
            val = sample[f]
            if isinstance(val, (list, tuple)):
                parts.append(separator.join(str(v) for v in val if isinstance(v, str)))
            else:
                parts.append(str(val))
        return separator.join(parts)

    # --- Single-field mode --------------------------------------------------
    if field_list is not None and len(field_list) == 1:
        f = field_list[0]
        if f not in sample:
            raise KeyError(f"Field '{f}' not found in sample. Available fields: {list(sample.keys())}")
        val = sample[f]
        if isinstance(val, (list, tuple)):
            return separator.join(str(v) for v in val if isinstance(v, str))
        return str(val)

    # --- Fallback: first string column ---------------------------------------
    for key, val in sample.items():
        if isinstance(val, str) and len(val.strip()) > 0:
            return val
        if isinstance(val, (list, tuple)) and _is_text_list(val):
            return separator.join(str(v) for v in val if isinstance(v, str))

    raise ValueError(
        f"Cannot extract text from sample. Available fields: {list(sample.keys())}. "
        f"Please specify 'fields' or 'template' explicitly."
    )


# ---------------------------------------------------------------------------# Dataset normalization
# ---------------------------------------------------------------------------


def normalize_dataset_spec(dataset: Union[str, "CalibDataset", list, tuple]) -> Union[str, list, tuple]:
    """Normalize a dataset specification to a spec string or pass through raw data.

    Accepts a ``CalibDataset`` instance, a plain string, or a list/tuple.
    ``CalibDataset`` objects are converted to their spec-string representation
    via :meth:`CalibDataset.to_spec_string`.

    A list/tuple is treated as **raw calibration data** (returned unchanged)
    unless it consists solely of ``CalibDataset`` objects, in which case the
    specs are joined into a comma-separated spec string.  A list of plain
    strings is therefore raw calibration text (each sample is tokenized on the
    fly), not a list of dataset names.  To combine multiple *named* datasets,
    pass a single comma-separated string (e.g. ``"ds1,ds2"``) or a list of
    ``CalibDataset`` objects.

    Args:
        dataset: A dataset specification — a string, a ``CalibDataset``, or a
            list/tuple of raw samples / ``CalibDataset`` objects.

    Returns:
        A comma-separated spec string suitable for ``_get_dataset_impl``, or
        the original list/tuple when it holds raw calibration data.

    Raises:
        TypeError: If *dataset* is ``None``.
    """
    if dataset is None:
        raise TypeError(
            f"dataset must be a str, CalibDataset, or list of str/CalibDataset, got {type(dataset).__name__}"
        )
    if isinstance(dataset, CalibDataset):
        return dataset.to_spec_string()
    if isinstance(dataset, str):
        return dataset
    if isinstance(dataset, (list, tuple)):
<<<<<<< HEAD
        # A list/tuple is raw calibration data (returned unchanged) unless it
        # is a list of CalibDataset specs.  A list of plain strings is raw
        # calibration text (each sample tokenized on the fly), not a list of
        # dataset names.  Multiple named datasets use a comma-separated string
        # or a list of CalibDataset objects.
        if not dataset:
            return ""
        if all(isinstance(item, CalibDataset) for item in dataset):
            return ",".join(item.to_spec_string() for item in dataset)
        return dataset
    # Non-spec types (DataLoader, BatchEncoding, etc.) are passed through.
    return dataset
=======
        parts = []
        for item in dataset:
            if isinstance(item, CalibDataset):
                parts.append(item.to_spec_string())
            elif isinstance(item, str):
                parts.append(item)
            else:
                raise TypeError(f"Dataset list entries must be str or CalibDataset, got {type(item).__name__}")
        return ",".join(parts)
    raise TypeError(f"dataset must be a str, CalibDataset, or list of str/CalibDataset, got {type(dataset).__name__}")
>>>>>>> refs/rewritten/refine-data


# ---------------------------------------------------------------------------# Spec-string building and parsing
# ---------------------------------------------------------------------------

#: Keys that are recognised in a dataset spec string.
_SPEC_KEYS = {
    "split",
    "num",
    "concat",
    "apply_chat_template",
    "system_prompt",
    "fields",
    "template",
    "separator",
    "timeout",
}


def build_dataset_spec(
    name: str,
    split: Optional[Union[str, List[str]]] = None,
    num: Optional[int] = None,
    concat: bool = False,
    fields: Optional[Union[str, List[str]]] = None,
    template: Optional[str] = None,
    separator: str = "\n\n",
    apply_chat_template: bool = False,
    system_prompt: Optional[str] = None,
    timeout: int = 300,
) -> str:
    """Build a dataset spec string from individual parameters.

    This is the inverse of :func:`parse_dataset_spec` and is useful for
    programmatically constructing dataset specifications.

    Args:
        name: Dataset name (HuggingFace ID or local path).
        split: Split name(s).  A single string or a list of strings.
        num: Number of samples to take from this dataset.
        concat: Whether to concatenate samples into full-length sequences.
        fields: Text field(s) to extract.  A single string for one column,
            or a list of strings for multiple columns (concatenated with
            *separator*).  If ``None``, the text field is auto-detected.
        template: A format string with ``{field_name}`` placeholders.
            Takes precedence over *fields* when both are set.
        separator: Separator used when concatenating multiple fields.
            Defaults to ``"\\n\\n"``.
        apply_chat_template: Whether to apply the tokenizer's chat template.
        system_prompt: Optional system prompt for chat template.
        timeout: Timeout in seconds for streaming data collection (default: 300).

    Returns:
        A spec string like ``"dataset:split=train:num=100:fields=q+a"``.

    Example::

        >>> build_dataset_spec("my/ds", split="train", num=500, fields=["q", "a"])
        'my/ds:split=train:num=500:fields=q+a'

        >>> build_dataset_spec("my/ds", fields="text")
        'my/ds:fields=text'

        >>> build_dataset_spec("my/ds", template="{question} {answer}", separator=" ")
        'my/ds:template={question} {answer}:separator= '
    """
    parts = [name]

    if split is not None:
        if isinstance(split, list):
            parts.append(f"split={'+'.join(split)}")
        else:
            parts.append(f"split={split}")

    if num is not None:
        parts.append(f"num={num}")

    if concat:
        parts.append("concat=true")

    if template is not None:
        parts.append(f"template={template}")
    elif fields is not None:
        if isinstance(fields, str):
            parts.append(f"fields={fields}")
        else:
            parts.append(f"fields={'+'.join(fields)}")

    # Only include separator if it differs from the default and fields/template is used
    if separator != "\n\n" and (template is not None or fields is not None):
        parts.append(f"separator={separator}")

    if apply_chat_template:
        parts.append("apply_chat_template=true")

    if system_prompt is not None:
        parts.append(f"system_prompt={system_prompt}")

    if timeout != 300:
        parts.append(f"timeout={timeout}")

    return ":".join(parts)


def parse_dataset_spec(spec: str) -> Dict[str, Any]:
    """Parse a dataset spec string into a structured dict.

    The spec string format is::

        dataset_name[:key=value[:key=value[...]]]

    Recognised keys: ``split``, ``num``, ``concat``, ``apply_chat_template``,
    ``system_prompt``, ``fields``, ``template``, ``separator``, ``timeout``.

    Multi-value keys (``split``, ``fields``) use ``+`` as the intra-value
    separator.

    Args:
        spec: The spec string to parse.

    Returns:
        A dict with the following keys (all optional except ``name``):

        - ``name`` (str): Dataset name / path.
        - ``split`` (list[str] | None): Split name(s).
        - ``num`` (int | None): Number of samples.
        - ``concat`` (bool): Whether to concatenate.
        - ``fields`` (str | list[str] | None): Text field(s) to extract.
        - ``template`` (str | None): Template string with ``{field}`` placeholders.
        - ``separator`` (str): Separator for multi-field concatenation.
        - ``apply_chat_template`` (bool): Whether to apply chat template.
        - ``system_prompt`` (str | None): System prompt.

    Example::

        >>> parse_dataset_spec("my/ds:split=train+test:num=100:fields=q+a")
        {'name': 'my/ds', 'split': ['train', 'test'], 'num': 100,
         'concat': False, 'fields': ['q', 'a'],
         'template': None, 'separator': '\\n\\n',
         'apply_chat_template': False, 'system_prompt': None}
    """
    result: Dict[str, Any] = {
        "name": spec,
        "split": None,
        "num": None,
        "concat": False,
        "fields": None,
        "template": None,
        "separator": "\n\n",
        "apply_chat_template": False,
        "system_prompt": None,
        "timeout": 300,
    }

    if ":" not in spec:
        return result

    name, kv_parts = spec.split(":", 1)
    result["name"] = name.strip()

    for part in kv_parts.split(":"):
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        # Don't strip value for separator (whitespace is meaningful)
        if key != "separator":
            value = value.strip()

        if key == "split":
            result["split"] = value.split("+") if value else None
        elif key == "num":
            result["num"] = int(value)
        elif key == "concat":
            result["concat"] = value.lower() != "false"
        elif key == "apply_chat_template":
            result["apply_chat_template"] = value.lower() != "false"
        elif key == "system_prompt":
            result["system_prompt"] = value
        elif key == "fields":
            # Single field: "fields=text" -> "text"
            # Multiple fields: "fields=q+a" -> ["q", "a"]
            if "+" in value:
                result["fields"] = value.split("+")
            else:
                result["fields"] = value
        elif key == "template":
            result["template"] = value
        elif key == "separator":
            # Support escaped newline: \n -> actual newline
            result["separator"] = value.replace("\\n", "\n").replace("\\t", "\t")
        elif key == "timeout":
            result["timeout"] = int(value)
        else:
            # Unknown key – store it for forward compatibility
            result[key] = value

    return result
