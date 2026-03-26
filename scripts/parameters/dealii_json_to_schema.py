#!/usr/bin/env python3
"""
Generate a JSON Schema from a deal.II-style JSON parameter export.

Usage
-----
python dealii_json_to_schema.py input.json output.schema.json

Notes
-----
- Assumes leaf parameter entries have keys like:
    value, default_value, documentation, pattern, pattern_description, actions
- Nested subsections are turned into JSON Schema objects.
- The produced schema is suitable for VS Code JSON completion/validation.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

verbose = False

LEAF_KEYS = {
    "value",
    "default_value",
    "documentation",
    "pattern",
    "pattern_description",
    "actions",
}


INT32_MIN = -2147483648
INT32_MAX = 2147483647


def is_leaf_parameter(node: Any) -> bool:
    return isinstance(node, dict) and "pattern_description" in node and "value" in node


def parse_default_value(raw: Any, schema_type: str | None, enum_values: List[str] | None = None) -> Any:
    if raw is None:
        return None

    if not isinstance(raw, str):
        return raw

    value = raw.strip()

    if schema_type == "boolean":
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

    if schema_type == "integer":
        try:
            return int(value)
        except ValueError:
            return raw

    if schema_type == "number":
        try:
            if value == "MAX_DOUBLE":
                return None
            if value == "-MAX_DOUBLE":
                return None
            return float(value)
        except ValueError:
            return raw

    if schema_type == "array":
        # deal.II often stores list-like defaults as plain strings.
        # If it is a simple comma-separated list, convert; otherwise keep raw.
        if value == "":
            return []
        if value == "all":
            return ["all"]
        return [part.strip() for part in value.split(",")]

    if schema_type == "string":
        return value

    if enum_values is not None and value in enum_values:
        return value

    return value


def parse_selection(pattern_description: str) -> List[str] | None:
    # Example:
    # [Selection none|SUPG ]
    # [Selection not_initialized|FE_Q|FE_SimplexP|FE_Q_iso_Q1|FE_DGQ ]
    m = re.fullmatch(r"\[Selection\s+(.*?)\s*\]", pattern_description)
    if not m:
        return None

    content = m.group(1).strip()
    if not content:
        return []

    return [item.strip() for item in content.split("|") if item.strip()]


def parse_integer_range(pattern_description: str) -> Tuple[int | None, int | None]:
    # Example:
    # [Integer range -2147483648...2147483647 (inclusive)]
    m = re.fullmatch(
        r"\[Integer range\s+(-?\d+)\.\.\.(-?\d+)\s+\(inclusive\)\]",
        pattern_description,
    )
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def parse_double_range(pattern_description: str) -> Tuple[float | None, float | None]:
    # Example:
    # [Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]
    # [Double -1...1 (inclusive)]  # supported too
    m = re.fullmatch(
        r"\[Double\s+([^\.\s][^ ]*?)\.\.\.([^\s]+)\s+\(inclusive\)\]",
        pattern_description,
    )
    if not m:
        return None, None

    def convert(token: str) -> float | None:
        token = token.strip()
        if token == "MAX_DOUBLE":
            return None
        if token == "-MAX_DOUBLE":
            return None
        try:
            return float(token)
        except ValueError:
            return None

    return convert(m.group(1)), convert(m.group(2))


def schema_from_pattern_description(pattern_description: str) -> Dict[str, Any]:
    pattern_description = pattern_description.strip()
    schema: Dict[str, Any] = {}

    enum_values = parse_selection(pattern_description)
    if enum_values is not None:
        schema["type"] = "string"
        schema["enum"] = enum_values

    elif pattern_description == "[Bool]":
        schema["type"] = "boolean"

    elif pattern_description == "[Integer]":
        schema["type"] = "integer"

    elif pattern_description.startswith("[Integer range "):
        schema["type"] = "integer"
        minimum, maximum = parse_integer_range(pattern_description)
        if minimum is not None:
            schema["minimum"] = minimum
        if maximum is not None:
            schema["maximum"] = maximum

    elif pattern_description.startswith("[Double "):
        schema["type"] = "number"
        minimum, maximum = parse_double_range(pattern_description)
        if minimum is not None:
            schema["minimum"] = minimum
        if maximum is not None:
            schema["maximum"] = maximum

    elif pattern_description == "[Anything]":
        schema["type"] = "string"

    elif pattern_description.startswith("[List of <"):
        schema["type"] = "array"

        inner_match = re.match(r"\[List of <(.*?)>", pattern_description)
        if inner_match:
            inner_pattern = inner_match.group(1).strip()
            item_schema = schema_from_pattern_description(inner_pattern)
            schema["items"] = item_schema if item_schema else {"type": "string"}
        else:
            schema["items"] = {"type": "string"}

        m = re.search(r"length\s+(\d+)\.\.\.(\d+)\s+\(inclusive\)", pattern_description)
        if m:
            schema["minItems"] = int(m.group(1))
            max_items = int(m.group(2))
            if max_items < 10**9:
                schema["maxItems"] = max_items

    else:
        schema["anyOf"] = [
            {"type": "string"},
            {"type": "number"},
            {"type": "integer"},
            {"type": "boolean"},
        ]

    return schema


def schema_for_leaf(param_name: str, node: Dict[str, Any]) -> Dict[str, Any]:
    pattern_description = node.get("pattern_description", "").strip()
    documentation = node.get("documentation", "").strip()
    default_raw = node.get("default_value")

    schema = schema_from_pattern_description(pattern_description)

    enum_values = schema.get("enum")

    if documentation:
        schema["description"] = documentation

    default_value = parse_default_value(default_raw, schema.get("type"), enum_values)
    if default_value is not None:
        schema["default"] = default_value

    return schema


def schema_for_node(name: str, node: Any) -> Dict[str, Any]:
    if is_leaf_parameter(node):
        return schema_for_leaf(name, node)

    if not isinstance(node, dict):
        return {}

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for child_name, child_node in node.items():
        properties[child_name] = schema_for_node(child_name, child_node)
        # Usually do not force all parameters to be required, since deal.II has defaults.
        # If you want strict user input later, you can change this behavior.
        # required.append(child_name)

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }

    if required:
        schema["required"] = required

    return schema


def build_root_schema(data: Dict[str, Any], title: str = "deal.II Input Schema") -> Dict[str, Any]:
    root = schema_for_node("root", data)
    root["$schema"] = "http://json-schema.org/draft-07/schema#"
    root["title"] = title
    return root


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("Usage: python dealii_json_to_schema.py input.json [output.schema.json]", file=sys.stderr)
        return 1

    input_path = Path(sys.argv[1])
    if len(sys.argv) == 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.with_suffix(".schema.json")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    schema = build_root_schema(data, title=input_path.stem)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if verbose:
        print(f"Wrote schema to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
