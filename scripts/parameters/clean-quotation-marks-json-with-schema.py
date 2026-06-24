#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def schema_types(schema):
    t = schema.get("type")

    if isinstance(t, list):
        return t

    if isinstance(t, str):
        return [t]

    return []


def convert_value(value, schema):
    types = schema_types(schema)

    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if "string" in types:
        return value

    if "boolean" in types:
        if stripped == "true":
            return True
        if stripped == "false":
            return False

    if "integer" in types:
        try:
            if "." not in stripped and "e" not in stripped.lower():
                return int(stripped)
        except ValueError:
            pass

    if "number" in types:
        try:
            return float(stripped)
        except ValueError:
            pass

    return value


def clean(data, schema):
    types = schema_types(schema)

    if isinstance(data, dict):
        properties = schema.get("properties", {})

        result = {}
        for key, value in data.items():
            sub_schema = properties.get(key, {})

            if sub_schema:
                result[key] = clean(value, sub_schema)
            else:
                result[key] = value

        return result

    if isinstance(data, list):
        item_schema = schema.get("items", {})
        return [clean(item, item_schema) for item in data]

    return convert_value(data, schema)


def main():
    json_file = Path(sys.argv[1])
    schema_file = Path(sys.argv[2])

    old_text = json_file.read_text()

    data = json.loads(old_text)
    schema = json.loads(schema_file.read_text())

    cleaned = clean(data, schema)

    new_text = json.dumps(cleaned, indent=2) + "\n"

    if new_text != old_text:
        json_file.write_text(new_text)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
