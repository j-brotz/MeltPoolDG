#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def remove_excluded_categories(data: Any, excluded_categories: set[str]) -> Any:
    if not isinstance(data, dict):
        return data

    filtered = deepcopy(data)

    for category in excluded_categories:
        filtered.pop(category, None)

    return filtered


def format_error(error: ValidationError) -> str:
    json_path = ".".join(str(part) for part in error.absolute_path)
    schema_path = ".".join(str(part) for part in error.absolute_schema_path)

    if not json_path:
        json_path = "<root>"

    return (
        f"    Path:   {json_path}\n"
        f"    Error:  {error.message}\n"
        f"    Schema: {schema_path}"
    )


def is_ignored_error(error: ValidationError) -> bool:
    """
    Ignore string values such as '0.0,-1.0' when the schema expects an array.

    This allows array-like values written as comma-separated strings.
    """
    return (
        error.validator == "type"
        and error.validator_value == "array"
        and isinstance(error.instance, str)
        and "," in error.instance
    )


def validate_and_print(
    json_file: Path,
    schema_file: Path,
    excluded_categories: set[str],
) -> bool:
    print(f"Checking {json_file} against {schema_file}")

    try:
        errors = validate_file(json_file, schema_file, excluded_categories)
    except json.JSONDecodeError as exc:
        print("  FAILED: invalid JSON")
        print(f"    {exc}")
        return False
    except Exception as exc:
        print("  FAILED: unexpected error")
        print(f"    {exc}")
        return False

    if errors:
        print("  FAILED:")
        for error in errors:
            print(error)
        return False

    print("  OK")
    return True


def validate_file(
    json_file: Path,
    schema_file: Path,
    excluded_categories: set[str],
) -> list[str]:
    schema = load_json(schema_file)
    data = load_json(json_file)

    if isinstance(data, dict):
        data = dict(data)
        data.pop("$schema", None)

    data = remove_excluded_categories(data, excluded_categories)

    validator = Draft202012Validator(schema)

    errors = sorted(
        validator.iter_errors(data),
        key=lambda error: list(error.absolute_path),
    )

    return [
        format_error(error)
        for error in errors
        if not is_ignored_error(error)
    ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one JSON file against one JSON schema."
    )

    parser.add_argument(
        "json_file",
        type=Path,
        nargs="?",
        help="JSON file to validate",
    )

    parser.add_argument(
        "schema_file",
        type=Path,
        nargs="?",
        help="Schema file",
    )
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=[],
        help="Top-level category to ignore. Can be passed multiple times.",
    )

    parser.add_argument(
        "--pairs-file",
        type=Path,
        help="Tab-separated file containing: json_file<TAB>schema_file",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    excluded_categories = set(args.exclude_category)

    pairs: list[tuple[Path, Path]] = []

    if args.pairs_file:
        with args.pairs_file.open("r", encoding="utf-8") as file:
            for line in file:
                json_name, schema_name = line.rstrip("\n").split("\t", 1)
                pairs.append((Path(json_name), Path(schema_name)))
    else:
        if args.json_file is None or args.schema_file is None:
            print("Error: provide json_file schema_file or --pairs-file", file=sys.stderr)
            return 2

        pairs.append((args.json_file, args.schema_file))

    failed_files: list[Path] = []

    for json_file, schema_file in pairs:
        ok = validate_and_print(json_file, schema_file, excluded_categories)
        if not ok:
            failed_files.append(json_file)

    print()
    print(f"Checked {len(pairs)} JSON file(s).")

    if failed_files:
        print()
        print("Failed JSON files:")
        for failed_file in failed_files:
            print(f"  {failed_file}")

        print()
        print(f"{len(failed_files)} file(s) failed validation.")
        return 1

    print("All files passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
