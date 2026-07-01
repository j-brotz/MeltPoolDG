#!/usr/bin/env python3

from pathlib import Path
import subprocess
import html


DESCRIPTION_WIDTH = "25em"


def read_description(app_dir: Path) -> str:
    description_file = app_dir / "description.md"

    if not description_file.exists():
        return ""

    return description_file.read_text(encoding="utf-8").strip()


def format_description_cell(value: str) -> str:
    if not value:
        return ""

    blocks = []
    current_list = []

    def flush_list():
        nonlocal current_list
        if current_list:
            items = "".join(f"<li>{item}</li>" for item in current_list)
            blocks.append(f"<ul>{items}</ul>")
            current_list = []

    for raw_line in value.splitlines():
        line = raw_line.strip()

        if not line:
            flush_list()
            blocks.append("<br>")
            continue

        escaped_line = html.escape(line)

        if line.startswith(("- ", "* ")):
            current_list.append(html.escape(line[2:].strip()))
        else:
            flush_list()
            blocks.append(escaped_line)

    flush_list()

    formatted = "<br>".join(
        block for block in blocks if block != "<br>"
    )

    return (
        f'<div style="width: {DESCRIPTION_WIDTH}; white-space: normal;">'
        f"{formatted}"
        f"</div>"
    )


def main():
    repo_root = Path(__file__).resolve().parents[2]
    print(repo_root)

    applications_dir = repo_root / "applications"
    output_dir = repo_root / "doc" / "sphinx" / "source" / "generated" / "parameters"
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_converter = repo_root / "scripts" / "parameters" / "schema_to_md.py"

    apps_with_schema = sorted(
        app_dir
        for app_dir in applications_dir.iterdir()
        if app_dir.is_dir() and (app_dir / "parameters.schema.json").exists()
    )

    index_lines = [
        "# 📦 Applications",
        "",
        "| Application name | Features | Parameters |",
        "|---|---|---|",
    ]

    for app_dir in apps_with_schema:
        app_name = app_dir.name
        schema_file = app_dir / "parameters.schema.json"
        output_file = output_dir / f"{app_name}.md"

        description = format_description_cell(read_description(app_dir))

        subprocess.run(
            [
                "python3",
                str(schema_converter),
                str(schema_file),
                "--output",
                str(output_file),
            ],
            check=True,
        )

        index_lines.append(
            f"| `{app_name}` | {description} | [parameters]({app_name}.md) |"
        )

    (output_dir / "index.md").write_text(
        "\n".join(index_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
