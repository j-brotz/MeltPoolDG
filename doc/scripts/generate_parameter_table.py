#!/usr/bin/env python3

from pathlib import Path
import subprocess


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
        "| Application name | Parameters |",
        "|---|---|",
    ]

    for app_dir in apps_with_schema:
        app_name = app_dir.name
        schema_file = app_dir / "parameters.schema.json"
        output_file = output_dir / f"{app_name}.md"

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
            f"| `{app_name}` | [parameters]({app_name}.md) |"
        )

    (output_dir / "index.md").write_text(
        "\n".join(index_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
