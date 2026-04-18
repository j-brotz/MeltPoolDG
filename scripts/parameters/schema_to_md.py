import json
import argparse
import re
from pathlib import Path

verbose = False


def format_enum_description(prop):
    description = prop.get("description", "").strip()
    enum_values = prop.get("enum", [])

    if not enum_values:
        return description

    lines = []
    if description:
        lines.append(description)
        lines.append("")
        lines.append("Allowed values:")

    for value in enum_values:
        lines.append(f"- `{value}`")

    return "<br>".join(lines)


def make_anchor(path_parts):
    text = "-".join(path_parts).strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9\-]", "", text)
    return text


def prettify_name(name):
    return name.replace("_", " ")


def format_default(prop):
    if "default" not in prop:
        return ""
    value = prop["default"]

    if isinstance(value, bool):
        return f"`{str(value)}`"
    return f"`{value}`"


def render_table(schema, path_parts):
    properties = schema.get("properties", {})
    if not properties:
        return ""

    md = []

    md.append("| Parameter | Type | Default | Description |")
    md.append("|---|---|---|---|")

    for name, prop in properties.items():
        param_type = prop.get("type", "N/A")
        description = prop.get("description", "")
        if "enum" in prop:
            description = format_enum_description(prop)
            allowed_values = ""   # remove duplication
        else:
            description = prop.get("description", "")
            allowed_values = ""
        if param_type == "object":
            child_anchor = make_anchor(path_parts + [name])
            name_md = f"[`{name}`](#{child_anchor})"
            link = f"[See table](#{child_anchor})"
            description = f"{description} {link}".strip()
        else:
            name_md = f"`{name}`"

        default = format_default(prop)

        md.append(
            f"| {name_md} | `{param_type}` | {default} | {description} |"
        )

    md.append("")
    return "\n".join(md)


def parse_schema(schema, path_parts, level=2, is_top_level=False):
    md = []
    anchor = make_anchor(path_parts)

    heading_text = ""

    if is_top_level:
        heading_text += "🔷 "

    heading_text += ": ".join(path_parts)

    md.append(f'<a id="{anchor}"></a>')
    md.append(f"{'#' * level} `{heading_text}`\n")

    table = render_table(schema, path_parts)
    if table:
        md.append(table)

    properties = schema.get("properties", {})
    for child_name, child_schema in properties.items():
        if child_schema.get("type") == "object":
            md.append(parse_schema(child_schema, path_parts + [child_name], level + 1))

    if is_top_level:
        md.append("\n---\n")

    return "\n".join(part for part in md if part)


def derive_app_name(schema_path, schema):
    return schema_path.parent.name


def schema_to_markdown(schema, schema_path):
    app_name = derive_app_name(schema_path, schema)
    md = [f"# {app_name}: Parameter description\n"]

    properties = schema.get("properties", {})
    if not properties:
        return "\n".join(md)

    top_level_nodes = []
    has_scalar_root_entries = False

    for name, prop in properties.items():
        if prop.get("type") == "object":
            top_level_nodes.append(name)
        else:
            has_scalar_root_entries = True

    if has_scalar_root_entries:
        top_level_nodes = ["root"]

    # Inhaltsverzeichnis
    md.append("## Contents\n")
    for name in top_level_nodes:
        anchor = make_anchor([name])
        md.append(f"- [`{name}`](#{anchor})")
    md.append("")
    md.append("---")
    md.append("")

    # Inhalt
    if has_scalar_root_entries:
        root_wrapper = {
            "properties": properties,
            "required": schema.get("required", []),
        }
        md.append(parse_schema(root_wrapper, ["root"], level=2, is_top_level=True))
    else:
        for name in top_level_nodes:
            prop = properties[name]
            md.append(parse_schema(prop, [name], level=2, is_top_level=True))

    return "\n".join(md)


def process_schema_file(schema_path, output_path=None):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    markdown = schema_to_markdown(schema, schema_path)

    if output_path is None:
        output_path = schema_path.with_suffix(".md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    if (verbose):
        print(f"Markdown written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSON schema to Markdown")
    parser.add_argument("schema", help="Path to schema.json file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output markdown file (default: same name with .md)",
    )
    args = parser.parse_args()

    schema_path = Path(args.schema)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    output_path = Path(args.output) if args.output else None
    process_schema_file(schema_path, output_path)


if __name__ == "__main__":
    main()
