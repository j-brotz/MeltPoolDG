#!/usr/bin/env bash
set -euo pipefail

VERBOSE=false

# Parse optional flag
if [[ "${1:-}" == "--verbose" ]]; then
  VERBOSE=true
fi

# force generation of markdown also if schema files didn't change
FORCE_MD=false

# Parse optional flag
if [[ "${1:-}" == "--force_md" ]]; then
  FORCE_MD=true
fi

find . -type f -path "*/mp-*" -name "detailed_parameters.output" -print0 |
while IFS= read -r -d '' file; do
  relative_file_dir=$(dirname "$file")
  absolute_dir=$(realpath "$relative_file_dir")

  # The detailed_parameters.output file is located in the unit_tests/ directory
  # of the application, so we need to go up one level to get to the
  # application directory.
  app_dir="${absolute_dir%/*}"

  out="${app_dir}/parameters.schema.json"

  if $VERBOSE; then
    echo "Processing: $file -> $out"
  fi

  tmp="$(mktemp)"

  SCRIPT_DIR=$(dirname "$0")
  python3 "$SCRIPT_DIR/dealii_json_to_schema.py" "$file" "$tmp"

  if [ ! -f "$out" ] || ! cmp -s "$tmp" "$out"; then
    mv "$tmp" "$out"
  else
    rm -f "$tmp"
  fi
done
