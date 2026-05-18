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

find applications/ -type f -name "detailed_parameters.output" -print0 |
while IFS= read -r -d '' file; do
  rel_path="${file#applications/}"
  app_name="${rel_path%%/*}"
  app_dir="applications/${app_name}"

  out="${app_dir}/parameters.schema.json"

  if $VERBOSE; then
    echo "Processing: $file -> $out"
  fi

  tmp="$(mktemp)"

  python3 scripts/parameters/dealii_json_to_schema.py "$file" "$tmp"

  if [ ! -f "$out" ] || ! cmp -s "$tmp" "$out"; then
    mv "$tmp" "$out"
  else
    rm -f "$tmp"
  fi
done
