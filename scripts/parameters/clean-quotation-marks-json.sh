#!/usr/bin/env bash
set -e

status=0

find applications -type f -name "*.json" ! -name "*parameters.schema.json" | while read -r json_file; do
  app_dir="$(echo "$json_file" | sed -E 's#^(applications/[^/]+).*#\1#')"
  schema_file="${app_dir}/parameters.schema.json"

  echo "Processing JSON file: $json_file"

  if [ ! -f "$schema_file" ]; then
    continue
  fi

  if python3 scripts/parameters/clean-quotation-marks-json-with-schema.py  "$json_file" "$schema_file"; then
    :
  else
    status=1
  fi
done

exit "$status"
