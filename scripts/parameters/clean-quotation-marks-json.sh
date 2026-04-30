#!/usr/bin/env bash
set -e

status=0

# Find all relevant json files
find applications -type f -name "*.json" ! -name "*parameters.schema.json" | while read -r json_file; do

  tmp="$(mktemp)"
  sed -E '
    s/^([[:space:]]*"[^"]+"[[:space:]]*:[[:space:]]*)"[[:space:]]*([+-]?[0-9]+)[.][[:space:]]*"([[:space:]]*,?)[[:space:]]*$/\1\2\3/
    s/^([[:space:]]*"[^"]+"[[:space:]]*:[[:space:]]*)"[[:space:]]*([+-]?[0-9]+[.][0-9]+([eE][+-]?[0-9]+)?|[+-]?[0-9]+([eE][+-]?[0-9]+)?)[[:space:]]*"([[:space:]]*,?)[[:space:]]*$/\1\2\5/
    s/^([[:space:]]*"[^"]+"[[:space:]]*:[[:space:]]*)"[[:space:]]*(true|false)[[:space:]]*"([[:space:]]*,?)[[:space:]]*$/\1\2\3/
  ' "$json_file" > "$tmp"

  # Ensure final newline
  if [ -s "$tmp" ]; then
    last_byte="$(tail -c 1 "$tmp" | od -An -t x1 | tr -d ' \n')"
    if [ "$last_byte" != "0a" ]; then
      printf '\n' >> "$tmp"
    fi
  fi

  if ! cmp -s "$json_file" "$tmp"; then
    mv "$tmp" "$json_file"
    status=1
  else
    rm "$tmp"
  fi

done

exit "$status"
