#!/bin/bash

for file in **/*.hpp; do
    base="${file%.hpp}"
    if [ ! -f "${base}.cpp" ]; then
        echo "Creating ${base}.cpp"
        touch "${base}.cpp"
    else
        echo "${base}.cpp already exists"
    fi
done
