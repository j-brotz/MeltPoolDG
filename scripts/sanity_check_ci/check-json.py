#!/usr/bin/env python3

import collections
import json
import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python3 " + sys.argv[0] +
              " <key path> <due value> <json>")
        print("  the <key path> must be a single sting with comma separated key names, e.g. 'output,paraview,enable'")
        sys.exit(1)
    with open(sys.argv[3], "r") as f:
        data = json.load(f, object_pairs_hook=collections.OrderedDict)
        key_path = sys.argv[1].split(",")
        for key in key_path:
            if key in data:
                data = data[key]
            else:
                sys.exit()
        if data != sys.argv[2]:
            print(sys.argv[3])
