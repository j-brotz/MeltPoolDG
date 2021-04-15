from argparse import ArgumentParser

import collections
import json

def parseArguments():
    parser = ArgumentParser(description="Format json file.")

    parser.add_argument('file_in' , help="Input file.")
    parser.add_argument('file_out' , help="Output file.")

    arguments = parser.parse_args()
    return arguments

def main():
  args = parseArguments()

  with open(args.file_in, 'r') as f:
    datastore = json.load(f, object_pairs_hook=collections.OrderedDict)

  with open(args.file_out, 'w') as f:
    json.dump(datastore, f, indent=2, separators=(',', ': '))

if __name__== "__main__":
  main()
