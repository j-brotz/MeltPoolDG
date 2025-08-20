import json
import argparse
from typing import Any, Dict, List


def load_json(filename: str) -> Dict[str, Any]:
    """Load a JSON file into a Python dictionary."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {filename}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON in {filename}: {e}")
        exit(1)


def find_missing_keys(reference: Dict[str, Any], target: Dict[str, Any], prefix: str = '') -> List[str]:
    """Recursively find missing keys from the target JSON compared to the reference."""
    missing = []
    for key in reference:
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in target:
            missing.append(full_key)
        else:
            ref_val = reference[key]
            tgt_val = target[key]
            if isinstance(ref_val, dict) and isinstance(tgt_val, dict):
                missing.extend(find_missing_keys(ref_val, tgt_val, full_key))
    return missing


def main():
    parser = argparse.ArgumentParser(description="Compare two JSON files and list missing parameters."
                                     "This file can be used to compare your json file to the output "
                                     "of case parameters, e.g. "
                                     "python scripts/parameters/compare_input_parameters_with_test_output.py "
                                     "-r my_input_file.json -t "
                                     "applications/mp-melt-pool/cases/unit_tests/melt_pool_case_parameter.with_adaflo_support\\=on.output")
    parser.add_argument('--reference', '-r', required=True, help="Path to the reference JSON file (typically your input file you want to check).")
    parser.add_argument('--target', '-t', required=True, help="Path to the target JSON file to be checked (typically the output of a parameter struct).")
    args = parser.parse_args()

    reference = load_json(args.reference)
    target = load_json(args.target)

    missing_keys = find_missing_keys(reference, target)

    print("\nProtocol of Missing Parameters:")
    if missing_keys:
        for key in missing_keys:
            print(f" - Missing: {key}")
    else:
        print("All parameters found.")


if __name__ == "__main__":
    main()
