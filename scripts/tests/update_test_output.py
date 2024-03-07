# -*- coding: utf-8 -*-

import numpy as np
import os
import argparse
import glob
from difflib import SequenceMatcher


def get_arguments():
    parser = argparse.ArgumentParser(description='Execute with python!')
    parser.add_argument('--build-dir', type=str, help='Directory where the build files are located (relative path). \
                                                       Only needed when no ctest log-file is given.',
                        required=True)
    parser.add_argument('--test-root-dir', type=str, help='Root directory to test output files that should be updated',
                        default=os.getcwd(),
                        required=False)
    parser.add_argument('--log-file', type=str, help='Path to the ctest log file (relative path).',
                        default="none",
                        required=False)
    parser.add_argument('--R', type=str, help='Regex for ctest.',
                        default="none",
                        required=False)
    parser.add_argument('--y', action='store_true', help='Set this action to automatically overwrite files.',
                        required=False)
    return vars(parser.parse_args())


def run_tests(build_dir, regex_flags, skip_overwrite):
    print(70 * "-")
    print("Run all tests")
    print(70 * "-")
    cwd = os.getcwd()
    os.chdir(build_dir)

    if os.path.exists("ctest.log") and not skip_overwrite:
        Question = input(
            "Overwrite {:} [Y/N]? ".format(os.path.join(build_dir, "ctest.log")))
        if Question == ("Y"):
            print("")
        else:
            print("Abort...")
            exit()

    print(os.getcwd())
    if (regex_flags != "none"):
        os.system("ctest -R {:} --output-log ctest.log".format(regex_flags))
    else:
        os.system("ctest --output-log ctest.log")
    os.chdir(cwd)
    return os.path.join(build_dir, "ctest.log")


def create_output_path(name):
    folder = name.split("/")[0]
    output_file = name.split(
        "/")[-1].replace(".debug", "").replace(".release", "") + ".output"
    output_file = output_file.replace(".", ".*", 1)
    output_file = output_file.replace(".output", "*.output", 1)
    return os.path.join(folder, output_file)


def copy_test_output(log_file, test_root_dir, build_dir, do_overwrite):
    # Record failed tests
    failed_tests = []
    failed_tests_path = []

    print(70 * "-")
    print("Read log file: {:}".format(os.path.abspath(log_file)))
    print(70 * "-")

    # Collect current test output
    for line in open(log_file, "r").readlines():
        if "(Failed)" in line:
            # remove decoration from test output
            name = line.split(" - ")[-1].split(" (Failed)")[0]
            failed_tests.append(name)
            p = name.split(".")[0] + "." + name.split(".")[-1]

            # check if failing output exists
            if glob.glob(os.path.join(
                    build_dir, "**", p, "**/failing_output"), recursive=True):
                print(
                    "ERROR: the following test >>> {:} <<< failed. Abort ...".format(name))
                exit()

            # locate test output
            output_found = glob.glob(os.path.join(
                build_dir, "**", p, "**/output"), recursive=True)
            if not output_found:
                print(
                    "ERROR: output to the test >>> {:} <<< not found. Abort ...".format(name))
                exit()
            else:
                failed_tests_path.append(output_found)

    # copy new test output to test_root_dir
    print(70 * "-")
    print("Update test output")
    print(70 * "-")
    for f, o in zip(failed_tests, failed_tests_path):
        n = create_output_path(f)
        test_found = glob.glob(os.path.join(
            test_root_dir, "**") + "/" + n, recursive=True)

        if not test_found:
            print(
                "ERROR: output to the test >>> {:} <<< not found. Abort ...".format(n))
            exit()
        elif len(test_found) > 1:
            # based on a similarity measure find corresponding files
            o_idx = []
            for t in test_found:
                similarity = 0.0
                o_idx.append(-1)
                for j, c in enumerate(o):
                    curr = SequenceMatcher(None, t, c).ratio()
                    if (SequenceMatcher(None, t, c).ratio() > similarity):
                        similarity = SequenceMatcher(None, t, c).ratio()
                        o_idx[-1] = j

            # check if list is unique
            assert len(set(o_idx)) == len(o_idx)

            print(
                "multiple outputs to the test >>> {:} <<< found. Copy files:".format(n))

            # copy files
            for i in range(len(test_found)):
                if do_overwrite:
                    copy_command = "yes | cp -rf {:} {:}".format(
                        o[o_idx[i]], test_found[i])
                else:
                    copy_command = "cp {:} {:}".format(
                        o[o_idx[i]], test_found[i])
                print(5 * " " + copy_command)
                os.system(copy_command)

            print("")
        else:
            if do_overwrite:
                copy_command = "yes | cp -rf {:} {:}".format(
                    o[0], test_found[0])
            else:
                copy_command = "cp {:} {:}".format(o[0], test_found[0])
            print(copy_command)
            os.system(copy_command)
            print("")

    print(70 * "-")
    print("{:} test outputs copied!".format(len(failed_tests)))
    print(70 * "-")


if __name__ == '__main__':
    # parse arguments
    a = get_arguments()
    # load ctest log file
    if (a["log_file"] == "none"):
        log_file = run_tests(a["build_dir"], a["R"], a["y"])
    else:
        log_file = a["log_file"]

    assert (log_file != "none")

    # copy test output
    copy_test_output(log_file, a["test_root_dir"], a["build_dir"], a["y"])
