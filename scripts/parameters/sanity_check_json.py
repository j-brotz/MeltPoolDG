#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: magdalena
"""

##############################################################################
#          USER INPUT
##############################################################################

import sys
import collections.abc
import click
import argparse
import os
from operator import getitem
from functools import reduce
import collections
import json
old_parameter_names = [
    ["laser", "laser center"],
    ["heat", "interpolate rho times cp"],
    ["rte", "problem type"],
    ["rte", "pseudo time stepping", "max n steps"],
    ["rte", "pseudo time stepping", "time step size"],
    ["rte", "verbosity level"],
    ["laser", "laser heat source model"],
    ["laser", "laser impact type"],
    ["laser", "laser power"],
    ["laser", "laser power over time"],
    ["laser", "laser power start time"],
    ["laser", "laser power end time"],
    # and ["laser", "analytical", "absorptivity gas"],
    ["laser", "laser gauss absorptivity gas"],
    # and ["laser", "analytical", "absorptivity liquid"],
    ["laser", "laser gauss absorptivity liquid"],
    ["laser", "laser do move"],
    ["laser", "laser scan speed"],
    # and ["laser", "laser gusarov laser beam radius"],
    ["laser", "laser gauss laser beam radius"],
    ["laser", "laser gusarov reflectivity"],
    ["laser", "laser gusarov extinction coefficient"],
    ["laser", "laser gusarov layer thickness"],
    ["heat", "heat convection coefficient"],
    ["heat", "heat emissivity"],
    ["heat", "heat temperature infinity"],
    ["heat", "heat nlsolve max nonlinear iterations"],
    ["heat", "heat nlsolve field correction tolerance"],
    ["heat", "heat nlsolve residual tolerance"],
    ["heat", "heat nlsolve max nonlinear iterations alt"],
    ["heat", "heat nlsolve field correction tolerance alt"],
    ["heat", "heat nlsolve residual tolerance alt"],
    ["paraview", "paraview do output"],
    ["paraview", "paraview filename"],
    ["paraview", "paraview directory"],
    ["paraview", "paraview write frequency"],
    ["paraview", "paraview write time step size"],
    ["paraview", "paraview print boundary id"],
    ["paraview", "paraview output subdomains"],
    ["paraview", "output material id"],
    ["paraview", "paraview n digits timestep"],
    ["paraview", "paraview n groups"],
    ["paraview", "paraview n patches"],
    ["paraview", "write higher order cells"],
    ["paraview", "output variables"],
    ["paraview", "do user defined postprocessing"],
    # ... add old parameter names
    # ["old", "my age"],
]
new_parameter_names = [
    ["laser", "starting position"],
    ["heat", "use volume-specific thermal capacity for phase interpolation"],
    ["rte", "predictor type"],
    ["rte", "pseudo time stepping", "time stepping", "max n steps"],
    ["rte", "pseudo time stepping", "time stepping", "time step size"],
    ["rte", "rte verbosity level"],
    ["laser", "model"],
    ["laser", "intensity profile"],
    ["laser", "power"],
    ["laser", "power over time"],
    ["laser", "power start time"],
    ["laser", "power end time"],
    ["laser", "absorptivity gas"],
    ["laser", "absorptivity liquid"],
    ["laser", "do move"],
    ["laser", "scan speed"],
    ["laser", "radius"],
    ["laser", "gusarov", "reflectivity"],
    ["laser", "gusarov", "extinction coefficient"],
    ["laser", "gusarov", "layer thickness"],
    ["heat", "convection coefficient"],
    ["heat", "emissivity"],
    ["heat", "temperature infinity"],
    ["heat", "nlsolve", "max nonlinear iterations"],
    ["heat", "nlsolve", "field correction tolerance"],
    ["heat", "nlsolve", "residual tolerance"],
    ["heat", "nlsolve", "max nonlinear iterations alt"],
    ["heat", "nlsolve", "field correction tolerance alt"],
    ["heat", "nlsolve", "residual tolerance alt"],
    ["output", "do output"],
    ["output", "filename"],
    ["output", "directory"],
    ["output", "write frequency"],
    ["output", "write time step size"],
    ["output", "print boundary id"],
    ["output", "output subdomains"],
    ["output", "output material id"],
    ["output", "n digits timestep"],
    ["output", "n groups"],
    ["output", "n patches"],
    ["output", "write higher order cells"],
    ["output", "output variables"],
    ["output", "do user defined postprocessing"],
    # ... add new parameter names
    # ["new", "new", "my new age"],
]

rename_parameter_values = [
    (["heat", "use volume-specific thermal capacity for phase interpolation"],
     "smooth", "true"),
    (["recoil pressure", "interface distributed flux type"],
     "continuous", "local_value"),
    (["laser", "model"],
     "Analytical", "analytical_temperature"),
    (["laser", "model"],
     "interface_projection", "interface_projection_regularized"),
    # add new parameter values to be renamed
    # (["parameter", "category"], "old_value", "new_value"),
]

delete_parameter_names = [
    # ... add parameter names to be deleted
    ["heat", "interpolate k"],
]

##############################################################################
#         END OF USER INPUT
##############################################################################


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_error(message):
    print(f"{bcolors.FAIL}ERROR: " + message + f"{bcolors.ENDC}")


def print_success(message):
    print(f"{bcolors.OKGREEN}" + message + f"{bcolors.ENDC}")


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_nested_value(dataDict, mapList):
    """Get item value in nested dictionary"""
    return reduce(getitem, mapList[:-1], dataDict)[mapList[-1]]


def set_nested_item(dataDict, mapList, val):
    """Set item in nested dictionary"""
    reduce(getitem, mapList[:-1], dataDict)[mapList[-1]] = val
    return dataDict


def delete_nested_item(dataDict, mapList, val):
    """Delete item in nested dictionary"""
    del reduce(getitem, mapList[:-1], dataDict)[mapList[-1]]


def remove_empty_items(d):
    final_dict = {}

    for a, b in d.items():
        if b:
            if isinstance(b, dict):
                final_dict[a] = remove_empty_items(b)
            elif isinstance(b, list):
                final_dict[a] = list(
                    filter(None, [remove_empty_items(i) for i in b]))
            else:
                final_dict[a] = b
    return final_dict


def remove_empty_nested_items(d):
    for i in range(50):  # max 50 nested parameters
        d = remove_empty_items(d)
    return (d)


def create_dict_from_tree_list(tree_list):
    tree_dict = {}
    for key in reversed(tree_list):
        tree_dict = {key: tree_dict}
    return tree_dict


def sanity_check_json(j, old_parameter_names, new_parameter_names, delete_parameter_names, rename_parameter_values, always_yes, write_json, appendix=""):
    assert len(old_parameter_names) == len(new_parameter_names)

    errors = 0

    with open(j, 'r') as f:
        print(70*"-")
        print("Process file: {:}".format(j))
        datastore = json.load(f, object_pairs_hook=collections.OrderedDict)

        val = datastore

        # process parameters to be renamed
        for i, o in enumerate(old_parameter_names):
            # get value of old parameter
            try:
                val = get_nested_value(datastore, o)
                print_error(
                    f"outdated parameter detected {o}; use new definition {new_parameter_names[i]}")
                errors += 1
                if write_json:
                    if always_yes or click.confirm('Do you want me to replace the old parameter by the new one?'):
                        # check if tree of new parameter exists
                        try:
                            get_nested_value(new_parameter_names[i])
                        # if it does not exist, introduce new tree
                        except:
                            print_error("Category does not exist yet")
                            tree_dict = create_dict_from_tree_list(
                                new_parameter_names[i])
                            update(datastore, tree_dict)

                        # replace parameter name and set value from the old parameter
                        set_nested_item(datastore, new_parameter_names[i], val)
                        # delete the outdated parameter pair
                        delete_nested_item(datastore, o, val)
                        print_success(
                            f"RENAME: {o} to {new_parameter_names[i]}")
            except:
                continue

        # process deleted parameters
        for i, o in enumerate(delete_parameter_names):
            try:
                val = get_nested_value(datastore, o)
                print_error(f"non-existing parameter found {o}")
                errors += 1
                if write_json:
                    if always_yes or click.confirm('Do you want me to delete the outdated parameter?'):
                        delete_nested_item(datastore, o, val)
                        print_success("DELETE: parameter '{:}'".format(o))
            except:
                continue

        # process renamed parameter value
        for i, (name, old_val, new_val) in enumerate(rename_parameter_values):
            try:
                val = get_nested_value(datastore, name)
                if (str(val) == str(old_val)):
                    print_error(
                        f"outdated parameter value for {name} detected '{val}'; Use new value '{new_val}'")
                    errors += 1
                    if write_json:
                        if always_yes or click.confirm('Do you want me to replace the old parameter value by the new one?'):
                            # introduce new parameter name and set value from the old parameter
                            set_nested_item(datastore, name, new_val)
                            print(f"    RENAME: {name} {val} to {new_val}")
            except:
                continue

        # delete potentially empty items
        datastore = remove_empty_nested_items(datastore)

    if write_json:
        new_file = j
        if appendix:
            base = os.path.basename(j).split(".json")[0]
            new_file = os.path.join(os.path.dirname(j), base+appendix+".json")

        with open(new_file, 'w') as f:
            print(f"Write file: updated parameters written to {new_file}")
            json.dump(datastore, f, indent=2, separators=(',', ': '))
    else:
        sys.exit(1)
        assert errors == 0, f"ERROR: {errors} errors in the parameter file {j} detected. Use the overwrite option '-w' to fix it."

    print_success(f"Done! {errors} errors fixed.")
    sys.exit(0)


def parseArguments():
    parser = argparse.ArgumentParser(description="Format json file.")

    parser.add_argument('--file', type=str, required=True, help="Input file.")
    parser.add_argument('-nv', action='store_true', help='Set this action to be not verbose.',
                        required=False)
    parser.add_argument('-w', action='store_true', help='Set this action to write files.',
                        required=False)
    parser.add_argument('-y', action='store_true', help='Set this action to automatically overwrite files.',
                        required=False)
    parser.add_argument('--appendix', type=str, required=False,
                        help="Appendix of written output file.")

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parseArguments()

    # block printing
    if args.nv:
        sys.stdout = open(os.devnull, 'w')

    sanity_check_json(args.file, old_parameter_names, new_parameter_names,
                      delete_parameter_names, rename_parameter_values, args.y, args.w, args.appendix)
