#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: magdalena
"""

import pyvista
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

import argparse
pyvista.set_plot_theme('document')

plt.rcParams["figure.figsize"] = (11.69, 8.27)
plt.rcParams["figure.titlesize"] = 'small'


def find_filenames(path_to_dir, suffix=".pvd", prefix=""):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix) and filename.startswith(prefix)]


def process_pvd(n, folder, pvdfile, enable_plot=True, vertical_axis=1, dT=5):
    reader = pyvista.get_reader(os.path.join(folder, pvdfile))

    # loop over time steps:
    time_range = list(range(0, len(reader.time_values), n))

    # make sure last time step is considered
    if len(reader.time_values) - 1 not in time_range:
        time_range.append(len(reader.time_values) - 1)

    # create points of base plate
    normal = [0, 0, 0]
    normal[vertical_axis] = 1
    origin = [0, 0, 0]
    origin[vertical_axis] = 1e-16
    print(f"normal direction {normal}")
    print(f"Temperature decrease: {dT}")

    time = []
    nusselt = []
    nusselt_welch = []

    factor = 1. / (2. * np.pi * np.sqrt(3) * dT)

    for i, t in enumerate(time_range):
        reader.set_active_time_point(t)
        mesh = reader.read()[0]

        # evaluate temperature gradient in vertical direction
        mesh["dT_dZ"] = mesh.compute_derivative(scalars="temperature")["gradient"][:, vertical_axis]

        # create horizontal plane
        slice_ = mesh.slice(normal=normal, origin=origin)

        dT_dz = np.abs(slice_.integrate_data()["dT_dZ"][0])

        nusselt.append(factor * dT_dz)  # Nusselt number according to Lee
        nusselt_welch.append(dT_dz / dT)  # Nusselt number according to Welch 2000 (41)
        time.append(reader.time_values[t])

    nusselt_averaged = np.mean(nusselt)
    print(f"Nusselt (space and time averaged): {nusselt_averaged}")

    nusselt_time_averaged = []
    nusselt_time_averaged_welch = []
    for i in range(len(nusselt)):
        nusselt_time_averaged.append(nusselt_averaged)
        nusselt_time_averaged_welch.append(np.mean(nusselt_welch))

    np.savetxt(os.path.join(folder, "averaged_nusselt_number.csv"),
               np.column_stack((time, nusselt, nusselt_time_averaged, nusselt_welch, nusselt_time_averaged_welch)),
               header="time space_averaged_nusselt_lee space_and_time_averaged_nusselt_lee space_averaged_nusselt_welch space_and_time_averaged_nusselt_welch")

#    ###########################################################################
#    # save figure
#    ###########################################################################

    if (enable_plot):
        fig, ax = plt.subplots(1, 1)

        ax.plot(time, nusselt, label="nusselt")
        ax.plot(time, nusselt_time_averaged, label="average")

        ax.grid()
        ax.set_xlim([0, time[-1]])
        ax.legend()
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(r"Nusselt (-)")

        fig.savefig(os.path.join(folder, "nusselt.png"), dpi=1200)
        print("file written: {:}".format(os.path.join(folder, "nusselt.png")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export the level set contour to a file. Execute with pvpython!')
    parser.add_argument('--folder', type=str,
                        help='define the folder, where the existing pvd-file is located')
    parser.add_argument('--pvdfile', type=str, required=False,
                        help='define the name of the processed pvd-file, e.g. solution.pvd')
    parser.add_argument('-n', type=int, help='Write only every n.', default=1,
                        required=False)
    parser.add_argument('-y', action='store_true', help='Set this action to automatically overwrite files.',
                        required=False)
    args = parser.parse_args()

    folder = args.folder
    if not folder:
        folder = "."
    folder = os.path.join(os.getcwd(), folder)
    pvdfile = args.pvdfile

    # find pvd file if none is given
    if not pvdfile:
        pvdfile = find_filenames(folder)
        assert len(pvdfile) == 1
        pvdfile = pvdfile[0]

    # read vertical axis from file
    json_file = find_filenames(folder, '.json')
    assert len(json_file) == 1
    json_file = os.path.join(folder, json_file[0])
    with open(json_file, 'r') as f:
        data = json.load(f)
    v_axis = int(data["base"]["dimension"]) - 1

    # create temp folder
    print(70 * "-")
    print(f" Vertical axis determined to {v_axis} from {json_file}")
    print(
        " Start processing pvd-file: {:}".format(os.path.abspath(os.path.join(folder, pvdfile))))
    process_pvd(args.n, folder, pvdfile, vertical_axis=v_axis)

    print(" The end")
    print(70 * "-")
