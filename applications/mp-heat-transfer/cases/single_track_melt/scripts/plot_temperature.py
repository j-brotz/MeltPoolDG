#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot temperature at selected coordinates over time from a PyVista-readable PVD file.

Execute with pvpython, for example:

pvpython temperature_at_points.py \
    --folder results/run_01 \
    --points "0,0,0" "50e-6,0,0" "100e-6,0,0" \
    --n 10
"""

import pyvista
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse

pyvista.set_plot_theme("document")

plt.rcParams["figure.figsize"] = (11.69, 8.27)
plt.rcParams["figure.titlesize"] = "small"


def find_filenames(path_to_dir, suffix=".pvd", prefix=""):
    filenames = os.listdir(path_to_dir)
    return [
        filename for filename in filenames
        if filename.endswith(suffix) and filename.startswith(prefix)
    ]


def parse_point(point_string):
    """
    Convert "x,y,z" into [x, y, z].
    Coordinates should be given in simulation units, usually meters.
    """
    values = [float(v.strip()) for v in point_string.split(",")]

    if len(values) == 2:
        values.append(0.0)

    assert len(values) == 3, "Each point must be given as x,y,z or x,y"

    return values


def process_pvd_temperature_points(
    n,
    folder,
    pvdfile,
    points,
    enable_plot=True,
    temperature_array="temperature",
):
    print("-- temperature is evaluated at selected coordinates")
    print(f"-- number of points: {len(points)}")

    reader = pyvista.get_reader(os.path.join(folder, pvdfile))

    time_range = list(range(0, len(reader.time_values), n))

    # make sure last time step is considered
    if len(reader.time_values) - 1 not in time_range:
        time_range.append(len(reader.time_values) - 1)

    points = np.asarray(points, dtype=float)

    time = []
    temperature_history = []

    for i, t in enumerate(time_range):
        reader.set_active_time_point(t)
        mesh = reader.read()[0]

        if temperature_array not in mesh.array_names:
            raise RuntimeError(
                f"Array '{temperature_array}' not found. "
                f"Available arrays are: {mesh.array_names}"
            )

        probe_points = pyvista.PolyData(points)

        sampled = probe_points.sample(
            mesh,
            tolerance=None,
        )

        temperature = sampled["temperature"]

        assert len(points) == len(temperature)

        time.append(reader.time_values[t] * 1e3)  # ms
        temperature_history.append(temperature)

    temperature_history = np.asarray(temperature_history)

    output_csv = os.path.join(folder, "temperature_at_points.csv")

    header_entries = ["time_ms"]
    for i, point in enumerate(points):
        header_entries.append(
            f"T_point_{i}_x={point[0]}_y={point[1]}_z={point[2]}"
        )

    np.savetxt(
        output_csv,
        np.column_stack((time, temperature_history)),
        header=" ".join(header_entries),
    )

    print(f"file written: {output_csv}")

    if enable_plot:
        fig, ax = plt.subplots(1, 1)

        for i, point in enumerate(points):
            label = (
                f"point {i}: "
                f"x={point[0]:.3e}, y={point[1]:.3e}, z={point[2]:.3e}"
            )
            ax.plot(time, temperature_history[:, i], label=label)

        ax.set_xlabel("time (ms)")
        ax.set_ylabel("temperature (K)")
        ax.grid()
        ax.set_xlim([0, time[-1]])
        ax.legend()

        output_png = os.path.join(folder, "temperature_at_points.png")
        fig.savefig(output_png, dpi=1200)

        print(f"file written: {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot temperature at selected coordinates over time. Execute with pvpython!"
    )

    parser.add_argument(
        "--folder",
        type=str,
        help="define the folder where the existing pvd-file is located",
    )

    parser.add_argument(
        "--pvdfile",
        type=str,
        required=False,
        help="define the name of the processed pvd-file, e.g. solution.pvd",
    )

    parser.add_argument(
        "--points",
        type=str,
        nargs="+",
        required=True,
        help='coordinates where temperature is evaluated, e.g. "0,0,0" "50e-6,0,0"',
    )

    parser.add_argument(
        "--temperature-array",
        type=str,
        default="temperature",
        required=False,
        help="name of the temperature array",
    )

    parser.add_argument(
        "--n",
        type=int,
        help="write only every n-th time step",
        default=1,
        required=False,
    )

    parser.add_argument(
        "-y",
        action="store_true",
        help="Set this action to automatically overwrite files.",
        required=False,
    )

    args = parser.parse_args()

    folder = args.folder
    if not folder:
        folder = "."

    folder = os.path.join(os.getcwd(), folder)
    pvdfile = args.pvdfile

    if not pvdfile:
        pvdfile = find_filenames(folder)
        assert len(pvdfile) == 1
        pvdfile = pvdfile[0]

    # read vertical axis from file, consistent with your folder structure
    json_file = find_filenames(folder, ".json")
    assert len(json_file) == 1

    json_file = os.path.join(folder, json_file[0])
    with open(json_file, "r") as f:
        data = json.load(f)

    v_axis = int(data["base"]["dimension"]) - 1

    points = [parse_point(p) for p in args.points]

    print(70 * "-")
    print(f" Vertical axis determined to {v_axis} from {json_file}")
    print(
        " Start processing pvd-file: {:}".format(
            os.path.abspath(os.path.join(folder, pvdfile))
        )
    )
    for p in points:
        print("point", p)

    process_pvd_temperature_points(
        args.n,
        folder,
        pvdfile,
        points,
        temperature_array=args.temperature_array,
    )

    print(" The end")
    print(70 * "-")
