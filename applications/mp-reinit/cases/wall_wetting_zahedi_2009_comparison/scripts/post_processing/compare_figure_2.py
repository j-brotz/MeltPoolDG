#!/usr/bin/env python3

"""
 Import libraries
"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import argparse
import pandas as pd
import pyvista as pv
import seaborn as sns

"""
 Usage: python3 compare_figure_2.py
"""

pv.OFF_SCREEN = True

"""
 Set figure properties
"""
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["figure.dpi"] = 300

"""
 Main
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the comparison figure for fig. 2 in Zahedi (2009).")
    parser.add_argument("-f1", "--folder1", type=str, default="zahedi_wall_wetting_000/",
                        help="Output folder with simulation results for gamma = epsilon_n")
    parser.add_argument("-f2", "--folder2", type=str, default="zahedi_wall_wetting_001/",
                        help="Output folder with simulation results for gamma = 2.5 epsilon_n")
    parser.add_argument("-f3", "--folder3", type=str, default="zahedi_wall_wetting_002/",
                        help="Output folder with simulation results for gamma = 5 epsilon_n")
    parser.add_argument("-f4", "--folder4", type=str, default="zahedi_wall_wetting_003/",
                        help="Output folder with simulation results for gamma = 10 epsilon_n")
    parser.add_argument("-of", "--output_folders_path", type=str, default="../../meltpooldg_results/",
                        help="Path to output folders")
    parser.add_argument("-zf", "--zahedi_results_folder_path", type=str, default="./zahedi_results_csv_files/",
                        help="Path to Zahedi et al. (2009) results folders")
    parser.add_argument("-ff", "--figures_folder_path", type=str, default="./comparison_figures/",
                        help="Path to folder where to store the saved figure")
    parser.add_argument("-fig", "--figure_name", type=str, default="zahedi_comparison_figure2.png",
                        help="Filename of the saved figure")
    parser.add_argument("-c", "--close_up", type=bool, default=False,
                        help="Produce a figure that is a close up near the wall/interface intersection.")
    parser.add_argument("-n", "--show_normal_vectors", type=bool, default=False,
                        help="Produce a figure that has the normal vectors at the interface.")
    parser.add_argument("-s", "--show_figure", type=bool, default=False,
                        help="Show figure at the end of the simulation if set to 'True'.")
    args = parser.parse_args()
    plt.close("all")

    # List of data files with Zahedi (2009) simulation results
    zahedi_results_folder = os.path.join(os.getcwd(), f"{args.zahedi_results_folder_path}/fig2/")
    zahedi_results_files = ["01E+0_epsilon.csv", "25E-1_epsilon.csv", "05E+0_epsilon.csv", "10E+0_epsilon.csv"]

    # Create figure
    fig1 = plt.figure(constrained_layout=True)
    ax1 = fig1.add_subplot(111)

    # Parameters list
    epsilon_list = [8, 8, 8, 8]
    gamma_list = [1, 2.5, 5, 10]
    eta_n_list = [gamma**2 * eps**2 for gamma, eps in zip(gamma_list, epsilon_list)]
    parameter_list = [r"$\left(\eta_\text{n} = %.1e,\ \epsilon_\text{n} = %dh\right)$" % (eta_n, eps) for eps, eta_n in zip(epsilon_list, eta_n_list)]

    # Set curve properties
    label_list = [r"$\gamma = \ \ \ 1\epsilon_\text{n}$", r"$\gamma = 2.5\epsilon_\text{n}$", r"$\gamma = \ \ \ 5\epsilon_\text{n}$", r"$\gamma = \ 10\epsilon_\text{n}$"]
    label_list = [label + " " + parameters for label, parameters in zip(label_list, parameter_list)]
    marker_list = ['.', 'x']
    linewidth_list = [2, 1]
    source_list = ["MeltPoolDG", "Zahedi et al. (2009)"]
    colors_list = sns.color_palette("Paired")

    # Define figure range
    if not args.close_up:
        x_min, x_max = 0.98, 1.1
        y_min, y_max = 0, 1.6
        arrow_scale = 1000
    else:
        x_min, x_max = 0.995, 1.09
        y_min, y_max = 0, 0.1
        arrow_scale = 500

    for i in range(1, 5):
        j = i - 1
        """
         Extract interface position from simulation
        """
        # Identify last outputted VTU file
        exec(f"folder = args.folder{i}")
        output_dir = args.output_folders_path + folder
        output_dir = os.path.join(os.getcwd(), output_dir)
        list_vtu = os.listdir(output_dir)
        list_vtu = [(output_dir + "/" + x) for x in list_vtu if ("vtu" in x)]
        last_vtu_file = max(list_vtu, key=os.path.getctime)

        # Read VTU file
        sim_df = pv.read(last_vtu_file)

        # Extract interface contour coordinates
        sim_df.set_active_scalars("psi")
        interface_psi_val = np.array([0.0])
        interface = sim_df.contour(interface_psi_val)
        sim_x, sim_y = interface.points[:, 0], interface.points[:, 1]

        if args.show_normal_vectors:
            # Filter interface points within the desired bounding box (for computation speedup)
            mask = (
                (sim_x >= x_min) & (sim_x <= x_max) &
                (sim_y >= y_min) & (sim_y <= y_max)
            )
            if not np.any(mask):
                print(f"No interface points in bounding box for folder {folder}")
                continue

            # Keep only filtered points and interpolate normals
            interface = interface.extract_points(mask)
            sim_x, sim_y = interface.points[:, 0], interface.points[:, 1]

            # Get normal points at interface
            normal_0_interface = interface.interpolate(sim_df)["normal_0"]
            normal_1_interface = interface.interpolate(sim_df)["normal_1"]

        """
         Get Zahedi results from csv files
        """
        # Read csv files
        zahedi_df = pd.read_csv(zahedi_results_folder + zahedi_results_files[j])
        zahedi_df.columns = zahedi_df.columns.str.strip()

        """
         Plot data in figure
        """
        ax1.scatter(sim_x, sim_y, s=5, color=colors_list[2 * j], marker=marker_list[0], label=label_list[j], lw=linewidth_list[0])
        if args.show_normal_vectors:
            plt.quiver(sim_x, sim_y, normal_0_interface, normal_1_interface, angles='xy', scale_units='xy', scale=arrow_scale, color=colors_list[2 * j], width=0.001)
        ax1.scatter(zahedi_df['x'], zahedi_df['y'], s=8, color=colors_list[2 * j + 1], marker=marker_list[1], lw=linewidth_list[1])

    """
     Construct legend
    """
    legend = []
    for i in range(0, 4):
        legend.append(mlines.Line2D([], [], color=colors_list[2 * i], label=''))
    for i in range(0, 4):
        legend.append(mlines.Line2D([], [], color=colors_list[2 * i + 1], label=label_list[i]))
    for i in range(0, 2):
        legend.append(mlines.Line2D([], [], c='k', marker=marker_list[i], ls="none", label=source_list[i], ms=8, lw=linewidth_list[i]))
    for i in range(0, 2):
        legend.append(mlines.Line2D([], [], c="none", marker="none", ls="none", label=''))

    if args.show_normal_vectors:
        name, ext = os.path.splitext(args.figure_name)
        args.figure_name = f"{name}_normal_vector{ext}"
    if args.close_up:
        name, ext = os.path.splitext(args.figure_name)
        args.figure_name = f"{name}_closeup{ext}"
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    ax1.tick_params(axis='x', which='both', top=True, direction='in')
    ax1.tick_params(axis='y', which='both', right=True, direction='in')
    ax1.set_axisbelow(False)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(handles=legend, ncol=3)
    plt.savefig(f"{args.figures_folder_path}/{args.figure_name}", bbox_inches='tight')
    if args.show_figure:
        plt.show()
