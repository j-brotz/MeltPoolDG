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
 Usage:
 python3 compare_figure_5a.py
"""

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
        description="Plot the comparison figure for fig. 5a in Zahedi (2009).")
    parser.add_argument("-f1", "--folder1", type=str, default="zahedi_wall_wetting_006/",
                        help="Output folder with simulation results for epsilon_tau = 3 epsilon_n")
    parser.add_argument("-f2", "--folder2", type=str, default="zahedi_wall_wetting_007/",
                        help="Output folder with simulation results for epsilon_tau = 12 epsilon_n")
    parser.add_argument("-f3", "--folder3", type=str, default="zahedi_wall_wetting_009/",
                        help="Output folder with simulation results for epsilon_tau = 48 epsilon_n")
    parser.add_argument("-f4", "--folder4", type=str, default="zahedi_wall_wetting_010/",
                        help="Output folder with simulation results for epsilon_tau = 96 epsilon_n")
    parser.add_argument("-f5", "--folder5", type=str, default="zahedi_wall_wetting_011/",
                        help="Output folder with simulation results for epsilon_tau = 192 epsilon_n")
    parser.add_argument("-of", "--output_folders_path", type=str, default="../../meltpooldg_results/",
                        help="Path to output folders")
    parser.add_argument("-zf", "--zahedi_results_folder_path", type=str, default="./zahedi_results_csv_files/",
                        help="Path to Zahedi et al. (2009) results folders")
    parser.add_argument("-ff", "--figures_folder_path", type=str, default="./comparison_figures/",
                        help="Path to folder where to store the saved figure")
    parser.add_argument("-fig", "--figure_name", type=str, default="zahedi_comparison_figure5a.png",
                        help="Filename of the saved figure")
    parser.add_argument("-nz", "--no_zahedi_comparison", type=bool, default=False,
                        help="If set to true, no curves from Zahedi will be added to the figure")
    parser.add_argument("-s", "--show_figure", type=bool, default=False,
                        help="Show figure at the end of the simulation if set to 'True'.")
    args = parser.parse_args()
    plt.close("all")

    # List of data files with Zahedi (2009) simulation results
    zahedi_results_folder = os.path.join(os.getcwd(), f"{args.zahedi_results_folder_path}/fig5/fig5a/")
    zahedi_results_files = ["003_epsilon.csv", "012_epsilon.csv", "048_epsilon.csv", "192_epsilon.csv"]

    # Create figure
    fig1 = plt.figure(constrained_layout=True)
    ax1 = fig1.add_subplot(111)

    # Parameters list
    epsilon_list = [8, 8, 8, 8, 8]
    gamma_list = [2.5, 2.5, 2.5, 2.5, 2.5]
    eta_n_list = [gamma**2 * eps**2 for gamma, eps in zip(gamma_list, epsilon_list)]
    parameter_list = [r"$\left(\eta_n = %.1e,\ \epsilon_n = %dh\right)$" % (eta_n, eps) for eps, eta_n in zip(epsilon_list, eta_n_list)]

    # Set curve properties
    label_list = [r"$\epsilon_\tau = \ \ \ \ 3\epsilon_n$ ", r"$\epsilon_\tau = \ \ 12\epsilon_n$ ",
                  r"$\epsilon_\tau = \ \ 48\epsilon_n$ ", r"$\epsilon_\tau = \ \ 96\epsilon_n$ ", r"$\epsilon_\tau = 192\epsilon_n$ "]
    label_list = [label + " " + parameters for label, parameters in zip(label_list, parameter_list)]
    marker_list = ['.', 'x']
    linewidth_list = [2, 1]
    marker_label_list = ["MeltPoolDG", "Zahedi (2009)"]
    colors_list = sns.color_palette("Paired")

    for i in range(1, 6):
        j = i - 1
        """
         Extract interface position from simulation
        """
        # Identify last outputted VTU file
        exec(f"output_dir = args.output_folders_path+args.folder{i}")
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

        """
         Get Zahedi results from csv files
        """
        # Read csv files
        if (not args.no_zahedi_comparison and i != 4):
            if (i < 4):
                zahedi_df = pd.read_csv(zahedi_results_folder + zahedi_results_files[j])
                zahedi_df.columns = zahedi_df.columns.str.strip()
            else:
                zahedi_df = pd.read_csv(zahedi_results_folder + zahedi_results_files[j - 1])
                zahedi_df.columns = zahedi_df.columns.str.strip()

        """
         Plot data in figure
        """
        ax1.scatter(sim_x, sim_y, s=5, color=colors_list[2 * j], marker=marker_list[0], label=label_list[j],
                    lw=linewidth_list[0])
        if (not args.no_zahedi_comparison and i != 4):
            ax1.scatter(zahedi_df['x'], zahedi_df['y'], s=8, color=colors_list[2 * j + 1], marker=marker_list[1], lw=linewidth_list[1])

    """
     Construct legend
    """
    legend = []
    if (not args.no_zahedi_comparison):
        n_col = 3
        for i in range(0, 5):
            legend.append(mlines.Line2D([], [], color=colors_list[2 * i], label=''))
        for i in range(0, 3):
            legend.append(mlines.Line2D([], [], color=colors_list[2 * i + 1], label=label_list[i]))
        legend.append(mlines.Line2D([], [], c="none", marker="none", ls="none", label=label_list[3]))
        legend.append(mlines.Line2D([], [], color=colors_list[9], label=label_list[4]))
        for i in range(0, 2):
            legend.append(mlines.Line2D([], [], c='k', marker=marker_list[i], ls="none", label=marker_label_list[i],
                                        ms=8, lw=linewidth_list[i]))
        for i in range(0, 3):
            legend.append(mlines.Line2D([], [], c="none", marker="none", ls="none", label=''))
    else:
        n_col = 2
        for i in range(0, 5):
            legend.append(mlines.Line2D([], [], color=colors_list[2 * i], label=label_list[i]))
        legend.append(mlines.Line2D([], [], c='k', marker=marker_list[0], ls="none", label=marker_label_list[0],
                                    ms=8, lw=linewidth_list[0]))
        for i in range(0, 4):
            legend.append(mlines.Line2D([], [], c="none", marker="none", ls="none", label=''))

    if (not args.no_zahedi_comparison):
        plt.xlim(0.99, 1.07)
    else:
        plt.xlim(0.98, 1.12)
    plt.ylim(0, 0.4)
    ax1.tick_params(axis='x', which='both', top=True, direction='in')
    ax1.tick_params(axis='y', which='both', right=True, direction='in')
    ax1.set_axisbelow(False)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(handles=legend, ncol=n_col, loc="upper right")
    plt.savefig(f"{args.figures_folder_path}/{args.figure_name}", bbox_inches='tight')
    if args.show_figure:
        plt.show()
