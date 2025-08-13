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
 Usage: python3 compare_figure_3.py
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
        description="Plot the comparison figure for fig. 3 in Zahedi (2009).")
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
    parser.add_argument("-fig", "--figure_name", type=str, default="zahedi_comparison_figure3.png",
                        help="Filename of the saved figure")
    parser.add_argument("-s", "--show_figure", type=bool, default=False,
                        help="Show figure at the end of the simulation if set to 'True'.")
    args = parser.parse_args()
    plt.close("all")

    # List of data files with Zahedi (2009) simulation results
    zahedi_results_folder = os.path.join(os.getcwd(), f"{args.zahedi_results_folder_path}/fig3/")
    zahedi_results_files = ["01E+0_epsilon.csv", "25E-1_epsilon.csv", "05E+0_epsilon.csv", "10E+0_epsilon.csv"]

    # Create figure
    fig1 = plt.figure(constrained_layout=True)
    ax1 = fig1.add_subplot(111)

    # Parameters list
    epsilon_list = [8, 8, 8, 8]
    gamma_list = [1, 2.5, 5, 10]
    eta_n_list = [gamma**2 * eps**2 for gamma, eps in zip(gamma_list, epsilon_list)]
    parameter_list = [f"$(\\eta_n = {eta_n:.1e}, \\epsilon_n = {eps}h)$" for eps, eta_n in zip(epsilon_list, eta_n_list)]

    # Set curve properties
    label_list = [r"$\gamma = \ \ \ 1\epsilon_\text{n}$", r"$\gamma = 2.5\epsilon_\text{n}$", r"$\gamma = \ \ \ 5\epsilon_\text{n}$", r"$\gamma = \ 10\epsilon_\text{n}$"]
    label_list = [label + " " + parameters for label, parameters in zip(label_list, parameter_list)]
    marker_list = ['.', '']
    linestyle_list = ['-', '-.']
    linewidth_list = [1, 2]
    source_list = ["MeltPoolDG", "Zahedi (2009)"]
    colors_list = sns.color_palette("Paired")

    for i in range(1, 5):
        j = i - 1
        """
         Get simulation results
        """
        exec(f"output_dir = args.output_folders_path+args.folder{i}")
        output_dir = os.path.join(os.getcwd(), output_dir)
        files_list = os.listdir(output_dir)
        contact_angle_file = [(output_dir + "/" + x) for x in files_list if (".txt" in x)][0]

        # Read contact angle evolution file
        sim_time, sim_contact_angle = np.loadtxt(contact_angle_file, skiprows=1, unpack=True)

        """
         Get Zahedi results from csv files
        """
        # Read csv files
        zahedi_df = pd.read_csv(zahedi_results_folder + zahedi_results_files[j])
        zahedi_df.columns = zahedi_df.columns.str.strip()

        """
         Plot data in figure
        """
        ax1.plot(sim_time, sim_contact_angle, lw=linewidth_list[0], color=colors_list[2 * j], ls=linestyle_list[0], label=label_list[j], marker=marker_list[0], ms=6, markerfacecolor='none')
        ax1.plot(zahedi_df['time'], zahedi_df['contact_angle'], lw=linewidth_list[1], color=colors_list[2 * j + 1], ls=linestyle_list[1])

    x_min = zahedi_df['time'].iloc[0]
    x_max = zahedi_df['time'].iloc[-1]

    ax1.plot([x_min, x_max], [45, 45], color='grey', lw=2, ls="--")

    """
     Construct legend
    """
    legend = []
    for i in range(0, 4):
        legend.append(mlines.Line2D([], [], color=colors_list[2 * i], label=''))
    for i in range(0, 4):
        legend.append(mlines.Line2D([], [], color=colors_list[2 * i + 1], label=label_list[i]))
    for i in range(0, 2):
        legend.append(mlines.Line2D([], [], c='k', ls=linestyle_list[i], marker=marker_list[i], label=source_list[i], markerfacecolor="none", ms=6, lw=linewidth_list[i]))
    # for i in range(0,2):
    legend.append(mlines.Line2D([], [], color="grey", lw=2, ls="--", label='static contact angle'))
    legend.append(mlines.Line2D([], [], c="none", marker="none", ls="none", label=''))

    plt.xscale("log")
    plt.xlim(x_min, x_max)
    plt.ylim(44, 85)
    ax1.tick_params(axis='x', which='both', top=True, direction='in')
    ax1.tick_params(axis='y', which='both', right=True, direction='in')
    ax1.set_axisbelow(False)
    plt.xlabel("Time [s]")
    plt.ylabel(r"Contact angle ($\alpha$) [deg]")
    plt.legend(handles=legend, ncol=3)
    plt.savefig(f"{args.figures_folder_path}/{args.figure_name}", bbox_inches='tight')
    if args.show_figure:
        plt.show()
