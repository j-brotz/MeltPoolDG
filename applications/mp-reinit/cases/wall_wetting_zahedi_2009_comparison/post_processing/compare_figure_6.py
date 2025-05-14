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
 Usage: python3 compare_figure_6.py [-r <path_to_file_with_simulation_results_summary>]
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
        description="Plot the comparison figure for fig. 6 in Zahedi (2009).")
    parser.add_argument("-r", "--simulation_results_file", type=str, required=True,
                        help="Path to simulation results summary CSV file")
    parser.add_argument("-c", "--cases_csv_file", type=str, default="../case_parameters.csv",
                        help="Path to CSV file with case parameters")
    parser.add_argument("-zf", "--zahedi_results_folder_path", type=str, default="./zahedi_results_csv_files/",
                        help="Path to Zahedi et al. (2009) results folders")
    parser.add_argument("-ff", "--figures_folder_path", type=str, default="./comparison_figures/",
                        help="Path to folder where to store the saved figure")
    parser.add_argument("-fig", "--figure_name", type=str, default="zahedi_comparison_figure6.png",
                        help="Filename of the saved figure")
    parser.add_argument("-s", "--show_figure", type=bool, default=False,
                        help="Show figure at the end of the simulation if set to 'True'.")
    args = parser.parse_args()
    plt.close("all")

    # List of data files with Zahedi (2009) simulation results
    zahedi_results_folder = os.path.join(os.getcwd(), f"{args.zahedi_results_folder_path}/fig6/")
    zahedi_results_files = ["25_angle.csv", "45_angle.csv"]

    # Get path to simulation outputs
    simulation_results_file = ["25_angle.csv", "45_angle.csv"]

    # Create figure
    fig1 = plt.figure(constrained_layout=True)
    ax1 = fig1.add_subplot(111)

    # Set curve properties
    label_list = [r"$\alpha_s = 25$", r"$\alpha_s =45$"]
    marker_list = ['.', 'x']
    linestyle_list = ['-', '-.']
    linewidth_list = [1, 2]
    source_list = ["MeltPoolDG", "Zahedi (2009)"]
    colors_list = sns.color_palette("Paired")

    # Create lists for simulation results
    sim_25 = []
    sim_45 = []

    # Load simulation results and case parameters
    sim_df = pd.read_csv(os.path.join(os.getcwd(), args.simulation_results_file))
    cases_df = pd.read_csv(os.path.join(os.getcwd(), args.cases_csv_file))

    for i, static_contact_angle in enumerate(sim_df["static_contact_angle"]):
        if sim_df['case_number'][i] > 3:
            case_name = f"zahedi_wall_wetting_{sim_df['case_number'][i]:03d}"
            matching_row = cases_df[cases_df["case_name"] == case_name]
            diffusion_ratio = matching_row["epsilon_t_factor"].iloc[0]
            contact_angle_ratio = sim_df["computed_contact_angle"][i] / static_contact_angle
            if static_contact_angle == 25:
                sim_25.append({
                    "diffusion_ratio": diffusion_ratio,
                    "contact_angle_ratio": contact_angle_ratio
                })
            elif static_contact_angle == 45:
                sim_45.append({
                    "diffusion_ratio": diffusion_ratio,
                    "contact_angle_ratio": contact_angle_ratio
                })

    # Build and sort DataFrames
    sim_25_df = pd.DataFrame(sim_25)
    sim_25_df = sim_25_df.sort_values(by="diffusion_ratio", ascending=True)
    sim_45_df = pd.DataFrame(sim_45)
    sim_45_df = sim_45_df.sort_values(by="diffusion_ratio", ascending=True)

    for i in range(0, len(zahedi_results_files)):
        zahedi_df = pd.read_csv(zahedi_results_folder + zahedi_results_files[i])
        zahedi_df.columns = zahedi_df.columns.str.strip()
        # Plot in figure
        ax1.plot(zahedi_df['diffusion_ratio'], zahedi_df['contact_angle_ratio'], lw=linewidth_list[1], color=colors_list[2 * i + 1], ls=linestyle_list[1], marker=marker_list[1], ms=6, markerfacecolor='none')

    """
     Plot simulation data in figure
    """
    ax1.plot(sim_25_df['diffusion_ratio'], sim_25_df['contact_angle_ratio'], lw=linewidth_list[0], color=colors_list[0], ls=linestyle_list[0], label=label_list[0], marker=marker_list[0], ms=6, markerfacecolor='none')
    ax1.plot(sim_45_df['diffusion_ratio'], sim_45_df['contact_angle_ratio'], lw=linewidth_list[0], color=colors_list[2], ls=linestyle_list[0], label=label_list[0], marker=marker_list[0], ms=6, markerfacecolor='none')

    """
     Construct legend
    """
    legend = []
    for i in range(0, 2):
        legend.append(mlines.Line2D([], [], color=colors_list[2 * i], label=''))
    for i in range(0, 2):
        legend.append(mlines.Line2D([], [], color=colors_list[2 * i + 1], label=label_list[i]))
    for i in range(0, 2):
        legend.append(mlines.Line2D([], [], c='k', ls=linestyle_list[i], marker=marker_list[i], label=source_list[i], markerfacecolor="none", ms=6, lw=linewidth_list[i]))

    plt.xscale("log")
    plt.xlim(1e-1, 1e3)
    plt.ylim(1, 2.1)
    ax1.tick_params(axis='x', which='both', top=True, direction='in')
    ax1.tick_params(axis='y', which='both', right=True, direction='in')
    ax1.set_axisbelow(False)
    plt.xlabel(r"Diffusion coefficient ratio $(\epsilon_\tau/\epsilon_n)$ [-]")
    plt.ylabel(r"Contact angle ratio ($\alpha/\alpha_s$) [-]")
    plt.legend(handles=legend, ncol=3)
    plt.savefig(f"{args.figures_folder_path}/{args.figure_name}", bbox_inches='tight')
    if args.show_figure:
        plt.show()
