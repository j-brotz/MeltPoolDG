# Generating and Launching Cases

Scripts used in the different studies are separated in two main folders, namely `pre_processing` and `post_processing`.

```
wall_wetting_zahedi_2009_comparison/scripts/
├── post_processing
│   ├── compare_figure_2.py
│   ├── compare_figure_3.py
│   ├── compare_figure_4.py
│   ├── compare_figure_5a.py
│   ├── compare_figure_5b.py
│   ├── compare_figure_6.py
│   └── post_process_cases.sh
├── pre_processing
│   ├── case_parameters.csv
│   ├── generate_cases.sh
│   ├── launch_cases.sh
│   └── zahedi_wall_wetting.tpl
└── README.md
```


1. The `pre_processing` folder contains scripts and files necessary to create and launch cases:
   - Bash script to generate cases: `generate_cases.sh`
   - Bash script to launch cases: `launch_cases.sh`
   - CSV file containing a list of cases and their parameters: `case_parameters.csv`
   - Template JSON parameter file: `zahedi_wall_wetting.tpl`

2. The `post_processing` folder contains scripts and files necessary to post-process the simulation results and generate comparison figures.
   - Bash script to post-process and generate all comparison figures: `post_process_cases.sh`
   - Folder with results from Zahedi *et al.* [^1]\: `zahedi_results_csv_files/`. The folder  contains results extracted from the figures 2-6 of Zahedi *et al.* [^1]. The data is stored in CSV format in subfolders for each figure.
   - Post-processing python scripts that generate comparison figures:
     - `compare_figure_2.py`: Generates the comparison figure of the final time-step interface contours for different $\eta_\text{n}$ values.
     - `compare_figure_3.py`: Generates the comparison figure with the time evolution of the contact angle for different $\eta_\text{n}$ values.
     - `compare_figure_4.py`: Generates the comparison figure with the time evolution of the contact angle for different $\varepsilon_\text{tau}$ values.
     - `compare_figure_5a.py`: Generates the comparison figure of the final time-step interface contours for different $\varepsilon_\text{tau}$ values.
     - `compare_figure_5b.py`: Generates a close up of figure 5a near the contact point region.
     - `compare_figure_6.py`: Generates the comparison figure of the contact angle ratio $(\frac{\alpha}{\alpha_\text{s}})$ as a function of the diffusion coefficient ratio $(\frac{\varepsilon_\text{t}}{\varepsilon_\text{n}})$.


## Generating the cases

In the `pre_processing/` folder, cases are generated in using the `generate_cases.sh` bash script with the CSV case parameter file (`case_parameters.csv`) and the template parameter file (`zahedi_wall_wetting.tpl`).

The CSV case parameter file takes the following form:
```csv
case_name,static_contact_angle,epsilon_n_factor,epsilon_t_factor,gamma_factor,end_time
zahedi_wall_wetting_000,45,8,6,1,0.05
zahedi_wall_wetting_001,45,8,6,2.5,0.05
zahedi_wall_wetting_002,45,8,6,5,0.05
...
```

Note that the `case_name` contains the padded case number, i.e. "zahedi_wall_wetting_000" is associated with case #0 and "zahedi_wall_wetting_001" with case #1, and so on.

Cases 0 to 3 correspond to the first study, and cases 4 to 20 correspond to the second study.

One can generate the different cases JSON files by simply running the following command:

```bash
./generate_cases.sh
```

by default, this will generate JSON files in a folder `wall_wetting_zahedi_2009_comparison/parameter_files/` for all cases ranging from #0 to #20.

You can specify a different range of cases using the `-c` argument.
For example, running
```bash
./generate_cases.sh -c "0-3, 20"
```

will generate JSON files for cases number 0, 1, 2, 3, and 20.
For more information on the different arguments that can be passed on, run:

```bash
./generate_cases.sh --help
```

## Running the cases

Calling the `launch_cases.sh` script in the `pre_processing/` folder, as shown below, runs by default simulations for cases 0 to 20 on 20 processes.

```bash
./launch_cases.sh
```

Similar to `generate_cases.sh`, one can specify which cases to run using the `-c` argument.
The number of processes can also be changed using the `-np` argument.
For more information on the other arguments, use `-h` or `--help`.

When `launch_cases.sh` is called, it:
- runs the specified cases;
- logs the console (terminal) outputs in the `wall_wetting_zahedi_2009_comparison/meltpooldg_results/logs/` folder;
- stores simulation outputs such as VTU files and contact angle evolution files (`wall_wetting.txt`) in subfolders of `wall_wetting_zahedi_2009_comparison/meltpooldg_results/` for each case, and;
- creates a summary file named with following syntax `contact_angles_summary_${timestamp}.csv` (e.g. `contact_angles_summary_20250530_144304.csv`); the file contains among other things the `computed_contact_angle` evaluated at $\tau_\text{end}$ of the simulations ran with the script. This file also can be found in the `wall_wetting_zahedi_2009_comparison/meltpooldg_results/` folder.


Here's what the tree of the results folder should look like:
```
wall_wetting_zahedi_2009_comparison/meltpooldg_results
├── comparison_figures
│   ├── zahedi_comparison_figure2_normal_vector_closeup.png
│   ├── zahedi_comparison_figure2.png
│   ├── zahedi_comparison_figure3.png
│   ├── ...
│   └── zahedi_comparison_figure6.png
├── contact_angles_summary_${time_stamp}.csv
├── logs
│   ├── zahedi_wall_wetting_000.txt
│   ├── zahedi_wall_wetting_001.txt
│   ├── ...
│   └── zahedi_wall_wetting_020.txt
├── zahedi_wall_wetting_000
│   ├── wall_wetting_0000.vtu
│   ├── wall_wetting_0001.vtu
│   ├── ...
│   ├── wall_wetting.pvd
│   ├── wall_wetting.txt
│   └── zahedi_wall_wetting_000.json
├── zahedi_wall_wetting_001
│   ├── ...
├── ...
└── zahedi_wall_wetting_020
    └── ...
```

## Post-processing Simulation Results
From the `post_processing` folder, the `post_process_cases.sh` script can be run to generate all figures automatically using:

```bash
./post_process_cases.sh -r <path_to_file_with_simulation_results_summary>
```

Calling this script also by default generates a `comparison_figures/` folder in the `meltpooldg_results/` directory where all figures will be saved. To change this, one may pass a different argument with the `-ff` or `--figures_folder` flags.
For more information on the other arguments, use `-h` or `--help`.


## Reference

[^1]: S. Zahedi, K. Gustavsson, and G. Kreiss, “A conservative level set method for contact line dynamics,” *J. Comput. Phys.*, vol. 228, no. 17, pp. 6361–6375, Sep. 2009, doi: 10.1016/j.jcp.2009.05.043.
