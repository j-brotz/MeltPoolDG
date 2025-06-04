#!/bin/bash

# This scripts post-processes the simulations results and generates specified comparison figures.

USAGE="Usage: $0 [-p <path_to_folder_with_python_scripts>] \n
      \t[-z <path_to_folder_with_zahedi_results>] \n
      \t[-s <\"list_of_python_scripts_to_launch\">] \n
      \t[-f <path_to_folder_where_to_save_resulting_figures>] \n
      \t[-r <path_to_file_with_simulation_results_summary>] \n
      \t[-cf <path_to_file_with_case_parameters>] \n
      \n
       -p:  Path to folder with python scripts (default: ./post_processing/) \n
       -z:  Path to folder with results from Zahedi et al. (2009) (default: ./post_processing/zahedi_results_csv_files/) \n
       -s:  List of python scripts to launch \n
       \t   (default: \"compare_figure_2.py, compare_figure_3.py, compare_figure_4.py, compare_figure_5a.py, compare_figure_5b.py, compare_figure_6.py\") \n
       -f:  Path to folder where figures are stored (default: ./post_processing/comparison_figures/) \n
       -r:  Path to CSV file with simulation results summary (no default value, it must be specified by the user) \n
       -cf: Path to CSV file with case parameters (default: ./case_parameters.csv) \n
       -h:  Show help \n"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -p|--post_processing_folder)
      post_processing_folder="$2"
      shift 2
      ;;
    -z|--zahedi_folder)
      zahedi_folder="$2"
      shift 2
      ;;
    -s|--python_scripts_list)
      python_scripts_list="$2"
      shift 2
      ;;
    -f|--figures_folder)
      figures_folder="$2"
      shift 2
      ;;
    -r|--simulation_results_file)
      simulation_results_file="$2"
      shift 2
      ;;
    -cf|--cases_csv_file)
      cases_csv_file="$2"
      shift 2
      ;;
    -h|--help)
      echo -e "$USAGE"
      exit 0
      ;;
    *)
      echo "Unknown parameter: $1"
      echo -e "$USAGE"
      exit 1
      ;;
  esac
done

# Ensure required argument -r is specified
if [[ -z "$simulation_results_file" ]]; then
  echo "Error: Missing required argument -r <path_to_file_with_simulation_results_summary>"
  echo -e "$USAGE"
  exit 1
fi

# Assign default values if not set
post_processing_folder="${post_processing_folder:-./post_processing/}"
zahedi_folder="${zahedi_folder:-./post_processing/zahedi_results_csv_files/}"
python_scripts_list="${python_scripts_list:-"compare_figure_2.py, compare_figure_3.py, compare_figure_4.py, compare_figure_5a.py, compare_figure_5b.py, compare_figure_6.py"}"
figures_folder="${figures_folder:-./post_processing/comparison_figures/}"
cases_csv_file="${cases_csv_file:-./case_parameters.csv}"

# Check if the folders and files exist
if [ ! -d "$post_processing_folder" ]; then
  echo "Folder not found: $post_processing_folder"
  exit 1
fi
if [ ! -d "$zahedi_folder" ]; then
  echo "Folder not found: $zahedi_folder"
  exit 1
fi
if [ ! -d "$figures_folder" ]; then
  echo  "$figures_folder"
  mkdir -p "$figures_folder" # Create folder if it does not exist
fi
if [ ! -f "$simulation_results_file" ]; then
  echo "File not found: $simulation_results_file"
  exit 1
fi

if [ ! -f "$cases_csv_file" ]; then
  echo "File not found: $cases_csv_file"
  exit 1
fi

# Print values to user
echo "Python scripts folder:                $post_processing_folder"
echo "Zahedi et al. (2009) results folder:  $zahedi_folder"
echo "List of python scripts to call:       $python_scripts_list"
echo "Figures folder:                       $figures_folder"
echo "Simulation summary results file:      $simulation_results_file"
echo "Case parameters CSV file:             $cases_csv_file"

# Convert comma-separated string into array
IFS=',' read -ra python_scripts_array <<< "$python_scripts_list"

echo "********************************************************************************"
echo "********************************************************************************"
echo " GENERATING FIGURES FOR ZAHEDI (2009) COMPARISON"
echo "********************************************************************************"

# Move into post-processing folder
cd $post_processing_folder

# Loop over specified cases
for python_script in ${python_scripts_array[@]}
do
  echo "********************************************************************************"

  # Check if file exists
  if [ ! -f "$python_script" ]; then
    echo "File not found: $python_script"
    exit 1
  fi

  case "$python_script" in
    "compare_figure_2.py")
      echo " Running figure 2 comparison"
      python3 compare_figure_2.py -zf ../$zahedi_folder -ff ../$figures_folder
      python3 compare_figure_2.py -zf ../$zahedi_folder -ff ../$figures_folder -c True -n True # Closeup with interface normal vectors
      ;;
    "compare_figure_3.py")
      echo " Running figure 3 comparison"
      python3 compare_figure_3.py -zf ../$zahedi_folder -ff ../$figures_folder
      ;;
    "compare_figure_4.py")
      echo " Running figure 4 comparison"
      python3 compare_figure_4.py -zf ../$zahedi_folder -ff ../$figures_folder
      ;;
    "compare_figure_5a.py")
      echo " Running figure 5a comparison"
      # 45 degree contact angle
      python3 compare_figure_5a.py -zf ../$zahedi_folder -ff ../$figures_folder
      # 25 degree contact angle
      python3 compare_figure_5a.py -f1 zahedi_wall_wetting_014/ -f2 zahedi_wall_wetting_016/ \
      -f3 zahedi_wall_wetting_018/ -f4 zahedi_wall_wetting_019/ -f5 zahedi_wall_wetting_020/\
      -zf ../$zahedi_folder -ff ../$figures_folder \
      -fig zahedi_figure5a_25deg.png -nz True
      ;;
    "compare_figure_5b.py")
      echo " Running figure 5b comparison"
      # 45 degree static contact angle
      python3 compare_figure_5b.py -zf ../$zahedi_folder -ff ../$figures_folder
      python3 compare_figure_5b.py -zf ../$zahedi_folder -ff ../$figures_folder -n True # With interface normal vectors
      # 25 degree static contact angle
      python3 compare_figure_5b.py -f1 zahedi_wall_wetting_014/ -f2 zahedi_wall_wetting_016/ \
      -f3 zahedi_wall_wetting_018/ -f4 zahedi_wall_wetting_019/ -f5 zahedi_wall_wetting_020/ \
      -zf ../$zahedi_folder -ff ../$figures_folder \
      -fig zahedi_figure5b_25deg.png -nz True -n True
      ;;
    "compare_figure_6.py")
      echo " Running figure 6 comparison"
      python3 compare_figure_6.py -zf ../$zahedi_folder -ff ../$figures_folder \
      -r ../$simulation_results_file -c ../$cases_csv_file
      ;;
    *)
      echo " Unknown script: $1"
      echo " Skipping file."
      continue
      ;;

  esac

done

echo "********************************************************************************"
echo "********************************************************************************"
echo " Done :D"
echo "********************************************************************************"
echo "********************************************************************************"
