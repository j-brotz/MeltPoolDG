#!/bin/bash

# This script launches cases specified and logs consol outputs in the "./logs/" folder

USAGE="Usage: $0 [-e path_to_mp-reinit-executable] [-np <number of processors to run simulations on>] \n
      \t[-f <path_to_parameter_files_folder>] [-c <\"sequence_of_case_numbers\">] \n
      \n
       -e:  Path to mp-reinit executable (default: ../../../../../../../build_release/applications/mp-reinit/mp-reinit) \n
       -np: Number of processes on which the simulations are run (default: 20) \n
       -f:  Path to parameter files folder (default: ../../parameter_files/) \n
       -c:  Case numbers (default: \"0-20\") \n
       -h:  Show help \n
       \n
       The  <\"sequence_of_case_numbers\"> argument corresponds to the case numbers in the csv file.\n
       \t The numbers should be seperated with a comma (',') delimiter. \n
       \t Ranges can also be specified using a hyphen ('-'). \n
       \t For example, \"0-5, 9, 30-32\" includes cases 0, 1, 2, 3, 4, 5, 9, 30, 31, and 32."

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -e|--exec)
      mp_reinit_executable_path="$2"
      shift 2
      ;;
    -np|--n_procs)
      number_of_processors="$2"
      shift 2
      ;;
    -f|--parameter_files_folder)
      parameter_files_folder="$2"
      shift 2
      ;;
    -c|--cases)
      case_sequence="$2"
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

# Assign default values if not set
mp_reinit_executable_path="${mp_reinit_executable_path:-../../../../../../../build_release/applications/mp-reinit/mp-reinit}"
parameter_files_folder="${parameter_files_folder:-../../parameter_files/}"
number_of_processors="${number_of_processors:-20}"
case_sequence="${case_sequence:-"0-20"}"

# Print values to user
echo "Executable:           $mp_reinit_executable_path"
echo "Number of processors: $number_of_processors"
echo "Params folder:        $parameter_files_folder"
echo "Cases:                $case_sequence"

# Function to expand case number ranges and remove superficial characters
expand_case_numbers() {
  local input="$1"
  local result=()
  input="${input//[\{\}]/}"  # Remove curly braces
  input="${input//[[:space:]]/}" # Remove whitespaces

  IFS=',' read -ra tokens <<< "$input"
  for token in "${tokens[@]}"; do
    if [[ "$token" =~ ^[0-9]+-[0-9]+$ ]]; then
      IFS='-' read -r start end <<< "$token"
      for ((i=start; i<=end; i++)); do
        result+=("$i")
      done
    elif [[ "$token" =~ ^[0-9]+$ ]]; then
      result+=("$token")
    else
      echo "Invalid range or number: $token"
      exit 1
    fi
  done

  echo "${result[@]}"
}

# Make an array of case numbers
IFS=',' read -ra case_numbers_array <<< "$(expand_case_numbers "$case_sequence")"

# Check if the outputs folder exits, and if not create it
results_folder="../../meltpooldg_results"
if [ ! -d "$results_folder" ]; then
  mkdir -p "$results_folder"
fi

# Check if the logs folder exits, and if not create it
logs_folder="${results_folder}/logs"
if [ ! -d "$logs_folder" ]; then
  mkdir -p "$logs_folder"
fi

# Create a summary file
timestamp=$(date +"%Y%m%d_%H%M%S")
output_csv="${results_folder}/contact_angles_summary_${timestamp}.csv"
echo "case_number,dt_value,static_contact_angle,computed_contact_angle" > "$output_csv"

# Case processing function
process_case() {
  local parameter_file="$1"
  echo "Processing file: $parameter_file"

  case_name="${parameter_file%.json}"
  case_number=$(echo "$case_name" | grep -oP '\d{3}$')
  log_file="${logs_folder}/${case_name}.txt"
  echo "Logging simulation into: $log_file"

  # Run simulation and log into log file
  mpirun -np $number_of_processors $mp_reinit_executable_path "$parameter_files_folder/$parameter_file" > "$log_file"

  log_tail=$(tail -n 25 "$log_file")
  dt_value=$(echo "$log_tail" | grep -oP 'dt = \K[0-9.e+-]+' | head -n 1)
  static_contact_angle=$(echo "$log_tail" | grep -oP 'Static contact angle: \K[0-9.]+' | tail -n 1)
  contact_angle=$(echo "$log_tail" | grep -oP 'Computed contact angle: \K[0-9.]+' | tail -n 1)

  echo " dt = $dt_value"
  echo " static_contact_angle = $static_contact_angle"
  echo " computed_contact_angle = $contact_angle"
  echo "$case_number,$dt_value,$static_contact_angle,$contact_angle" >> "$output_csv"
  echo "********************************************************************************"
}

echo "********************************************************************************"
echo "********************************************************************************"
echo " LAUNCHING CASES FOR ZAHEDI (2009) COMPARISON"
echo "********************************************************************************"
echo "********************************************************************************"

# Get template parameter file
template_parameter_file=$(ls | grep .tpl)
template_parameter_file_prefix=$(basename "$template_parameter_file" .tpl)

# Process each specified case
for case_number in ${case_numbers_array[@]}
 do
  padded_case_number=$(printf "%03d" $case_number)
  parameter_file=$(echo $template_parameter_file_prefix"_"$padded_case_number".json")
  # Check if the file exists
  if [ ! -f "$parameter_files_folder/$parameter_file" ]; then
    echo "File not found: $parameter_files_folder/$parameter_file"
    echo "Please generate case using 'generate_cases.sh'"
    exit 1
  fi
  process_case "$parameter_file"
done

echo "********************************************************************************"
echo " Done :D"
echo "********************************************************************************"
echo "********************************************************************************"