#!/bin/bash

# This script generates JSON parameter files for cases specified

USAGE="Usage: $0 [-t <path_to_template_parameter_file>] \n
      \t[-p <path_to_the_csv_file_with_case_parameters>] \n
      \t[-f <path_to_parameter_files_folder>] \n
      \t[-c <\"sequence_of_case_numbers\">] \n
      \n
       -t: Path to template parameter file (default: ./zahedi_wall_wetting.tpl) \n
       -p: Path to CSV file with case parameters (default: ./case_parameters.csv) \n
       -f:  Path to parameter files folder (default: ../../parameter_files/) \n
       -c: Case numbers (default: \"0-20\") \n
       -h: Show help \n
       \n
      The  <\"sequence_of_case_numbers\"> argument corresponds to the case numbers in the csv file.\n
      \t The numbers should be separated with a comma (',') delimiter. \n
      \t Ranges can also be specified using a hyphen ('-'). \n
      \t For example, \"0-5, 9, 30-32\" includes cases 0, 1, 2, 3, 4, 5, 9, 30, 31, and 32."

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -t|--template)
      template_parameter_file="$2"
      shift 2
      ;;
    -p|--parameters_csv_file)
      cases_csv_file="$2"
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
template_parameter_file="${template_parameter_file:-./zahedi_wall_wetting.tpl}"
cases_csv_file="${cases_csv_file:-./case_parameters.csv}"
parameter_files_folder="${parameter_files_folder:-../../parameter_files/}"
case_sequence="${case_sequence:-"0-20"}"

# Check if the file exists
if [ ! -f "$template_parameter_file" ]; then
  echo "File not found: $template_parameter_file"
  exit 1
fi

# Template parameter file prefix for generating case parameter files
template_parameter_file_prefix=$(basename "$template_parameter_file" .tpl)

# Check if the parameter_files folder exists, and if not create it
if [ ! -d "$parameter_files_folder" ]; then
  mkdir -p "$parameter_files_folder"
fi

# Check if the file exists
if [ ! -f "$cases_csv_file" ]; then
  echo "File not found: $cases_csv_file"
  exit 1
fi

# Print values to user
echo "Template parameter file:  $template_parameter_file"
echo "CSV case parameter files: $cases_csv_file"
echo "Params folder:            $parameter_files_folder"
echo "Cases:                    $case_sequence"

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

echo "********************************************************************************"
echo "********************************************************************************"
echo " GENERATING SIMULATION CASES FOR ZAHEDI (2009) COMPARISON"
echo "********************************************************************************"

# Loop over specified cases
for case_number in ${case_numbers_array[@]}
do
  echo "********************************************************************************"
  # Get parameters of the case and store them in appropriate variables
  padded_case_number=$(printf "%03d" $case_number)
  case_name=$(echo $template_parameter_file_prefix"_"$padded_case_number)
  line=$(grep "^${case_name}," "$cases_csv_file")

  # Check if the case is in the csv file
  if [[ -z "$line" ]]; then
    echo  -e " Warning: case \"$case_name\" was not found in \"$cases_csv_file\"."\
    "\n Skipping case."
    continue
  fi

  # Parse parameters
  IFS=',' read -r -a case_parameters <<< "$line"
  static_contact_angle=${case_parameters[1]}
  epsilon_n_factor=${case_parameters[2]}
  epsilon_t_factor=${case_parameters[3]}
  gamma_factor=${case_parameters[4]}
  end_time=${case_parameters[5]}
  end_time="${end_time//[$'\t\r\n ']/}" # strip last argument of whitespace characters

  echo " Writing files for:"
  echo " \"$case_name\""
  echo ""
  echo " Parameter values:"
  echo "  STATIC_CONTACT_ANGLE:       $static_contact_angle"
  echo "  EPSILON_N_FACTOR:           $epsilon_n_factor"
  echo "  EPSILON_T_FACTOR:           $epsilon_t_factor"
  echo "  GAMMA_FACTOR:               $gamma_factor"
  echo "  END_TIME:                   $end_time"
  echo ""

  # Create parameter file for the case
  case_parameter_file=$(echo $case_name".json")
  sed "s/CASE_NAME/$case_name/g" $template_parameter_file > "$parameter_files_folder/$case_parameter_file"
  sed -i "s/STATIC_CONTACT_ANGLE/$static_contact_angle/g" "$parameter_files_folder/$case_parameter_file"
  sed -i "s/EPSILON_N_FACTOR/$epsilon_n_factor/g" "$parameter_files_folder/$case_parameter_file"
  sed -i "s/EPSILON_T_FACTOR/$epsilon_t_factor/g" "$parameter_files_folder/$case_parameter_file"
  sed -i "s/GAMMA_FACTOR/$gamma_factor/g" "$parameter_files_folder/$case_parameter_file"
  sed -i "s/END_TIME/$end_time/g" "$parameter_files_folder/$case_parameter_file"
done

echo "********************************************************************************"
echo "********************************************************************************"
echo " Done :D"
echo "********************************************************************************"
echo "********************************************************************************"
