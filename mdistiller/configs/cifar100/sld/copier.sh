#!/bin/bash

# Assign the original file name and the new file name to variables
original_file="$1"
new_file="$2"

# Check if the correct number of arguments were passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <original_file_name> <new_file_name>"
    exit 1
fi

# Check if the new file name already exists
if [ -f "$new_file" ]; then
    echo "Error: The file $new_file already exists."
    exit 2
else
    # Copy the original file to the new file name
    cp "$original_file" "$new_file"
    echo "File copied to $new_file successfully."
    
    base_name=$(basename "$new_file" .yaml)
    upper_case_name=$(echo "$base_name" | tr '[:lower:]' '[:upper:]')
    
    # Use sed to update the NAME field in the copied YAML file
    sed -i "s/^  NAME: .*/  NAME: \"$upper_case_name\"/" "$new_file"
    echo "Updated NAME in $new_file to $upper_case_name."
fi

script_dir="/projectnb/textconv/distill/scripts"
script_path="${script_dir}/${upper_case_name}.sh"

# Check if the script already exists
if [ -f "$script_path" ]; then
    echo "Error: The script $script_path already exists."
    exit 3
else
    # Create the new shell script with replacements
    cat > "$script_path" <<- EOM
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=05:00:00   # Specify the hard time limit for the job
#$ -N $upper_case_name # Project name. unique every time
#$ -o std_out_$upper_case_name # standard out file
#$ -e err_$upper_case_name # error file
#$ -l gpus=1
#$ -l gpu_type=V100
#$ -pe omp 2
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules
module load python3/3.8.10
# module load pytorch/1.9.0
cd /projectnb/textconv/distill 
source venv/bin/activate
export PYTHONPATH="/projectnb/textconv/distill/mdistiller:\$PYTHONPATH"
cd /projectnb/textconv/distill/mdistiller
python3 setup.py develop
pip install -r requirements.txt

# Run Script:
python3 tools/train.py --cfg configs/cifar100/sld/$base_name.yaml

deactivate
EOM
    echo "Shell script $script_path created successfully."
fi
