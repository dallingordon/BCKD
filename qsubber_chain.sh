#!/bin/bash

choose_yaml_list() {
    echo "Select a YAML file:"
    local files=($(ls /projectnb/textconv/distill/mdistiller/configs/imagenet/_sldmse/r50mnv1/*.yaml))
    local num_files=${#files[@]}
    
    if [[ $num_files -eq 0 ]]; then
        echo "No YAML files found in the directory."
        return
    fi

    local index=1
    for f in "${files[@]}"; do
        echo "$index) $(basename "$f")"
        ((index++))
    done

    while true; do
        read -p "Enter choice (1-$num_files): " choice
        if [[ $choice -ge 1 && $choice -le $num_files ]]; then
            yaml_path="${files[$choice-1]}"
            echo "Selected YAML file: $yaml_path"
            break
        else
            echo "Invalid selection. Please enter a value between 1 and $num_files."
        fi
    done
}



# Function to extract the next job number
getNextJobNumber() {
    # Get the highest job number currently used
    local highest_job_number=$(qstat -u dgordon | grep -o 'dg_[0-9]\+' | cut -d '_' -f2 | sort -n | tail -1)
    
    # If no jobs are found, default to 0
    if [ -z "$highest_job_number" ]; then
        highest_job_number=0
    else
        # Increment the highest job number by 1 to use for the next job
        highest_job_number=$((highest_job_number + 1))
    fi
    
    echo $highest_job_number
}
echo "hi"
# Main script starts here
choose_yaml_list


# Proceed with the rest of the script using $YAML_LIST and $QSUB_SCRIPT
echo "yaml_file = $yaml_path"
read -p "Enter an integer: " chain_count
echo "CHAIN = $chain_count"
file_name="${yaml_path##*/}"

# Check if the extension is .yaml
if [[ $file_name == *.yaml ]]; then
    # Remove the .yaml extension
    file_base="${file_name%.yaml}"
    echo "Base name: $file_base"
else
    echo "The file does not have a .yaml extension. Exiting..."
    exit 1
fi


job_number=$(getNextJobNumber)

# Initial setup for previous job name; there is no previous job initially
previous_job_name=""

highest_num=$(ls /projectnb/textconv/distill/mdistiller/configs/qsub_batches/qsub_*.sh | sed 's/[^0-9]//g' | sort -nr | head -n 1)
let highest_num++
QSUB_SCRIPT="/projectnb/textconv/distill/mdistiller/configs/qsub_batches/qsub_${highest_num}.sh"
touch "$QSUB_SCRIPT"


# Iterate chain_count times
for ((i = 1; i <= chain_count; i++)); do
    # Generate the current job name
    job_name="dg_$job_number"
    
    # Only add the hold_jid option if previous_job_name is not empty
    if [ -z "$previous_job_name" ]; then
        qsub_command="qsub -N \"$job_name\" -v CONFIG=\"$yaml_path\" -e \"/projectnb/textconv/distill/mdistiller/configs/imagenet_qsub_output/${file_base}.err\" -o \"/projectnb/textconv/distill/mdistiller/configs/imagenet_qsub_output/${file_base}.out\" /projectnb/textconv/distill/scripts/submit10hr.sh"
    else
        qsub_command="qsub -N \"$job_name\" -hold_jid \"$previous_job_name\" -v CONFIG=\"$yaml_path\" -e \"/projectnb/textconv/distill/mdistiller/configs/imagenet_qsub_output/${file_base}.err\" -o \"/projectnb/textconv/distill/mdistiller/configs/imagenet_qsub_output/${file_base}.out\" /projectnb/textconv/distill/scripts/submit10hrt1resume.sh"
    fi

    # Update previous_job_name for the next iteration
    previous_job_name=$job_name

    # Increment job_number for the next job
    job_number=$((job_number + 1))
    
    # Echo the qsub command (for now, instead of appending to a file)
    echo "$qsub_command" >> "$QSUB_SCRIPT"
done
chmod +x "$QSUB_SCRIPT"
echo "Created new QSUB_SCRIPT: $QSUB_SCRIPT"
