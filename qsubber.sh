#!/bin/bash

choose_yaml_list() {
    echo "Select a YAML list file:"
    local files=($(ls -t /projectnb/textconv/distill/mdistiller/configs/yaml_lists/* | head -n 5))
    local index=1
    for f in "${files[@]}"; do
        echo "$index) $(basename "$f")"
        let index++
    done

    while true; do
        read -p "Enter choice (1-5): " choice
        if [[ $choice =~ ^[1-5]$ ]]; then
            YAML_LIST="${files[$choice-1]}"
            echo "Selected YAML_LIST: $YAML_LIST"
            break
        else
            echo "Invalid selection. Please enter a value between 1 and 5."
        fi
    done
}

# Function to choose or create QSUB_SCRIPT
choose_qsub_script() {
    echo "Select a QSUB script file or create a new one:"
    local files=($(ls -t /projectnb/textconv/distill/mdistiller/configs/qsub_batches/*.sh | head -n 5))
    local index=1
    for f in "${files[@]}"; do
        echo "$index) $(basename "$f")"
        let index++
    done
    echo "N) Create a new QSUB script file"

    while true; do
        read -p "Enter choice (1-5 or N): " choice
        if [[ $choice =~ ^[1-5]$ ]]; then
            QSUB_SCRIPT="${files[$choice-1]}"
            echo "Selected QSUB_SCRIPT: $QSUB_SCRIPT"
            break
        elif [[ $choice =~ ^[Nn]$ ]]; then
            local highest_num=$(ls /projectnb/textconv/distill/mdistiller/configs/qsub_batches/qsub_*.sh | sed 's/[^0-9]//g' | sort -nr | head -n 1)
            let highest_num++
            QSUB_SCRIPT="/projectnb/textconv/distill/mdistiller/configs/qsub_batches/qsub_${highest_num}.sh"
            touch "$QSUB_SCRIPT"
            echo "Created new QSUB_SCRIPT: $QSUB_SCRIPT"
            break
        else
            echo "Invalid selection. Please enter a value between 1 and 5, or N for a new file."
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

# Main script starts here
choose_yaml_list
choose_qsub_script

# Proceed with the rest of the script using $YAML_LIST and $QSUB_SCRIPT
echo "YAML_LIST = $YAML_LIST"
echo "QSUB_SCRIPT = $QSUB_SCRIPT"




job_number=$(getNextJobNumber)

# Main loop to read each YAML file path and write qsub command to script file
while IFS= read -r yaml_path; do
    
    
    # Create the job name using the next job number
    job_name="dg_$job_number"
    
    # Remove the .yaml extension and prepend err_ and stdo_ for error and output file paths
    file_base=$(echo "$yaml_path" | sed 's/\.yaml$//')
    err_file="${file_base}.err"
    out_file="${file_base}.out"
    
    # Construct the qsub command
    qsub_command="qsub -N \"$job_name\" -v CONFIG=\"$yaml_path\" -e \"$err_file\" -o \"$out_file\" /projectnb/textconv/distill/scripts/submit10hr.sh"
    
    # Write the qsub command to the script file
    echo "$qsub_command" >> "$QSUB_SCRIPT"
    
    job_number=$((job_number + 1))
    
done < "$YAML_LIST"

# Make the output script executable
chmod +x "$QSUB_SCRIPT"