#!/bin/bash

# Check if the correct number of arguments were passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_original_file> <path_to_new_file>"
    exit 1
fi

# Assign the original file path and the new file path to variables
original_file_path="$1"
new_file_path="$2"

# Extract base names without the extension
original_file_name=$(basename "$original_file_path" .yaml)
new_file_name=$(basename "$new_file_path" .yaml)

# Check if the new file already exists
if [ -f "$new_file_path" ]; then
    echo "Error: The file $new_file_path already exists."
    exit 2
else
    # Copy the original file to the new file path
    cp "$original_file_path" "$new_file_path"
    echo "File copied to $new_file_path successfully."
    
    # Convert the base name of the new file to upper case
    upper_case_name=$(echo "$new_file_name" | tr '[:lower:]' '[:upper:]')
    
    # Use sed to update the NAME field in the copied YAML file
    sed -i "s/^  NAME: .*/  NAME: \"$upper_case_name\"/" "$new_file_path"
    echo "Updated NAME in $new_file_path to $upper_case_name."
    
    echo "To run for 5 hours, execute: 'qsub -N [NameForProjectInQ] -v CONFIG=\"$new_file_path\" /projectnb/textconv/distill/scripts/submit5hr.sh'"
    echo "To run for 10 hours, execute: 'qsub -N [NameForProjectInQ]-v CONFIG=\"$new_file_path\" /projectnb/textconv/distill/scripts/submit10hr.sh'"
fi
