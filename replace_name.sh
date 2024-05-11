#!/bin/bash

# Check if an argument is given
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_list_of_files>"
    exit 1
fi

# Read each line from the file
while IFS= read -r file; do
    # Extract the file name without extension and convert to uppercase
    filename=$(basename "$file" .yaml | tr '[:lower:]' '[:upper:]')
    
    # Use sed to replace the NAME: value with the new name
    sed -i "s/\(NAME:\).*/\1 \"$filename\"/" "$file"

done < "$1"