#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import yaml
import re
import os
import pandas as pd
from datetime import datetime

def extract_float(s):
    # This regex pattern looks for any sequence of digits (\d+), optionally followed by
    # a decimal point and more digits (\.\d+)? The entire pattern is wrapped in parentheses
    # to capture the match as a group.
    match = re.search(r'(\d+(\.\d+)?)', s)
    if match:
        return float(match.group(0))
    else:
        return None

def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flattens a nested dictionary and concatenates keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def read_yaml_files(directory):
    data = []
    unique_keys = set()
    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                # Load and flatten the YAML file
                yaml_data = yaml.safe_load(file)
                flattened_data = flatten_dict(yaml_data)
                data.append(flattened_data)
                # Update unique keys set
                unique_keys.update(flattened_data.keys())
    return data, unique_keys

def write_to_flat_file(data, unique_keys, output_file):
    with open(output_file, 'w') as file:
        # Write the header
        headers = list(unique_keys)
        headers.sort()  # Optionally sort the headers for consistent ordering
        file.write('\t'.join(headers) + '\n')
        # Write the data
        for item in data:
            row = []
            for key in headers:
                row.append(str(item.get(key, "")))
            file.write('\t'.join(row) + '\n')






# In[3]:




def add_best_acc_to_file(input_file, output_file, directory_prefix):
    # Load the tab-delimited file into a DataFrame
    df = pd.read_csv(input_file, delimiter='\t')
    
    # Ensure the EXPERIMENT.NAME column exists
    if 'EXPERIMENT.NAME' not in df.columns:
        raise ValueError("EXPERIMENT.NAME column not found in the input file.")
    
    # Initialize the BEST_ACC column with NaNs (or a default value of your choice)
    df['BEST_ACC'] = float('nan')
    
    # Iterate over the rows in the DataFrame
    for index, row in df.iterrows():
        experiment_name = str(row['EXPERIMENT.NAME'])
        folder_path = os.path.join(directory_prefix, experiment_name)
        worklog_path = os.path.join(folder_path, 'worklog.txt')
        
        try:
            # Attempt to open and read the worklog.txt file
            with open(worklog_path, 'r') as worklog_file:
                for line in worklog_file:
                    # Look for the line starting with 'best_acc'
                    if line.startswith('best_acc'):
                        # Extract the value after 'best_acc' and update the DataFrame
                        best_acc_value = extract_float(line.split()[-1])  # Assuming the value is the last item on the line
                        
                        df.at[index, 'BEST_ACC'] = best_acc_value
                        break
        except FileNotFoundError:
            # Handle the case where the worklog.txt file does not exist
            print(f"worklog.txt not found for {experiment_name} in {folder_path}")
    
    # Save the updated DataFrame to a new tab-delimited file
    df.to_csv(output_file, sep='\t', index=False)



def main():
    directory = '/projectnb/textconv/distill/mdistiller/configs/cifar100/sld/'
    output_file = 'sample_out.tsv'

    # Process the YAML files
    data, unique_keys = read_yaml_files(directory)

    # Write the collected data to a flat file
    write_to_flat_file(data, unique_keys, output_file)

    input_file = 'sample_out.tsv'  # The path to your tab-delimited flat file


    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time in a way that is safe for filenames
    filename_format = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    output_file = f'sample_out_updated_{filename_format}.tsv'  # The path for the output file with the BEST_ACC column
    directory_prefix = '/projectnb/textconv/distill/mdistiller/output/cifar100_baselines/'  # The prefix to the directory containing experiment folders

    # Call the function with the specified paths
    add_best_acc_to_file(input_file, output_file, directory_prefix)


if __name__ == "__main__":
    main()




