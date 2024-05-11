#!/usr/bin/env python
import yaml
import os
import re
import shutil
import sys

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def show_options(copy_file):
    copy_yaml = load_yaml(copy_file)
    print(f"Options for config file {copy_file}" )
    for base_item in copy_yaml.items():
        print("___________________________")
        for sub_key in base_item[1]:
            print(f"{base_item[0]}.{sub_key}")
            


def generate_next_filename(file_location):
    # Extract directory and base filename
    directory, filename = os.path.split(file_location)
    base_name, ext = os.path.splitext(filename)
    
    # Pattern to match files with similar names and an underscore followed by digits
    pattern = re.compile(rf'^{re.escape(base_name)}_(\d+){re.escape(ext)}$')
    
    # Initialize the highest number found
    max_num = -1
    
    # Scan the directory for files that match the pattern
    for file in os.listdir(directory):
        match = pattern.match(file)
        if match:
            # Extract the number and update max_num if this number is higher
            num = int(match.group(1))
            max_num = max(max_num, num)
    
    # Generate the next filename
    next_num = max_num + 1
    next_filename = f"{base_name}_{next_num}{ext}"
    next_file_location = os.path.join(directory, next_filename)
    
    return next_file_location


            
def update_yaml(copy_file,yaml_output):
    copy_yaml = load_yaml(copy_file)
    valid_configs = {}
    i = 1
    print("select: ")
    for base_item in copy_yaml.items():
        for sub_key in base_item[1]:
            config_val = f"{base_item[0]}.{sub_key}"
            valid_configs[i] = config_val
            print(f"{i}. {config_val}")
            i += 1
            
    try:
        selection = int(input("Select an option by number: "))
        # Validate selection
        if selection in valid_configs.keys():
            selected_option = valid_configs[selection]
            print(f"You selected: {selected_option}")
            
            try:
                config_input = input(f"input comma delimited (no spaces) for {selected_option}:")
                config_values = config_input.split(',')
                #print(config_values)
                ##iterate over these and make a file for each.  check if the file is valid name, underscores and such
                for c_v in config_values:
                    #print(c_v)
                    next_file_location = generate_next_filename(copy_file) #gets the next file name 
                    shutil.copyfile(copy_file, next_file_location)
                    with open(next_file_location, 'r') as file:
                        lines = file.readlines()
                   


                    keys = selected_option.split('.')
                    #print(len(lines))
                    updated = False
                    key_index = 0  # Start with the first key
                    for i in range(len(lines)):
                        line = lines[i].strip()
                        # Check if the current line contains the current key
                        if keys[key_index] + ":" in line:

                            

                            if key_index == 1:  # Last key, time to update the value
                                # Find the colon index and replace everything after it with the new value

                                colon_index = lines[i].find(':')

                                lines[i] = lines[i][:colon_index + 1] + ' ' + str(c_v) + '\n'
                                updated = True
                                break
                                
                            key_index = 1  # Move to the next key
                        

                    if updated:
                        # Write the updated content back to the file
                        file_path = next_file_location #update this to be the new config file
                        with open(file_path, 'w') as file:
                            file.writelines(lines)
                        print(f"made {file_path}")
                        ##write to the list of yaml files.  
                        if not os.path.exists(yaml_output):
                            os.makedirs(os.path.dirname(yaml_output), exist_ok=True)
                            with open(yaml_output, 'w') as file:  # 'w' mode will create the file if it doesn't exist
                                pass  # File is created, nothing written yet
    
                        # Append the string to the file
                        with open(yaml_output, 'a') as file:  # 'a' mode for appending to the file
                            file.write(file_path + '\n') 
                        
                    else:
                        raise ValueError("The specified cfg_key_string does not match any key in the file")
                
                
                
            except ValueError:
                print("commas, no spaces please.  the split needs at least one comma momma:")
        else:
            print("Invalid selection. Please select a number listed above.")
    except ValueError:
        print("Please enter a valid number.")
        
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_script.py copy_file_path yaml_list_file_path")
        sys.exit(1)
    copy_file = sys.argv[1]
    yaml_output = sys.argv[2]
    update_yaml(copy_file,yaml_output)