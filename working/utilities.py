"""
utility.py

A set of helper functions.

Author: Peter Millitz
Created: 2025-06-27

"""

import random
import subprocess
import csv


def train_test_split(lst, test_ratio=0.1, seed=None):
    """
    Function to randomly allocate scene ids into train and test lists
    """
    if seed is not None:
        random.seed(seed)
        
    # Make a copy to avoid modifying the original list
    shuffled = lst.copy()
    random.shuffle(shuffled)
    
    # Calculate split point
    test_size = max(1, int(len(lst) * test_ratio))
    
    # Split the data
    test_set = shuffled[:test_size]
    train_set = shuffled[test_size:]
    
    return train_set, test_set

def print_list_formatted(lst, items_per_row=10, var_name="scene_list"):
    """
    Function to print a scene list formatted
    """
    print(f'{var_name} = [', end='')
    
    for i, item in enumerate(lst):
        # Add newline and indentation for new rows (except the first item)
        if i > 0 and i % items_per_row == 0:
            print()
            print(' ' * (len(var_name) + 4), end='')  # Indent to align with opening bracket
        
        # Add comma and space before item (except for first item)
        if i > 0:
            print(',', end='')
        
        # Print the item with quotes
        print(f'"{item}"', end='')
    
    print(']')

def save_list_to_csv(lst, filename, column_name="item"):
    """
    Function to save a list as a CSV file.
    Usage example: save_list_to_csv(train_set, "train_set.csv", "scene_id")
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([column_name])  # Header row
        
        for item in lst:
            writer.writerow([item])
    
    print(f"List saved to {filename}\n")

def save_list_to_txt(my_list, filename):
    """
    Function that takes a list and saves it as text file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for item in my_list:
            f.write(f"{item}\n")

    print(f"List saved to {filename}\n")

def filter_rows(df_chunk, column, values):
    """
    Function to filter based on a column's value 
    """
    return df_chunk[df_chunk[column].isin(values)]


def extract_list_from_command(command, output_file=None, print_summary=False, columns=1, list_name="my_list"):
    """
    Runs a shell command, optionally saves output to a file, and returns the result as a list of strings.
    
    Parameters:
        command (str): Shell command to execute.
        output_file (str): Optional path to save the command output.
        print_summary (bool): Whether to print the list summary.
        columns (int): Number of columns to use when printing the list.
        list_name (str): Label to use when printing the list.
    
    Returns:
        List[str]: The processed list of strings.
    """
    # Run the shell command and capture output
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    lines = result.stdout.strip().split('\n')

    # Optionally write to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"List saved to {output_file}\n")

    # Optionally print summary
    if print_summary:
        print(f"Total number of items in {list_name}: {len(lines)}")
        print_list_formatted(lines, columns, list_name)

    return lines


