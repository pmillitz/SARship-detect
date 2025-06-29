"""
utility.py

A set of helper functions.

Author: Peter Millitz
Created: 2025-06-27

"""

import random
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

