"""
utility.py

A set of helper functions.

Author: Peter Millitz
Created: 2025-07-14

"""

import random
import subprocess
import csv
import numpy as np
import cv2
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm

def train_val_test_split(lst, test_ratio=0.1, val_ratio=0.1, seed=None):
    """
    Function to randomly allocate scene ids into train, validation, and test lists
    
    Args:
        lst: List of items to split
        test_ratio: Proportion of data for test set (default 0.1)
        val_ratio: Proportion of data for validation set (default 0.1)
        seed: Random seed for reproducibility (default None)
    
    Returns:
        tuple: (train_set, val_set, test_set)
    """
    if seed is not None:
        random.seed(seed)
        
    # Make a copy to avoid modifying the original list
    shuffled = lst.copy()
    random.shuffle(shuffled)
    
    # Calculate split points
    test_size = max(1, int(len(lst) * test_ratio))
    val_size = int(len(lst) * val_ratio) if val_ratio > 0 else 0
    
    # Ensure we don't exceed the list length
    total_holdout = test_size + val_size
    if total_holdout >= len(lst):
        raise ValueError(f"Combined test and validation ratios ({test_ratio + val_ratio}) are too large for dataset size {len(lst)}")
    
    # Split the data
    test_set = shuffled[:test_size]
    val_set = shuffled[test_size:test_size + val_size]
    train_set = shuffled[test_size + val_size:]
    
    return train_set, val_set, test_set
  
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

import numpy as np
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_sar_dataset_to_png(images_input_path, images_output_path, labels_input_path, labels_output_path):
    """
    Convert SAR dataset from .npy to .png format.
    
    Args:
        images_input_path (str or Path): Path to directory containing .npy image files
        images_output_path (str or Path): Path to output directory for PNG images
        labels_input_path (str or Path): Path to directory containing .txt label files
        labels_output_path (str or Path): Path to output directory for copied labels
    """
    print("Converting SAR Dataset to PNG Format...")
    
    # Setup paths
    images_input_path = Path(images_input_path)
    images_output_path = Path(images_output_path)
    labels_input_path = Path(labels_input_path)
    labels_output_path = Path(labels_output_path)
    
    # Create output directories
    images_output_path.mkdir(parents=True, exist_ok=True)
    labels_output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all .npy files
    npy_files = list(images_input_path.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files to convert")
    
    if not npy_files:
        print(f"No .npy files found in {images_input_path}")
        return
    
    # Convert with progress bar
    converted = 0
    failed = 0
    
    for npy_file in tqdm(npy_files, desc="Converting"):
        try:
            # Load SAR data (3, H, W) float32 [0,1] - variable dimensions
            arr = np.load(npy_file).astype(np.float32)
            
            # Convert to standard image format
            if arr.shape[0] == 3:  # (3, H, W) to (H, W, 3)
                arr = np.transpose(arr, (1, 2, 0))
           
            # Convert to uint8 [0, 255]
            arr = (arr * 255).astype(np.uint8)
            
            # Save as PNG
            png_file = images_output_path / f"{npy_file.stem}.png"
            success = cv2.imwrite(str(png_file), arr)
            
            if success:
                # Copy corresponding label
                label_file = labels_input_path / f"{npy_file.stem}.txt"
                if label_file.exists():
                    new_label_file = labels_output_path / f"{npy_file.stem}.txt"
                    shutil.copy(label_file, new_label_file)
                    converted += 1
                else:
                    print(f"Warning: No label for {npy_file.name}")
            else:
                failed += 1
                
        except Exception as e:
            failed += 1
            if failed <= 5:  # Show first 5 errors
                print(f"Error with {npy_file.name}: {e}")
    
    print(f"Conversion complete: {converted} successful, {failed} failed")

