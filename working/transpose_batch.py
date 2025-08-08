#!/usr/bin/env python3

"""
transpose_batch.py

Simple script to transpose .npy files from (H, W, 3) to (3, H, W) format.

Usage:
    python transpose_batch.py input_dir output_dir
"""

import numpy as np
import argparse
import os
from pathlib import Path
import sys
from datetime import datetime


def find_npy_files(directory):
    """Find all .npy files in directory."""
    directory = Path(directory)
    npy_files = list(directory.glob("*.npy"))
    npy_files.extend(directory.glob("*.NPY"))
    return sorted([str(f) for f in npy_files])


def transpose_file(input_path, output_path):
    """
    Transpose a single .npy file from (H, W, 3) to (3, H, W).
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Load and transpose
        data = np.load(input_path)
        
        # Validate shape
        if data.ndim != 3 or data.shape[2] != 3:
            return False
        
        # Transpose from (H, W, 3) to (3, H, W)
        transposed_data = np.transpose(data, (2, 0, 1))
        
        # Save
        np.save(output_path, transposed_data)
        return True
        
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Transpose .npy files from (H, W, 3) to (3, H, W)")
    parser.add_argument('input_dir', help='Input directory containing .npy files')
    parser.add_argument('output_dir', help='Output directory for transposed files')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find files
    npy_files = find_npy_files(args.input_dir)
    
    if not npy_files:
        print(f"No .npy files found in {args.input_dir}")
        sys.exit(1)
    
    # Process files
    print(f"Transposing {len(npy_files)} files...")
    start_time = datetime.now()
    
    successful = 0
    failed = 0
    
    for i, input_path in enumerate(npy_files, 1):
        print(f"Processing {i}/{len(npy_files)}", end='\r', flush=True)
        
        # Create output path
        output_path = Path(args.output_dir) / Path(input_path).name
        
        # Process file
        if transpose_file(input_path, output_path):
            successful += 1
        else:
            failed += 1
    
    # Summary
    duration = datetime.now() - start_time
    print(f"\nCompleted: {successful}/{len(npy_files)} files successful")
    if failed > 0:
        print(f"Failed: {failed} files")
    print(f"Duration: {duration}")


if __name__ == "__main__":
    main()
