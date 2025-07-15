#!/usr/bin/env python3
"""
compute_sar_stats.py

Author:
Created: 2025-07-15

usage: compute_sar_stats.py [-h] [-q(uiet)] input_path

positional arguments:
  input_path   Path to the directory containing .npy files and manifest.json

options:
  -h, --help   show help message and exit
  -q, --quiet  show only warnings and final summary

Reads NumPy arrays from the specified input directory and computes statistics
for SAR data. The input directory should contain .npy files created by
unload_sar_data.py along with a manifest.json file. Statistics are saved
to a CSV file in the current working directory.
"""

import argparse
import time
import json
from pathlib import Path
import gc
import pandas as pd
import numpy as np

# Compute statistics for SLC product
def slc_stats(array):
    """
    Computation of basic statistics for SAR SLC data.
    Input is a 2D complex-valued numpy array.
    """
    # Pre-compute masks
    valid_mask = np.isfinite(array)
    nan_count = array.size - np.sum(valid_mask)
    zero_count = np.sum(array == 0)
    
    if np.sum(valid_mask) == 0:
        # All invalid data
        return None
    
    # Use views instead of copying data
    valid_data = array[valid_mask] if nan_count > 0 else array.ravel()
    
    # Compute real/imag parts
    real_part = valid_data.real
    imag_part = valid_data.imag
    
    # Compute amplitude and phase
    amplitude = np.abs(valid_data)
    phase = np.angle(valid_data)
    
    # Circular statistics - compute exp(1j * phase) once
    complex_phase = np.exp(1j * phase)
    mean_complex_phase = np.mean(complex_phase)
    circular_mean = np.angle(mean_complex_phase)
    r_value = np.abs(mean_complex_phase)
    circular_variance = 1 - r_value
    # Avoid log(0) issues
    circular_std = np.sqrt(-2 * np.log(np.maximum(r_value, 1e-16)))
    
    stats_dict = {
        # Data validity
        'valid_pixels': np.sum(valid_mask),
        'nan_count': nan_count,
        'zero_count': zero_count,
        'valid_percentage': np.sum(valid_mask) / array.size * 100,
        
        # Real component statistics
        'real_mean': np.mean(real_part),
        'real_std': np.std(real_part),
        'real_min': np.min(real_part),
        'real_max': np.max(real_part),
        
        # Imaginary component statistics  
        'imag_mean': np.mean(imag_part),
        'imag_std': np.std(imag_part),
        'imag_min': np.min(imag_part),
        'imag_max': np.max(imag_part),
        
        # Amplitude statistics
        'amplitude_mean': np.mean(amplitude),
        'amplitude_std': np.std(amplitude),
        'amplitude_min': np.min(amplitude),
        'amplitude_max': np.max(amplitude),
        'amplitude_median': np.median(amplitude),
        
        # Phase statistics
        'phase_mean': np.mean(phase),
        'phase_std': np.std(phase),
        'phase_min': np.min(phase),
        'phase_max': np.max(phase),
        'phase_circular_mean': circular_mean,
        'phase_circular_variance': circular_variance,
        'phase_circular_std': circular_std,
    }

    return stats_dict

# Compute statistics for GRD product
def grd_stats(array):
    """
    Computation of basic statistics for SAR GRD data.
    Input is a 2D real-valued numpy array.
    """
    # Pre-compute mask
    valid_mask = np.isfinite(array)
    nan_count = array.size - np.sum(valid_mask)
    zero_count = np.sum(array == 0)
    
    if np.sum(valid_mask) == 0:
        return None
    
    # Use valid data only
    valid_data = array[valid_mask] if nan_count > 0 else array.ravel()
    
    stats_dict = {
        # Data validity
        'valid_pixels': np.sum(valid_mask),
        'nan_count': nan_count,
        'zero_count': zero_count,
        'valid_percentage': np.sum(valid_mask) / array.size * 100,
        
        # Amplitude statistics (for GRD, the data itself is amplitude)
        'amplitude_mean': np.mean(valid_data),
        'amplitude_std': np.std(valid_data),
        'amplitude_min': np.min(valid_data),
        'amplitude_max': np.max(valid_data),
        'amplitude_median': np.median(valid_data),
        
        # Set unused columns to None for consistency
        'real_mean': None,
        'real_std': None,
        'real_min': None,
        'real_max': None,
        'imag_mean': None,
        'imag_std': None,
        'imag_min': None,
        'imag_max': None,
        'phase_mean': None,
        'phase_std': None,
        'phase_min': None,
        'phase_max': None,
        'phase_circular_mean': None,
        'phase_circular_variance': None,
        'phase_circular_std': None,
    }
    
    return stats_dict

def create_error_row(file_info: dict) -> dict:
    """Create a minimal row for files that couldn't be processed."""
    return {
        "safe_directory": file_info.get('safe_directory', ''),
        "filename": file_info.get('filename', '')
    }

def process_one_array(array_path: Path, file_info: dict, mode: str, quiet: bool = False) -> dict:
    """
    Process a single NumPy array file with shape recovery and statistics computation.
    """
    try:
        # Load the NumPy array
        try:
            data = np.load(str(array_path), allow_pickle=False)
            if not quiet:
                print(f"Loaded {array_path.name}: shape={data.shape}, size={data.size}, dtype={data.dtype}")
        except Exception as load_error:
            if not quiet:
                print(f"*** Error loading {array_path.name}: {load_error}")
            return create_error_row(file_info)
        
        # Validate the data array
        if data is None or data.size == 0:
            if not quiet:
                print(f"*** Warning: Empty or invalid data array for {array_path.name}")
            return create_error_row(file_info)

        # Check array dimensions - must be 2D
        if data.ndim != 2:
            if not quiet:
                print(f"*** Error: Array {array_path.name} has {data.ndim} dimensions, expected 2D. Shape: {data.shape}")
            return create_error_row(file_info)
        
        if not quiet:
            print(f"2D array: {data.shape}")

        # Validate data type for mode
        if "slc" in mode.lower() and not np.iscomplexobj(data):
            if not quiet:
                print(f"*** Warning: SLC data expected to be complex, but got {data.dtype} for {array_path.name}")
        elif "grd" in mode.lower() and np.iscomplexobj(data):
            if not quiet:
                print(f"*** Warning: GRD data expected to be real, but got complex data for {array_path.name}")
            # Convert complex to real by taking magnitude
            data = np.abs(data)
            if not quiet:
                print(f"Converted complex GRD data to magnitude for {array_path.name}")

        # Compute statistics using relevant function
        if "grd" in mode:
            stats_dict = grd_stats(data)
            stats_name = "grd_stats"
        else:
            stats_dict = slc_stats(data)
            stats_name = "slc_stats" 

        # Clean up memory
        del data
        gc.collect()

        if stats_dict is None:
            if not quiet:
                print(f"*** Warning: {stats_name} returned None for {array_path.name}.")
            return create_error_row(file_info)

        # Create row with metadata from file_info
        row = {
            "safe_directory": file_info.get('safe_directory', ''),
            "filename": file_info.get('filename', '')
        }
        row.update(stats_dict)
        return row
        
    except Exception as e:
        error_type = type(e).__name__
        if not quiet:
            print(f"*** {error_type} processing {array_path.name}: {e}")
        return create_error_row(file_info)
    finally:
        # Force cleanup
        if 'data' in locals() and locals()['data'] is not None:
            try:
                del data
            except:
                pass
        gc.collect()

def add_scene_id(df: pd.DataFrame, correspondence_file: Path) -> tuple[pd.DataFrame, int]:
    """
    Add scene_id column to dataframe using correspondence file.
    Returns the updated dataframe and count of missing mappings.
    """
    if not correspondence_file.exists():
        print(f"Warning: Correspondence file '{correspondence_file}' not found")
        return df, len(df)
    
    try:
        corr_df = pd.read_csv(correspondence_file)
    except Exception as e:
        print(f"Warning: Error reading correspondence file: {e}")
        return df, len(df)
    
    # Create mapping dictionary based on available columns
    if 'SLC_product_identifier' in corr_df.columns and 'GRD_product_identifier' in corr_df.columns:
        # Create combined mapping - try SLC first, then GRD
        slc_mapping = corr_df.set_index('SLC_product_identifier')['scene_id'].to_dict()
        grd_mapping = corr_df.set_index('GRD_product_identifier')['scene_id'].to_dict()
        
        # Combine mappings
        mapping_dict = {**slc_mapping, **grd_mapping}
    elif 'SLC_product_identifier' in corr_df.columns:
        mapping_dict = corr_df.set_index('SLC_product_identifier')['scene_id'].to_dict()
    elif 'GRD_product_identifier' in corr_df.columns:
        mapping_dict = corr_df.set_index('GRD_product_identifier')['scene_id'].to_dict()
    else:
        print("Warning: No SLC or GRD product identifier columns found in correspondence file")
        return df, len(df)
    
    # Remove .SAFE suffix and map to scene_id
    safe_dirs_clean = df['safe_directory'].str.replace('.SAFE', '', regex=False)
    scene_ids = safe_dirs_clean.map(mapping_dict)
    
    # Insert scene_id as first column
    df.insert(0, 'scene_id', scene_ids)
    
    # Count and report missing mappings
    missing_count = scene_ids.isnull().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} rows could not be mapped to scene_id")
    
    return df, missing_count

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process NumPy arrays and compute statistics. "
            "The input directory should contain .npy files and a manifest.json file."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the directory containing .npy files and manifest.json"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-file progress messages; only warnings and final summary are shown"
    )
   
    args = parser.parse_args()
    input_path = args.input_path
    quiet = args.quiet

    if not input_path.is_dir():
        parser.error(f"Provided input_path ({input_path}) is not a directory or does not exist.")

    # Load manifest file
    manifest_path = input_path / "manifest.json"
    if not manifest_path.exists():
        parser.error(f"Manifest file not found: {manifest_path}")

    try:
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
    except Exception as e:
        parser.error(f"Error reading manifest file: {e}")

    mode = manifest_data.get('pattern', 'slc-vh')
    correspondence_file = Path(manifest_data.get('correspondence_file', 'xView3_SLC_GRD_correspondences.csv'))

    print("Processing arrays...")
    if not quiet:
        print(f"Processing {mode} arrays from: {input_path}")
        print(f"Manifest contains {len(manifest_data.get('files', []))} file entries")

    # Filter for successfully unloaded files AND check if they actually exist
    files_info = manifest_data.get('files', [])
    successful_files = []
    missing_files = 0
    
    for f in files_info:
        if f['status'] == 'success':
            array_path = Path(f['array_path'])
            if array_path.exists():
                successful_files.append(f)
            else:
                missing_files += 1
                if not quiet:
                    print(f"*** Warning: Array file not found: {array_path}")
    
    total_files = len(successful_files)
    
    if total_files == 0:
        print("No valid array files found in manifest")
        if missing_files > 0:
            print(f"Note: {missing_files} files listed in manifest but not found on disk")
        return

    if missing_files > 0:
        print(f"Found {total_files} valid arrays ({missing_files} missing from manifest)")

    # Process arrays sequentially
    start_time = time.time()
    rows = []
    
    for idx, file_info in enumerate(successful_files, start=1):
        array_path = Path(file_info['array_path'])
        
        if not quiet:
            print(f"\nProcessing array {idx}/{total_files}: {array_path.name}")
            file_start_time = time.time()
        
        row = process_one_array(array_path, file_info, mode, quiet)
        rows.append(row)

        if not quiet:
            elapsed = time.time() - file_start_time
            print(f"Completed in {elapsed:.2f} seconds")
            
            # Show progress
            progress = idx / total_files * 100
            print(f"Progress: {idx}/{total_files} ({progress:.1f}%)")

    processing_time = time.time() - start_time

    if not rows:
        print("No arrays processed successfully")
        return

    # Build pandas DataFrame with predefined columns
    columns = [
        "safe_directory",
        "filename",
        "valid_pixels",
        "nan_count",
        "zero_count",
        "valid_percentage",
        "real_mean",
        "real_std",
        "real_min",
        "real_max",
        "imag_mean",
        "imag_std",
        "imag_min",
        "imag_max",
        "amplitude_mean",
        "amplitude_std",
        "amplitude_min",
        "amplitude_max",
        "amplitude_median",
        "phase_mean",
        "phase_std",
        "phase_min",
        "phase_max",
        "phase_circular_mean",
        "phase_circular_variance",
        "phase_circular_std",
    ]

    df = pd.DataFrame(rows, columns=columns)
    
    # Add scene_id mapping
    df_with_scene, missing_count = add_scene_id(df, correspondence_file)

    # Save the DataFrame as a CSV file
    output_csv = Path.cwd() / f"{mode}_stats.csv"
    df_with_scene.to_csv(output_csv, index=False)

    print(f"\nStats for {len(df)} array(s) written to: {output_csv}")
    if missing_count > 0:
        print(f"Note: {missing_count} files could not be mapped to scene_id")
    print(f"Total processing time: {processing_time:.2f} seconds")
    if len(rows) > 0:
        print(f"Average time per array: {processing_time/len(rows):.2f} seconds")


if __name__ == "__main__":
    main()
