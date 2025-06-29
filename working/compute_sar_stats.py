#!/usr/bin/env python3
"""
compute_sar_stats.py

Author: Peter Millitz
Created: 2025-06-28

usage: compute_sar_stats.py [-h]
                            [--pattern {slc-vh, slc-vv, grd-vh, grd-vv}]
                            [--save-array] [--correspondence-file PATH] 
                            [--include-scene-ids PATH]
                            [-q(uiet)] 
                            root_path 

Recursively searches for all GeoTIFF files under a given root directory whose filename
contains any one pattern in ["slc-vh", "slc-vv", "grd-vh", "grd-vv"] and ends with ".tiff".
For each matching file, the SLC or GRD data is unpacked into a NumPy array and a range of
statistics computed. Optionally, the extracted array can be saved for later use. Each tiff
file's statistics are inserted as a new row into a dataframe which is saved to a CSV file
named "{pattern}_stats.csv", on exit. The correspondence file is used for mapping scene ids
with image tiffs. An optional inclusion list can be used to process only specific scene_ids.
"""

import argparse
import time
from pathlib import Path
import gc
import os
import signal
import random
import pandas as pd
import numpy as np
from GeoTiff import load_GeoTiff

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

def find_matching_files(root_path: Path, mode: str, included_safe_dirs: set = None) -> list[Path]:
    """
    Walk through `root_path` recursively and return a list of all file Paths
    whose names contain `mode` (e.g. "slc-vh") and end with ".tiff".
    Only include files whose safe_directory is in the inclusion set.
    """
    pattern = f"*{mode}*.tiff"
    
    if included_safe_dirs is None:
        return list(root_path.rglob(pattern))
    
    # Only process files whose safe_directory is in the inclusion set
    filtered_files = []
    for tiff_path in root_path.rglob(pattern):
        safe_dir = tiff_path.parent.parent.name.replace('.SAFE', '')
        if safe_dir in included_safe_dirs:
            filtered_files.append(tiff_path)
    
    return filtered_files

def create_error_row(tiff_path: Path) -> dict:
    """Create a minimal row for files that couldn't be processed."""
    parent_dir = tiff_path.parent.parent.name
    filename_with_ext = tiff_path.name
    return {"safe_directory": parent_dir, "filename": filename_with_ext}

def process_one_file(tiff_path: Path, mode: str, save_array_path: Path) -> dict:
    """
    Process a single TIFF file with optimised statistics computation and robust error handling.
    """
    # Set up signal handler for graceful termination
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Processing timeout for {tiff_path.name}")
    
    # Set a timeout to prevent hanging processes
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5 minute timeout per file
    
    try:
        # Load the GeoTIFF file
        try:
            loaded = load_GeoTiff(str(tiff_path))
        except Exception as load_error:
            print(f"*** Error loading {tiff_path.name}: {load_error}")
            return create_error_row(tiff_path)
        
        if not loaded or loaded[0] is None:
            print(f"*** Warning: load_GeoTiff returned None for {tiff_path.name}. Skipping stats.")
            return create_error_row(tiff_path)

        data = loaded[0]
        
        # Validate the data array
        if data is None or data.size == 0:
            print(f"*** Warning: Empty or invalid data array for {tiff_path.name}")
            return create_error_row(tiff_path)

        if save_array_path:
            # Create output directory if it doesn't exist
            save_array_path.mkdir(parents=True, exist_ok=True)
            out_npy = save_array_path / f"{tiff_path.stem}.npy"
            try:
                np.save(str(out_npy), data)
                print(f"Array saved as: {out_npy}")
            except Exception as e:
                print(f"*** Warning: failed to save array for {tiff_path.name}: {e}")

        # Compute statistics using relevant function
        if "grd" in mode:
            stats_dict = grd_stats(data)
            stats_name = "grd_stats"
        else:
            stats_dict = slc_stats(data)
            stats_name = "slc_stats" 

        # Clean up memory
        del data, loaded
        gc.collect()

        if stats_dict is None:
            print(f"*** Warning: {stats_name} returned None for {tiff_path.name}.")
            return create_error_row(tiff_path)

        # Extract safe directory name and filename separately
        parent_dir = tiff_path.parent.parent.name  # Get the .SAFE directory name
        filename_with_ext = tiff_path.name  # Get filename with extension
        
        # Insert the safe directory and filename into the stats dict
        row = {"safe_directory": parent_dir, "filename": filename_with_ext}
        row.update(stats_dict)
        return row
        
    except (TimeoutError, MemoryError, Exception) as e:
        error_type = type(e).__name__
        print(f"*** {error_type} processing {tiff_path.name}: {e}")
        return create_error_row(tiff_path)
    finally:
        # Clean up and disable alarm
        signal.alarm(0)
        # Force cleanup of any remaining variables
        for var_name in ['data', 'loaded']:
            if var_name in locals() and locals()[var_name] is not None:
                try:
                    del locals()[var_name]
                except:
                    pass
        gc.collect()

def process_files(tiff_files: list[Path], mode: str, save_array_path: Path, 
                  quiet: bool = False) -> list[dict]:
    """
    Process files sequentially.
    """
    rows = []
    total_files = len(tiff_files)
    
    for idx, tiff_path in enumerate(tiff_files, start=1):
        if not quiet:
            print(f"Processing file {idx}/{total_files}: {tiff_path.name}")
            start_time = time.time()
        
        row = process_one_file(tiff_path, mode, save_array_path)
        rows.append(row)

        if not quiet:
            elapsed = time.time() - start_time
            print(f"Finished {tiff_path.name} in {elapsed:.2f} seconds.\n")
    
    return rows

def add_scene_id(df: pd.DataFrame, corr_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Add scene_id column to dataframe using correspondence file.
    Returns the updated dataframe and count of missing mappings.
    """
    # Create mapping dictionary based on mode
    # Determine if we're processing SLC or GRD files
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
        mapping_dict = {}
    
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
            "Search recursively under a root directory for all GeoTIFF files (.tiff suffix) "
            "and whose names contain one of the specified substrings (e.g. 'slc-vh'). "
            "For each such file, load the data, compute statistics, then save the results to "
            "a CSV file."
        )
    )
    parser.add_argument(
        "root_path",
        type=Path,
        help="Path to the root directory under which to search for matching .tiff files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        choices=["slc-vh", "slc-vv", "grd-vh", "grd-vv"],
        default="slc-vh",
        help="Substring to match in filenames (e.g. 'slc-vh')"
    )
    parser.add_argument(
        "--save-array",
        type=Path,
        help="Path to directory where .npy files will be saved. If not provided, arrays are not saved."
    )
    parser.add_argument(
        "--correspondence-file",
        type=Path,
        default="xView3_SLC_GRD_correspondences.csv",
        help="Path to the correspondence CSV file for scene_id mapping (default: correspondences.csv)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-file progress messages; only warnings and final summary are shown"
    )
    parser.add_argument(
        "--include-scene-ids",
        type=Path,
        help="Path to a text file containing scene_ids to include (one per line). If not provided, all files are processed."
    )
   
    args = parser.parse_args()
    root_path = args.root_path
    mode = args.pattern
    save_array_path = args.save_array
    correspondence_file = args.correspondence_file
    quiet = args.quiet

    if not root_path.is_dir():
        parser.error(f"Provided root_path ({root_path}) is not a directory or does not exist.")

    # Load inclusion list if provided
    included_scene_ids = None
    if args.include_scene_ids:
        if args.include_scene_ids.exists():
            try:
                with open(args.include_scene_ids, 'r') as f:
                    included_scene_ids = {line.strip() for line in f if line.strip()}
                if not quiet:
                    print(f"Loaded {len(included_scene_ids)} scene IDs to include")
            except Exception as e:
                print(f"Warning: Error reading inclusion file: {e}")
                included_scene_ids = None
        else:
            print(f"Warning: Inclusion file '{args.include_scene_ids}' not found")

    # Convert included scene_ids to safe_directory names for filtering
    included_safe_dirs = None
    if included_scene_ids and correspondence_file.exists():
        try:
            corr_df = pd.read_csv(correspondence_file)
            included_safe_dirs = set()
            
            # Map scene_ids to safe_directory names
            for _, row in corr_df.iterrows():
                scene_id = row['scene_id']
                if scene_id in included_scene_ids:
                    if 'SLC_product_identifier' in corr_df.columns and pd.notna(row['SLC_product_identifier']):
                        safe_name = row['SLC_product_identifier'].replace('.SAFE', '')
                        included_safe_dirs.add(safe_name)
                    if 'GRD_product_identifier' in corr_df.columns and pd.notna(row['GRD_product_identifier']):
                        safe_name = row['GRD_product_identifier'].replace('.SAFE', '')
                        included_safe_dirs.add(safe_name)
                        
            if not quiet:
                print(f"Mapped to {len(included_safe_dirs)} safe directories to include")
        except Exception as e:
            print(f"Warning: Error loading correspondence file for filtering: {e}")
            included_safe_dirs = None

    print("Processing...")

    # Find all matching files
    if not quiet:
        filter_msg = f" (filtered by inclusion list)" if included_safe_dirs else ""
        print(f"Searching for files matching '*{mode}*.tiff' under {root_path}{filter_msg}")
    
    tiff_files = find_matching_files(root_path, mode, included_safe_dirs)

    if not tiff_files:
        filter_msg = " matching inclusion criteria" if included_safe_dirs else ""
        print(f"No files{filter_msg} matching '*{mode}*.tiff' found under {root_path}")
        return

    total_files = len(tiff_files)
    if not quiet:
        print(f"Found {total_files} matching files")

    # Process files
    start_time = time.time()
    rows = process_files(tiff_files, mode, save_array_path, quiet)
    processing_time = time.time() - start_time

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
    
    # Load correspondence table and add scene_id (reuse existing dataframe if already loaded)
    try:
        if correspondence_file.exists():
            if 'corr_df' not in locals():
                corr_df = pd.read_csv(correspondence_file)
            df_with_scene, missing_count = add_scene_id(df, corr_df)
        else:
            print(f"Warning: Correspondence file '{correspondence_file}' not found. Proceeding without scene_id mapping.")
            df_with_scene = df
            missing_count = len(df)
    except Exception as e:
        print(f"Warning: Error processing correspondence file '{correspondence_file}': {e}. Proceeding without scene_id mapping.")
        df_with_scene = df  
        missing_count = len(df)

    # Save the DataFrame as a CSV file
    output_csv = Path.cwd() / f"{mode}_stats.csv"
    df_with_scene.to_csv(output_csv, index=False)

    print(f"Stats for {len(df)} file(s) written to: {output_csv}")
    if 'missing_count' in locals() and missing_count > 0:
        print(f"Note: {missing_count} files could not be mapped to scene_id")
    print(f"Total processing time: {processing_time:.2f} seconds")
    if total_files > 0:
        print(f"Average time per file: {processing_time/total_files:.2f} seconds")


if __name__ == "__main__":
    main()
