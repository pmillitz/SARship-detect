#!/usr/bin/env python3
"""
compute_sar_stats.py

Author: Peter Millitz
Date: 06/06/2025

usage: compute_sar_stats.py [-h]
                            [--pattern {slc-vh, slc-vv, grd-vh, grd-vv}]
                            [--save-array] [-q(uiet)] [--validate-files]
                            root_path 

Recursively searches for all GeoTIFF files under a given root directory whose filename
contains any one pattern in ["slc-vh", "slc-vv", "grd-vh", "grd-vv"] and ends with ".tiff".
For each matching file, the SLC or GRD data is unpacked into a NumPy array and a range of
statistics computed. Optionally, the extracted array can be saved for later use.  Each tiff
file's statistics are inserted as a new row into a dataframe which is saved to a CSV file
named "{pattern}_stats.csv", on exit.
"""

import argparse
import time
from pathlib import Path
import gc
import os
import signal

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

def find_matching_files(root_path: Path, mode: str) -> list[Path]:
    """
    Walk through `root_path` recursively and return a list of all file Paths
    whose names contain `mode` (e.g. "slc-vh") and end with ".tiff".
    """
    pattern = f"*{mode}*.tiff"
    return list(root_path.rglob(pattern))

def process_one_file(tiff_path: Path, mode: str, save_array: bool) -> dict:
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
        # Load the GeoTIFF file with additional error checking
        loaded = None
        data = None
        
        try:
            loaded = load_GeoTiff(str(tiff_path))
        except Exception as load_error:
            print(f"*** Error loading {tiff_path.name}: {load_error}")
            # Create separate safe_directory and filename for error cases
            parent_dir = tiff_path.parent.parent.name
            filename_with_ext = tiff_path.name
            return {"safe_directory": parent_dir, "filename": filename_with_ext}
        
        if not loaded or loaded[0] is None:
            print(f"*** Warning: load_GeoTiff returned None for {tiff_path.name}. Skipping stats.")
            # Create separate safe_directory and filename for error cases
            parent_dir = tiff_path.parent.parent.name
            filename_with_ext = tiff_path.name
            return {"safe_directory": parent_dir, "filename": filename_with_ext}

        data = loaded[0]
        
        # Validate the data array
        if data is None or data.size == 0:
            print(f"*** Warning: Empty or invalid data array for {tiff_path.name}")
            # Create separate safe_directory and filename for error cases
            parent_dir = tiff_path.parent.parent.name
            filename_with_ext = tiff_path.name
            return {"safe_directory": parent_dir, "filename": filename_with_ext}

        if save_array:
            out_npy = Path.cwd() / f"{tiff_path.stem}.npy"
            try:
                np.save(str(out_npy), data)
                print(f"Array saved as: {out_npy.name}")
            except Exception as e:
                print(f"*** Warning: failed to save array for {tiff_path.name}: {e}")

        # Compute statistics using relevant function
        stats_dict = None
        if "grd" in mode:
            stats_dict = grd_stats(data)
            stats_name = "grd_stats"
        else:
            stats_dict = slc_stats(data)
            stats_name = "slc_stats" 

        # Delete the current input array to free memory immediately
        del data
        del loaded
        gc.collect()  # Force garbage collection

        if stats_dict is None:
            print(f"*** Warning: {stats_name} returned None for {tiff_path.name}.")
            # Create separate safe_directory and filename for error cases
            parent_dir = tiff_path.parent.parent.name
            filename_with_ext = tiff_path.name
            return {"safe_directory": parent_dir, "filename": filename_with_ext}

        # Extract safe directory name and filename separately
        parent_dir = tiff_path.parent.parent.name  # Get the .SAFE directory name
        filename_with_ext = tiff_path.name  # Get filename with extension
        
        # Insert the safe directory and filename into the stats dict
        row = {"safe_directory": parent_dir, "filename": filename_with_ext}
        row.update(stats_dict)
        return row
        
    except TimeoutError as e:
        print(f"*** Timeout error processing {tiff_path.name}: {e}")
        # Create separate safe_directory and filename for error cases
        parent_dir = tiff_path.parent.parent.name
        filename_with_ext = tiff_path.name
        return {"safe_directory": parent_dir, "filename": filename_with_ext}
    except MemoryError as e:
        print(f"*** Memory error processing {tiff_path.name}: {e}")
        # Create separate safe_directory and filename for error cases
        parent_dir = tiff_path.parent.parent.name
        filename_with_ext = tiff_path.name
        return {"safe_directory": parent_dir, "filename": filename_with_ext}
    except Exception as e:
        print(f"*** Error processing {tiff_path.name}: {e}")
        # Create separate safe_directory and filename for error cases
        parent_dir = tiff_path.parent.parent.name
        filename_with_ext = tiff_path.name
        return {"safe_directory": parent_dir, "filename": filename_with_ext}
    finally:
        # Clean up and disable alarm
        signal.alarm(0)
        try:
            if 'data' in locals() and data is not None:
                del data
            if 'loaded' in locals() and loaded is not None:
                del loaded
            gc.collect()
        except:
            pass

def process_files(tiff_files: list[Path], mode: str, save_array: bool, 
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
        
        row = process_one_file(tiff_path, mode, save_array)
        rows.append(row)

        if not quiet:
            elapsed = time.time() - start_time
            print(f"Finished {tiff_path.name} in {elapsed:.2f} seconds.\n")
    
    return rows

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
        action="store_true",
        help="If set, save each loaded array to a .npy file named <tiff_stem>.npy"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-file progress messages; only warnings and final summary are shown"
    )
    parser.add_argument(
        "--validate-files",
        action="store_true",
        help="Pre-validate TIFF files before processing (slower but more robust)"
    )
    
    args = parser.parse_args()
    root_path = args.root_path
    mode = args.pattern
    save_array = args.save_array
    quiet = args.quiet
    validate_files = args.validate_files

    if not root_path.is_dir():
        parser.error(f"Provided root_path ({root_path}) is not a directory or does not exist.")

    print("Processing...")

    # Find all matching files
    if not quiet:
        print(f"Searching for files matching '*{mode}*.tiff' under {root_path}")
    
    tiff_files = find_matching_files(root_path, mode)

    if not tiff_files:
        print(f"No files matching '*{mode}*.tiff' found under {root_path}")
        return

    total_files = len(tiff_files)
    if not quiet:
        print(f"Found {total_files} matching files")

    # Pre-validate files if requested
    if validate_files:
        if not quiet:
            print("Pre-validating TIFF files...")
        valid_files = []
        for tiff_path in tiff_files:
            try:
                # Quick validation - try to open without loading full data
                from osgeo import gdal
                gdal.UseExceptions()
                ds = gdal.Open(str(tiff_path))
                if ds is not None:
                    valid_files.append(tiff_path)
                    ds = None
                else:
                    print(f"*** Skipping invalid file: {tiff_path.name}")
            except Exception as e:
                print(f"*** Skipping corrupted file {tiff_path.name}: {e}")
        
        tiff_files = valid_files
        if not quiet:
            print(f"Pre-validation complete: {len(tiff_files)}/{total_files} files are valid")
        total_files = len(tiff_files)

    # Process files
    start_time = time.time()
    rows = process_files(tiff_files, mode, save_array, quiet)
    processing_time = time.time() - start_time

    # Build pandas DataFrame with predefined columns (safe_directory added as first column)
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

    # Save the DataFrame as a CSV file
    output_csv = Path.cwd() / f"{mode}_stats.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"Stats for {len(df)} file(s) written to: {output_csv}")
    print(f"Total processing time: {processing_time:.2f} seconds")
    if total_files > 0:
        print(f"Average time per file: {processing_time/total_files:.2f} seconds")


if __name__ == "__main__":
    main()

