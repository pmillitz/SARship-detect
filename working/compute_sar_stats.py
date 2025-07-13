#!/usr/bin/env python3
"""
compute_sar_stats.py

Author:
Created: 2025-07-12

usage: compute_sar_stats.py [-h] [-j JOBS] [-q(uiet)] input_path

Reads NumPy arrays from the specified input directory and computes statistics
for SAR data. The input directory should contain .npy files created by
unload_sar_data.py along with a manifest.json file. Statistics are saved
to a CSV file in the current working directory.
"""

import argparse
import time
import json
import socket
import os
from pathlib import Path
import gc
import multiprocessing as mp
from functools import partial
import pandas as pd
import numpy as np
import psutil

def detect_environment():
    """
    Detect if running in Jupyter notebook environment.
    """
    try:
        # Check if we're in a Jupyter environment
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return 'jupyter'
    except NameError:
        pass
    return 'script'

def get_optimal_workers(user_specified: int = None) -> tuple[int, str, str]:
    """
    Determine optimal number of workers based on system and environment.
    Returns: (workers, system, environment)
    """
    # Detect environment first
    environment = detect_environment()
    
    # Determine the current host
    hostname = socket.gethostname()
    if "kaya" in hostname.lower() or os.getenv("HPC_ENV") == "true":
        system = "kaya"
    else:
        system = "local"
    
    # Get available CPU cores
    cpu_count = mp.cpu_count()
    
    # Environment-specific handling
    if environment == 'jupyter':
        # Jupyter notebooks have multiprocessing issues - use conservative approach
        if user_specified is not None and user_specified > 1:
            print("Warning: Multiprocessing in Jupyter can be unstable. Consider using -j 1 for sequential processing.")
            default_workers = min(user_specified, 4)  # Cap at 4 workers max in Jupyter
        else:
            default_workers = 1  # Default to sequential in Jupyter
    else:
        # Script environment - full multiprocessing support
        if system == "kaya":
            # HPC environment - can use more aggressive parallelization
            slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK')
            if slurm_cpus:
                default_workers = int(slurm_cpus)
            else:
                default_workers = max(1, cpu_count - 1)  # Leave one core free
        else:
            # Local system - more conservative
            default_workers = max(1, min(cpu_count // 2, 8))  # Use half cores, max 8
    
    # Use user specification if provided, otherwise use system default
    if user_specified is not None:
        workers = min(user_specified, cpu_count)
    else:
        workers = default_workers
    
    return workers, system, environment

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

def process_one_array(array_path: Path, file_info: dict, mode: str) -> dict:
    """
    Process a single NumPy array file with optimised statistics computation.
    """
    try:
        # Load the NumPy array
        try:
            data = np.load(str(array_path), allow_pickle=False)
            print(f"Loaded {array_path.name}: shape={data.shape}, size={data.size}, dtype={data.dtype}")
        except Exception as load_error:
            print(f"*** Error loading {array_path.name}: {load_error}")
            return create_error_row(file_info)
        
        # Validate the data array
        if data is None or data.size == 0:
            print(f"*** Warning: Empty or invalid data array for {array_path.name}")
            return create_error_row(file_info)

        # Check array shape and fix if needed
        if data.ndim == 1:
            # Try to determine original shape from file size and data type
            print(f"*** Warning: 1D array detected for {array_path.name} with {data.size} elements")
            
            total_elements = data.size
            
            # Calculate possible square dimensions
            sqrt_elements = int(np.sqrt(total_elements))
            
            # Try common SAR image dimensions
            possible_shapes = []
            
            # Perfect square
            if sqrt_elements * sqrt_elements == total_elements:
                possible_shapes.append((sqrt_elements, sqrt_elements))
            
            # Try factorization for rectangular shapes
            # Look for factors close to square root
            for width in range(max(1000, sqrt_elements - 1000), sqrt_elements + 1000):
                if total_elements % width == 0:
                    height = total_elements // width
                    if abs(width - height) < max(width, height) * 0.5:  # Aspect ratio not too extreme
                        possible_shapes.append((height, width))
            
            # Add some common SAR dimensions if they fit
            common_widths = [25000, 20000, 15000, 10000, 5000]
            for width in common_widths:
                if total_elements % width == 0:
                    height = total_elements // width
                    possible_shapes.append((height, width))
            
            print(f"Possible shapes for {total_elements} elements: {possible_shapes[:5]}")  # Show first 5
            
            if possible_shapes:
                new_shape = possible_shapes[0]  # Use the first valid shape
                try:
                    data = data.reshape(new_shape)
                    print(f"Successfully reshaped {array_path.name} to {new_shape}")
                except Exception as reshape_error:
                    print(f"*** Error reshaping {array_path.name}: {reshape_error}")
                    return create_error_row(file_info)
            else:
                print(f"*** Error: Cannot determine valid shape for {array_path.name} with {total_elements} elements")
                return create_error_row(file_info)
        
        elif data.ndim > 2:
            # Handle multi-dimensional arrays by flattening to 2D
            print(f"*** Warning: {data.ndim}D array detected for {array_path.name} with shape {data.shape}")
            original_shape = data.shape
            try:
                # Reshape to 2D by combining all but the last dimension
                if data.ndim == 3:
                    data = data.reshape(original_shape[0] * original_shape[1], original_shape[2])
                else:
                    data = data.reshape(original_shape[0], -1)
                print(f"Reshaped from {original_shape} to {data.shape}")
            except Exception as reshape_error:
                print(f"*** Error reshaping multi-dimensional array {array_path.name}: {reshape_error}")
                return create_error_row(file_info)
        else:
            # Already 2D - show dimensions for confirmation
            print(f"2D array: {data.shape}")

        # Ensure we have a 2D array
        if data.ndim != 2:
            print(f"*** Error: Final array for {array_path.name} is not 2D (shape: {data.shape})")
            return create_error_row(file_info)

        # Additional validation for data type
        if "slc" in mode.lower() and not np.iscomplexobj(data):
            print(f"*** Warning: SLC data expected to be complex, but got {data.dtype} for {array_path.name}")
        elif "grd" in mode.lower() and np.iscomplexobj(data):
            print(f"*** Warning: GRD data expected to be real, but got complex data for {array_path.name}")
            # Convert complex to real by taking magnitude
            data = np.abs(data)
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

def process_one_array_wrapper(args):
    """
    Wrapper function for multiprocessing. Unpacks arguments and calls process_one_array.
    """
    file_info, mode, quiet = args
    array_path = Path(file_info['array_path'])
    
    # Double-check file exists (should have been filtered already)
    if not array_path.exists():
        if not quiet:
            print(f"*** Warning: Array file not found during processing: {array_path}")
        return create_error_row(file_info)
    
    return process_one_array(array_path, file_info, mode)

def process_arrays(manifest_data: dict, input_path: Path, quiet: bool = False, n_workers: int = None) -> list[dict]:
    """
    Process all arrays listed in the manifest using parallel processing with timeout protection.
    """
    files_info = manifest_data.get('files', [])
    mode = manifest_data.get('pattern', 'slc-vh')
    
    # Filter for successfully unloaded files AND check if they actually exist
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
        return []
    
    if missing_files > 0:
        print(f"Found {total_files} valid arrays ({missing_files} missing from manifest)")
    
    # Determine optimal number of workers using host detection
    optimal_workers, system, environment = get_optimal_workers(n_workers)
    
    # For large files (>1GB), be more conservative with workers
    sample_file = Path(successful_files[0]['array_path'])
    if sample_file.exists():
        file_size_gb = sample_file.stat().st_size / (1024**3)
        if file_size_gb > 1.0:
            # Reduce workers for large files to prevent memory issues
            memory_limited_workers = max(1, min(optimal_workers, 4))
            if memory_limited_workers < optimal_workers:
                print(f"Large files detected ({file_size_gb:.1f}GB), reducing workers from {optimal_workers} to {memory_limited_workers}")
                optimal_workers = memory_limited_workers
    
    if not quiet:
        print(f"Detected system: {system}")
        print(f"Environment: {environment}")
        if environment == 'jupyter' and optimal_workers > 1:
            print("Note: Running in Jupyter - multiprocessing may be less stable")
        if n_workers is not None:
            print(f"Using user-specified {optimal_workers} workers")
        else:
            print(f"Using auto-detected {optimal_workers} workers")
        print(f"Processing {total_files} arrays")
    
    # Prepare arguments for parallel processing - only for existing files
    args_list = [(file_info, mode, quiet) for file_info in successful_files]
    
    if optimal_workers == 1:
        # Sequential processing
        rows = []
        for idx, args in enumerate(args_list, start=1):
            if not quiet:
                print(f"Processing array {idx}/{total_files}: {Path(args[0]['array_path']).name}")
                start_time = time.time()
            
            row = process_one_array_wrapper(args)
            rows.append(row)
            
            if not quiet:
                elapsed = time.time() - start_time
                print(f"Finished {Path(args[0]['array_path']).name} in {elapsed:.2f} seconds.\n")
    else:
        # Parallel processing with timeout and progress monitoring
        import signal
        import threading
        import time as time_module
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Processing timeout - workers may be stuck")
        
        rows = []
        try:
            with mp.Pool(processes=optimal_workers) as pool:
                if not quiet:
                    print("Starting parallel processing...")
                
                # Set up shorter timeout for testing (5 minutes instead of 30)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)  # 5 minute timeout for debugging
                
                # Use map_async for better control and immediate feedback
                progress_interval = max(1, total_files // 20)  # Update every 5%
                completed = 0
                start_time = time_module.time()
                
                try:
                    # Submit all jobs and track them
                    print(f"Submitting {len(args_list)} jobs to {optimal_workers} workers...")
                    result = pool.map_async(process_one_array_wrapper, args_list, chunksize=1)
                    
                    # Poll for completion with timeout
                    while not result.ready():
                        time_module.sleep(10)  # Check every 10 seconds
                        elapsed = time_module.time() - start_time
                        print(f"Still processing... {elapsed:.0f}s elapsed (timeout in {300-elapsed:.0f}s)")
                        
                        # Check memory usage
                        memory_percent = psutil.virtual_memory().percent
                        if memory_percent > 85:
                            print(f"Warning: High memory usage ({memory_percent:.1f}%)")
                    
                    # Get results
                    print("Workers completed, collecting results...")
                    rows = result.get(timeout=60)  # 1 minute to collect results
                    print(f"Successfully collected {len(rows)} results")
                
                except mp.TimeoutError:
                    print("*** Timeout waiting for worker results")
                    print("This suggests workers are stuck - falling back to sequential processing")
                    # Don't return empty results, fall through to sequential fallback
                    rows = []
                
                except Exception as e:
                    print(f"Error during parallel processing: {e}")
                    print("Attempting to continue with completed results...")
                    rows = []
                
                finally:
                    signal.alarm(0)  # Cancel timeout
                    
        except TimeoutError:
            print("Processing timed out after 5 minutes")
            print(f"Workers appear to be stuck - this often happens with large arrays in multiprocessing")
            print("Falling back to sequential processing...")
            rows = []
        except Exception as e:
            print(f"Parallel processing failed: {e}")
            print("Falling back to sequential processing...")
            rows = []
        
        # If parallel processing failed or returned no results, fall back to sequential
        if not rows:
            print("\n=== FALLING BACK TO SEQUENTIAL PROCESSING ===")
            print("This is more reliable for large arrays but will be slower...")
            
            for idx, args in enumerate(args_list, start=1):
                if not quiet:
                    print(f"\nProcessing array {idx}/{total_files}: {Path(args[0]['array_path']).name}")
                    start_time = time_module.time()
                
                row = process_one_array_wrapper(args)
                rows.append(row)
                
                if not quiet:
                    elapsed = time_module.time() - start_time
                    print(f"Completed in {elapsed:.2f} seconds")
                    
                    # Show progress
                    progress = idx / total_files * 100
                    print(f"Progress: {idx}/{total_files} ({progress:.1f}%)")
    
    return rows

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
            "Process NumPy arrays created by unload_sar_data.py and compute statistics. "
            "The input directory should contain .npy files and a manifest.json file."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the directory containing .npy files and manifest.json"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect based on CPU cores)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-file progress messages; only warnings and final summary are shown"
    )
   
    args = parser.parse_args()
    input_path = args.input_path
    n_workers = args.jobs
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

    # Process arrays
    start_time = time.time()
    rows = process_arrays(manifest_data, input_path, quiet, n_workers)
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

    print(f"Stats for {len(df)} array(s) written to: {output_csv}")
    if missing_count > 0:
        print(f"Note: {missing_count} files could not be mapped to scene_id")
    print(f"Total processing time: {processing_time:.2f} seconds")
    if len(rows) > 0:
        print(f"Average time per array: {processing_time/len(rows):.2f} seconds")
    
    # Show worker info in final summary
    optimal_workers, system, environment = get_optimal_workers(n_workers)
    if optimal_workers > 1:
        print(f"Used {optimal_workers} parallel workers on {system} system ({environment} environment)")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    main()
