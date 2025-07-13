#!/usr/bin/env python3
"""
unload_sar_data.py

Author: Peter Millitz
Created: 2025-07-12

usage: unload_sar_data.py [-h]
                          [--pattern {slc-vh, slc-vv, grd-vh, grd-vv}]
                          [--correspondence-file PATH] 
                          [--include-scene-ids PATH]
                          [--overwrite]
                          [-q(uiet)] 
                          root_path output_path

Recursively searches for all GeoTIFF files under a given root directory whose filename
contains any one pattern in ["slc-vh", "slc-vv", "grd-vh", "grd-vv"] and ends with ".tiff".
For each matching file, the SLC or GRD data is unpacked into a NumPy array and saved
to the specified output directory. A manifest file is created to track the mapping
between original TIFF files and saved arrays.
"""

import argparse
import time
import json
from pathlib import Path
import gc
import os
import signal
import pandas as pd
import numpy as np
from GeoTiff import load_GeoTiff

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

def unload_one_file(tiff_path: Path, output_path: Path, quiet: bool = False) -> dict:
    """
    Unload a single TIFF file to NumPy array with robust error handling.
    Returns a dictionary with metadata about the operation.
    """
    # Set up signal handler for graceful termination
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Processing timeout for {tiff_path.name}")
    
    # Set a timeout to prevent hanging processes
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5 minute timeout per file
    
    try:
        # Create output filename
        out_npy = output_path / f"{tiff_path.stem}.npy"
        
        # Check if file already exists and skip if it does
        if out_npy.exists():
            if not quiet:
                print(f"Skipping {tiff_path.name} - array already exists")
            return {
                'tiff_path': str(tiff_path),
                'array_path': str(out_npy),
                'safe_directory': tiff_path.parent.parent.name,
                'filename': tiff_path.name,
                'status': 'skipped',
                'error': None
            }
        
        # Load the GeoTIFF file
        try:
            loaded = load_GeoTiff(str(tiff_path))
        except Exception as load_error:
            print(f"*** Error loading {tiff_path.name}: {load_error}")
            return {
                'tiff_path': str(tiff_path),
                'array_path': str(out_npy),
                'safe_directory': tiff_path.parent.parent.name,
                'filename': tiff_path.name,
                'status': 'error',
                'error': str(load_error)
            }
        
        if not loaded or loaded[0] is None:
            error_msg = f"load_GeoTiff returned None for {tiff_path.name}"
            print(f"*** Warning: {error_msg}")
            return {
                'tiff_path': str(tiff_path),
                'array_path': str(out_npy),
                'safe_directory': tiff_path.parent.parent.name,
                'filename': tiff_path.name,
                'status': 'error',
                'error': error_msg
            }

        data = loaded[0]
        
        # Validate the data array
        if data is None or data.size == 0:
            error_msg = f"Empty or invalid data array for {tiff_path.name}"
            print(f"*** Warning: {error_msg}")
            return {
                'tiff_path': str(tiff_path),
                'array_path': str(out_npy),
                'safe_directory': tiff_path.parent.parent.name,
                'filename': tiff_path.name,
                'status': 'error',
                'error': error_msg
            }

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the array
        try:
            np.save(str(out_npy), data)
            print(f"Array saved as: {out_npy}")
        except Exception as e:
            error_msg = f"Failed to save array for {tiff_path.name}: {e}"
            print(f"*** Error: {error_msg}")
            return {
                'tiff_path': str(tiff_path),
                'array_path': str(out_npy),
                'safe_directory': tiff_path.parent.parent.name,
                'filename': tiff_path.name,
                'status': 'error',
                'error': error_msg
            }

        # Clean up memory
        del data, loaded
        gc.collect()

        return {
            'tiff_path': str(tiff_path),
            'array_path': str(out_npy),
            'safe_directory': tiff_path.parent.parent.name,
            'filename': tiff_path.name,
            'status': 'success',
            'error': None
        }
        
    except (TimeoutError, MemoryError, Exception) as e:
        error_type = type(e).__name__
        error_msg = f"{error_type} processing {tiff_path.name}: {e}"
        print(f"*** {error_msg}")
        return {
            'tiff_path': str(tiff_path),
            'array_path': str(out_npy) if 'out_npy' in locals() else '',
            'safe_directory': tiff_path.parent.parent.name,
            'filename': tiff_path.name,
            'status': 'error',
            'error': error_msg
        }
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

def process_files(tiff_files: list[Path], output_path: Path, quiet: bool = False) -> list[dict]:
    """
    Process files sequentially.
    """
    results = []
    total_files = len(tiff_files)
    
    for idx, tiff_path in enumerate(tiff_files, start=1):
        if not quiet:
            print(f"Processing file {idx}/{total_files}: {tiff_path.name}")
            start_time = time.time()
        
        result = unload_one_file(tiff_path, output_path, quiet)
        results.append(result)

        if not quiet:
            elapsed = time.time() - start_time
            print(f"Finished {tiff_path.name} in {elapsed:.2f} seconds.\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Search recursively under a root directory for all GeoTIFF files (.tiff suffix) "
            "whose names contain one of the specified substrings (e.g. 'slc-vh'). "
            "For each such file, load the data and save as NumPy array to the output directory."
        )
    )
    parser.add_argument(
        "root_path",
        type=Path,
        help="Path to the root directory under which to search for matching .tiff files"
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to the output directory where .npy files will be saved"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        choices=["slc-vh", "slc-vv", "grd-vh", "grd-vv"],
        default="slc-vh",
        help="Substring to match in filenames (e.g. 'slc-vh')"
    )
    parser.add_argument(
        "--correspondence-file",
        type=Path,
        default="xView3_SLC_GRD_correspondences.csv",
        help="Path to the correspondence CSV file for scene_id mapping (default: xView3_SLC_GRD_correspondences.csv)"
    )
    parser.add_argument(
        "--include-scene-ids",
        type=Path,
        help="Path to a text file containing scene_ids to include (one per line). If not provided, all files are processed."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-file progress messages; only warnings and final summary are shown"
    )
   
    args = parser.parse_args()
    root_path = args.root_path
    output_path = args.output_path
    mode = args.pattern
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
    results = process_files(tiff_files, output_path, quiet)
    processing_time = time.time() - start_time

    # Create and save manifest
    manifest_path = output_path / "manifest.json"
    manifest_data = {
        'pattern': mode,
        'root_path': str(root_path),
        'output_path': str(output_path),
        'correspondence_file': str(correspondence_file) if correspondence_file.exists() else None,
        'total_files': total_files,
        'processing_time': processing_time,
        'files': results
    }
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        if not quiet:
            print(f"Manifest saved to: {manifest_path}")
    except Exception as e:
        print(f"Warning: Could not save manifest: {e}")

    # Summary statistics
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')

    print(f"\nSummary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Successfully unloaded: {success_count}")
    print(f"  Skipped (already exist): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total processing time: {processing_time:.2f} seconds")
    if total_files > 0:
        print(f"  Average time per file: {processing_time/total_files:.2f} seconds")

    if error_count > 0:
        print(f"\nFiles with errors:")
        for result in results:
            if result['status'] == 'error':
                print(f"  {result['filename']}: {result['error']}")


if __name__ == "__main__":
    main()
