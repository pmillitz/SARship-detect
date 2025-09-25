#!/usr/bin/env python3

"""
complex_scale_and_norm_ext_png.py

Author: Peter Millitz
Created: 2025-09-24

Takes as input either a single .npy file or a directory containing .npy files.
Processes all complex-valued 2D SAR SLC products sequentially. Outputs arrays
with shape (H, W, 3) and converts the data to uint8 PNG format where:
- Red channel: Extracted amplitude (clipped, dB-scaled, normalized to [0,1])
- Green channel: sin(phase) normalized to [0,1]
- Blue channel: cos(phase) normalized to [0,1]

This extended version preserves more phase information by encoding both sine and
cosine components, providing richer representation for ML models.

usage:
    # Process single file
    python complex_scale_and_norm_ext_png.py --input-dir input_array.npy --output-dir /results/processed/

    # Process all .npy files in a directory
    python complex_scale_and_norm_ext_png.py --input-dir images/ --output-dir /results/images/

    # Directory processing with amplitude clipping
    python complex_scale_and_norm_ext_png.py --input-dir images_hvrt_bal/ --output-dir /results/images_hvrt_bal/ --amp-clip-params 1.00 71.51
"""

import numpy as np
import argparse
import os
from pathlib import Path
import sys
import shutil
from datetime import datetime
import cv2


def process_complex_data(slc_data, epsilon=1e-6, nan_strategy='skip', verbose=False, amp_clip_params=None):
    """
    Processes a single complex-valued 2D SAR SLC data array. Extracts amplitude and
    phase components, applies scaling and normalisation, then outputs RGB uint8 format.

    Phase information is encoded as:
    - Green channel: sin(phase) normalized to [0,1]
    - Blue channel: cos(phase) normalized to [0,1]

    Returns:
    --------
    numpy.ndarray
        3D array of shape (H, W, 3) with uint8 RGB format
        Returns None if nan_strategy='skip' and NaNs are present
    """
    
    # Handle NaN/invalid values
    valid_mask = np.isfinite(slc_data)
    nan_count = slc_data.size - np.sum(valid_mask)
    
    if nan_count > 0:
        if verbose:
            print(f"Found {nan_count} invalid values ({nan_count/slc_data.size*100:.1f}%)")
        
        if nan_strategy == 'skip':
            if verbose:
                print("Skipping sample due to NaN values")
            return None
        elif nan_strategy == 'zero':
            # In-place modification to avoid copying
            slc_data[~valid_mask] = 0 + 0j
        elif nan_strategy == 'mean':
            # In-place modification to avoid copying
            mean_val = np.mean(slc_data[valid_mask])
            slc_data[~valid_mask] = mean_val
        elif nan_strategy == 'interpolate':
            # Memory-efficient interpolation using chunked processing
            slc_data = _interpolate_chunked(slc_data, valid_mask, verbose=verbose)
    
    # Extract amplitude clipping parameters if provided
    if amp_clip_params is not None:
        amp_min, amp_max = amp_clip_params
        use_amp_clipping = True
        if verbose:
            print(f"Using amplitude clipping parameters: [{amp_min:.6f}, {amp_max:.6f}]")
    else:
        use_amp_clipping = False
        amp_min = amp_max = None
    
    # Direct RGB construction
    H, W = slc_data.shape
    rgb_data = np.zeros((H, W, 3), dtype=np.float32)
    
    # Process amplitude with clipping before dB conversion
    amplitude = np.abs(slc_data)
    
    if use_amp_clipping:
        # Clip amplitude in linear space before dB conversion
        np.clip(amplitude, amp_min, amp_max, out=amplitude)
        
        # Convert to dB
        np.log10(amplitude + epsilon, out=amplitude)
        amplitude *= 20
        
        # Normalize using clipped bounds
        amp_db_min = 20 * np.log10(amp_min + epsilon)
        amp_db_max = 20 * np.log10(amp_max + epsilon)
        amplitude = (amplitude - amp_db_min) / (amp_db_max - amp_db_min)
        
        if verbose:
            print(f"Clipped amplitude range: [{amp_min:.6f}, {amp_max:.6f}]")
            print(f"dB range: [{amp_db_min:.3f}, {amp_db_max:.3f}]")
    else:
        # Adaptive normalization
        np.log10(amplitude + epsilon, out=amplitude)
        amplitude *= 20
        amp_min_db = amplitude.min()
        amp_max_db = amplitude.max()
        amplitude = (amplitude - amp_min_db) / (amp_max_db - amp_min_db)
        
        if verbose:
            print(f"Adaptive amplitude range (dB): [{amp_min_db:.3f}, {amp_max_db:.3f}]")
    
    # Enhanced phase processing with sin/cos components
    phase = np.angle(slc_data)  # Returns (-π, π]

    # Compute sin and cos of phase, then normalize to [0, 1]
    sin_phase = np.sin(phase)  # Range: [-1, 1]
    cos_phase = np.cos(phase)  # Range: [-1, 1]

    # Normalize sin and cos to [0, 1] range
    sin_phase_norm = (sin_phase + 1) / 2  # Map [-1, 1] to [0, 1]
    cos_phase_norm = (cos_phase + 1) / 2  # Map [-1, 1] to [0, 1]

    # Assign to RGB channels
    rgb_data[:, :, 0] = amplitude      # Red channel: amplitude
    rgb_data[:, :, 1] = sin_phase_norm # Green channel: sin(phase) normalized
    rgb_data[:, :, 2] = cos_phase_norm # Blue channel: cos(phase) normalized
    
    # Convert to uint8
    rgb_data = (rgb_data * 255).astype(np.uint8)
    
    return rgb_data

def _interpolate_chunked(slc_data, valid_mask, chunk_size=1000, verbose=False):
    """
    Memory-efficient chunked interpolation for large arrays.
    """
    try:
        from scipy.interpolate import griddata
    except ImportError:
        print("Error: scipy is required for interpolation. Install with: pip install scipy")
        sys.exit(1)
    
    if verbose:
        print(f"Using chunked interpolation with chunk_size={chunk_size}")
    
    # Create a copy only for interpolation (unavoidable for this strategy)
    slc_data_clean = slc_data.copy()
    
    h, w = slc_data.shape
    invalid_indices = np.where(~valid_mask)
    
    if len(invalid_indices[0]) == 0:
        return slc_data_clean
    
    # Get valid points coordinates (memory efficient)
    valid_indices = np.where(valid_mask)
    valid_points = np.column_stack(valid_indices)
    valid_values_real = slc_data[valid_mask].real
    valid_values_imag = slc_data[valid_mask].imag
    
    # Process invalid points in chunks to manage memory
    invalid_points = np.column_stack(invalid_indices)
    n_invalid = len(invalid_points)
    
    for start_idx in range(0, n_invalid, chunk_size):
        end_idx = min(start_idx + chunk_size, n_invalid)
        chunk_points = invalid_points[start_idx:end_idx]
        
        # Interpolate real and imaginary parts for this chunk
        real_interp = griddata(valid_points, valid_values_real, 
                             chunk_points, method='linear', fill_value=0)
        imag_interp = griddata(valid_points, valid_values_imag, 
                             chunk_points, method='linear', fill_value=0)
        
        # Update the invalid points in this chunk
        chunk_rows = invalid_indices[0][start_idx:end_idx]
        chunk_cols = invalid_indices[1][start_idx:end_idx]
        slc_data_clean[chunk_rows, chunk_cols] = real_interp + 1j * imag_interp
        
        if verbose and (start_idx % (chunk_size * 10) == 0):
            print(f"Interpolated {end_idx}/{n_invalid} invalid points")
    
    return slc_data_clean


def validate_amp_clip_params(params, verbose=False):
    """
    Validate amplitude clipping parameters.
    
    Parameters:
    -----------
    params : list of float
        [amp_min, amp_max]
    
    Returns:
    --------
    bool
        True if parameters are valid
    """
    if len(params) != 2:
        print(f"Error: amp-clip-params requires exactly 2 values, got {len(params)}")
        return False
    
    amp_min, amp_max = params
    
    # Validate amplitude parameters (unscaled, must be non-negative)
    if amp_min < 0 or amp_max <= 0:
        print(f"Error: amplitude parameters must be non-negative with amp_max > 0, got [{amp_min:.6f}, {amp_max:.6f}]")
        return False
    
    # Handle amp_min = 0.0 case (convert to small positive value to avoid log(0))
    if amp_min == 0.0:
        amp_min = 1e-10 # Use a very small positive value
        params[0] = amp_min  # Update the original list
        if verbose:
            print(f"Note: amp_min of 0.0 converted to {amp_min:.2e} to avoid log(0) issues")
    
    if amp_min >= amp_max:
        print(f"Error: amp_min ({amp_min:.6f}) must be less than amp_max ({amp_max:.6f})")
        return False
    
    if verbose:
        print("Amplitude clipping parameters validated successfully:")
        print(f"  Amplitude (unscaled): [{amp_min:.6f}, {amp_max:.6f}]")
    
    return True


def load_array(input_path, verbose=False):
    """
    Load numpy array from file with error handling.
    """
    try:
        if verbose:
            print(f"Loading array from: {input_path}")
        
        data = np.load(input_path)
        
        if verbose:
            print(f"Array shape: {data.shape}")
            print(f"Array dtype: {data.dtype}")
            print(f"Array size: {data.size} elements ({data.nbytes / 1024**2:.1f} MB)")
        
        # Validate that it's complex data
        if not np.iscomplexobj(data):
            if verbose:
                print(f"Warning: Input array is not complex-valued (dtype: {data.dtype})")
                print("Expected complex64 or complex128 for SAR data")
            return None, "not_complex"
        
        # Validate that it's 2D
        if data.ndim != 2:
            if verbose:
                print(f"Warning: Input array must be 2D, got {data.ndim}D array")
            return None, "not_2d"
            
        return data, "success"
        
    except FileNotFoundError:
        if verbose:
            print(f"Warning: File not found: {input_path}")
        return None, "not_found"
    except Exception as e:
        if verbose:
            print(f"Warning: Error loading array: {e}")
        return None, "load_error"


def copy_label_file(input_path, labels_dir, output_labels_dir, verbose=False):
    """
    Copy corresponding label file with _proc suffix.
    
    Parameters:
    -----------
    input_path : str or Path
        Path to the input .npy file
    labels_dir : str or Path or None
        Directory containing label files (None if no labels available)
    output_labels_dir : str or Path
        Output directory for processed label files
    verbose : bool
        Verbose output flag
        
    Returns:
    --------
    tuple: (success, info)
        success (bool): True if label was copied or doesn't exist, False if error
        info (str): Status message or error description
    """
    input_path = Path(input_path)
    output_labels_dir = Path(output_labels_dir)
    
    # Skip if no labels directory (inference scenario)
    if labels_dir is None:
        if verbose:
            print(f"No labels directory - inference mode for {input_path.name}")
        return True, "inference_mode"
    
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        if verbose:
            print(f"Labels directory does not exist: {labels_dir}")
        return True, "labels_dir_not_found"
    
    # Find corresponding label file in labels directory
    label_filename = f"{input_path.stem}.txt"
    label_path = labels_dir / label_filename
    
    if not label_path.exists():
        if verbose:
            print(f"No label file found for {input_path.name} in {labels_dir}")
        return True, "no_label"  # Not an error - labels are optional
    
    try:
        # Create output labels directory if it doesn't exist
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output label filename with _proc suffix
        base_name = input_path.stem
        output_label_name = f"{base_name}_proc.txt"
        output_label_path = output_labels_dir / output_label_name
        
        # Copy the label file
        shutil.copy2(label_path, output_label_path)
        
        if verbose:
            print(f"Copied label: {label_path.name} → {output_label_path}")
        
        return True, str(output_label_path)
        
    except Exception as e:
        if verbose:
            print(f"Error copying label file: {e}")
        return False, str(e)


def save_as_png(rgb_data, input_path, output_dir, verbose=False):
    """
    Save RGB data as PNG file.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    input_path = Path(input_path)
    base_name = input_path.stem  # filename without extension
    output_filename = f"{base_name}_proc.png"
    output_path = output_dir / output_filename
    
    try:
        if verbose:
            print(f"Saving RGB image to: {output_path}")
            print(f"Output shape: {rgb_data.shape}")
            print(f"Output dtype: {rgb_data.dtype}")
        
        # Convert RGB to BGR for cv2.imwrite (OpenCV expects BGR format)
        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(output_path), bgr_data)
        if not success:
            return False, "cv2.imwrite failed"
            
        if verbose:
            print(f"Successfully saved: {output_path}")
        
        return True, str(output_path)
        
    except Exception as e:
        if verbose:
            print(f"Error saving PNG: {e}")
        return False, str(e)


def auto_detect_labels_dir(images_dir, verbose=False):
    """
    Auto-detect corresponding labels directory using naming conventions.
    
    Supports two patterns:
    - images_* → labels_* (e.g., images_hvrt_bal → labels_hvrt_bal)
    - images → labels (plain naming)
    
    Parameters:
    -----------
    images_dir : str or Path
        Path to images directory
    verbose : bool
        Verbose output flag
        
    Returns:
    --------
    Path or None
        Path to labels directory if it exists, None otherwise
    """
    images_dir = Path(images_dir)
    images_dir_name = images_dir.name
    
    # Pattern 1: images_* → labels_*
    if images_dir_name.startswith('images_'):
        labels_dir_name = images_dir_name.replace('images_', 'labels_', 1)
        labels_dir = images_dir.parent / labels_dir_name
        
        if labels_dir.exists():
            if verbose:
                print(f"Auto-detected labels directory: {labels_dir}")
            return labels_dir
        else:
            if verbose:
                print(f"No corresponding labels directory found: {labels_dir}")
            return None
    
    # Pattern 2: images → labels
    elif images_dir_name == 'images':
        labels_dir = images_dir.parent / 'labels'
        
        if labels_dir.exists():
            if verbose:
                print(f"Auto-detected labels directory: {labels_dir}")
            return labels_dir
        else:
            if verbose:
                print(f"No corresponding labels directory found: {labels_dir}")
            return None
    
    # No recognized pattern - assume inference mode
    else:
        if verbose:
            print(f"Directory name doesn't match expected patterns (images_* or images): {images_dir_name} - assuming inference mode")
        return None

def find_npy_files(input_path, verbose=False):
    """
    Find all .npy files in the input path.
    
    Parameters:
    -----------
    input_path : str
        Path to directory or single file
    
    Returns:
    --------
    list
        List of .npy file paths
    """
    input_path = Path(input_path)
    
    if input_path.is_file():
        if input_path.suffix.lower() == '.npy':
            return [str(input_path)]
        else:
            if verbose:
                print(f"Warning: {input_path} is not a .npy file")
            return []
    
    elif input_path.is_dir():
        # Find all .npy files in the directory
        npy_files = list(input_path.glob("*.npy"))
        npy_files.extend(input_path.glob("*.NPY"))  # Case insensitive
        
        # Convert to strings and sort
        npy_files = sorted([str(f) for f in npy_files])
        
        if verbose:
            print(f"Found {len(npy_files)} .npy files in {input_path}")
        
        return npy_files
    
    else:
        if verbose:
            print(f"Warning: {input_path} does not exist")
        return []

def process_batch(file_list, args, labels_dir, output_images_dir, output_labels_dir, verbose=False):
    """
    Process a batch of .npy files.
    
    Parameters:
    -----------
    file_list : list
        List of file paths to process
    args : argparse.Namespace
        Command line arguments
    labels_dir : Path or None
        Directory containing label files
    output_images_dir : Path
        Output directory for processed images
    output_labels_dir : Path
        Output directory for processed labels
    verbose : bool
        Verbose output flag
        
    Returns:
    --------
    dict
        Processing statistics
    """
    stats = {
        'total_files': len(file_list),
        'processed_successfully': 0,
        'skipped_nan': 0,
        'failed_load': 0,
        'failed_save': 0,
        'not_complex': 0,
        'not_2d': 0,
        'processing_errors': 0,
        'labels_copied': 0,
        'labels_not_found': 0,
        'labels_failed': 0,
        'start_time': datetime.now()
    }
    
    if verbose:
        print(f"\nStarting batch processing of {stats['total_files']} files...")
        print("=" * 80)
    else:
        print(f"Processing {stats['total_files']} files...")
    
    for i, file_path in enumerate(file_list, 1):
        if verbose:
            print(f"\n[{i:3d}/{stats['total_files']:3d}] Processing: {Path(file_path).name}")
            print("-" * 60)
        else:
            # Simple progress bar for non-verbose mode
            progress = i / stats['total_files']
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r[{bar}] {i:3d}/{stats['total_files']:3d} ({progress*100:5.1f}%)", end='', flush=True)
            
        try:
            # Load array
            slc_data, load_status = load_array(file_path, verbose)
            
            if load_status != "success":
                if load_status == "not_complex":
                    stats['not_complex'] += 1
                elif load_status == "not_2d":
                    stats['not_2d'] += 1
                else:
                    stats['failed_load'] += 1
                continue
            
            # Process the data
            processed_data = process_complex_data(
                slc_data,
                epsilon=args.epsilon,
                nan_strategy=args.nan_strategy,
                verbose=verbose,
                amp_clip_params=args.amp_clip_params
            )
            
            # Handle case where processing was skipped
            if processed_data is None:
                stats['skipped_nan'] += 1
                if verbose:
                    print("Skipped due to NaN values")
                continue
            
            # Save processed data
            save_success, save_info = save_as_png(
                processed_data, file_path, output_images_dir, verbose
            )
            
            if save_success:
                stats['processed_successfully'] += 1
                if verbose:
                    print(f"Successfully saved: {save_info}")
                
                # Copy corresponding label file if it exists
                label_success, label_info = copy_label_file(file_path, labels_dir, output_labels_dir, verbose)
                
                if label_success:
                    if label_info == "no_label":
                        stats['labels_not_found'] += 1
                    else:
                        stats['labels_copied'] += 1
                else:
                    stats['labels_failed'] += 1
                    if verbose:
                        print(f"Label copy failed: {label_info}")
            else:
                stats['failed_save'] += 1
                if verbose:
                    print(f"Save failed: {save_info}")
                
        except Exception as e:
            stats['processing_errors'] += 1
            if verbose:
                print(f"Processing error: {e}")
                import traceback
                traceback.print_exc()
    
    # Clear the progress bar line for non-verbose mode
    if not verbose:
        print()  # New line after progress bar
    
    stats['end_time'] = datetime.now()
    stats['duration'] = stats['end_time'] - stats['start_time']
    
    return stats

def print_batch_summary(stats, verbose=False):
    """
    Print batch processing summary.
    """
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total files found:        {stats['total_files']:4d}")
    print(f"Successfully processed:   {stats['processed_successfully']:4d}")
    print(f"Skipped (NaN values):     {stats['skipped_nan']:4d}")
    print(f"Failed to load:           {stats['failed_load']:4d}")
    print(f"Not complex data:         {stats['not_complex']:4d}")
    print(f"Not 2D data:              {stats['not_2d']:4d}")
    print(f"Failed to save:           {stats['failed_save']:4d}")
    print(f"Processing errors:        {stats['processing_errors']:4d}")
    print("-" * 80)
    print(f"Labels copied:            {stats['labels_copied']:4d}")
    print(f"Labels not found:         {stats['labels_not_found']:4d}")
    print(f"Label copy failures:      {stats['labels_failed']:4d}")
    print("-" * 80)
    
    success_rate = (stats['processed_successfully'] / stats['total_files'] * 100) if stats['total_files'] > 0 else 0
    print(f"Success rate:             {success_rate:.1f}%")
    print(f"Processing time:          {stats['duration']}")
    print(f"Average time per file:    {stats['duration'] / stats['total_files'] if stats['total_files'] > 0 else 0}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description="Process complex-valued SAR data into extended RGB PNG format with enhanced phase encoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file processing
  %(prog)s --input-dir data.npy --output-dir /results/processed_images/
  %(prog)s --input-dir images_hvrt_bal/data.npy --output-dir /results/images_hvrt_bal/
  
  # Directory processing (auto-detects input labels, infers output labels)
  %(prog)s --input-dir images_hvrt_bal/ --output-dir /results/images_hvrt_bal/
  %(prog)s --input-dir images/ --output-dir /results/images/
  %(prog)s --input-dir images_mosaic/ --output-dir /results/images_mosaic/ --verbose
  %(prog)s --input-dir images_rotate/ --output-dir /results/images_rotate/ --nan-strategy interpolate
  %(prog)s --input-dir images_hvrt_bal/ --output-dir /results/images_hvrt_bal/ --amp-clip-params 1.00 71.51
  
  # Inference mode (no labels, but labels directory still created for consistency)
  %(prog)s --input-dir raw_inference_images/ --output-dir /results/processed_data/

Output Format:
  Images go to specified output directory, labels auto-inferred:
  - --output-dir /results/images_hvrt_bal/ → /results/images_hvrt_bal/ (PNG files)
  - Labels automatically inferred → /results/labels_hvrt_bal/ (TXT files)

  RGB PNG images where:
  - Red channel: Clipped amplitude → dB scale → normalized to [0,1]
  - Green channel: sin(phase) normalized to [0,1] (preserves phase quadrant info)
  - Blue channel: cos(phase) normalized to [0,1] (preserves phase quadrant info)

  Label files copied with _proc suffix:
  - filename.npy + filename.txt → filename_proc.png + filename_proc.txt

Amplitude Clipping:
  The --amp-clip-params option takes 2 values: [amp_min, amp_max]
  - Values are unscaled amplitude bounds (clipped before dB conversion)
  - Use this for consistent normalization across multiple SAR images
        """
    )
    
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Path to input numpy array file (.npy) or image directory containing .npy files (e.g., images_hvrt_bal/)'
    )
    
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for processed images (corresponding labels directory will be auto-inferred)'
    )
    
    parser.add_argument(
        '--nan-strategy',
        choices=['skip', 'zero', 'mean', 'interpolate'],
        default='skip',
        help='Strategy for handling NaN/invalid values (default: skip)'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1e-6,
        help='Small value to prevent log(0) in amplitude processing (default: 1e-6)'
    )
    
    parser.add_argument(
        '--amp-clip-params',
        nargs=2,
        type=float,
        metavar=('AMP_MIN', 'AMP_MAX'),
        help='Amplitude clipping parameters: amp_min amp_max. '
             'Values are unscaled (clipped before dB conversion). Use for consistent normalization across dataset.'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input path exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input path does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Validate global normalisation parameters if provided
    if args.amp_clip_params is not None:
        if not validate_amp_clip_params(args.amp_clip_params, args.verbose):
            sys.exit(1)
    
    # Find files to process
    file_list = find_npy_files(args.input_dir, args.verbose)
    
    if not file_list:
        print(f"Error: No .npy files found in {args.input_dir}")
        sys.exit(1)
    
    # Determine processing mode and auto-detect labels
    input_path = Path(args.input_dir)
    is_directory = input_path.is_dir()
    
    # Auto-detect labels directory
    labels_dir = None
    if is_directory:
        labels_dir = auto_detect_labels_dir(input_path, args.verbose)
    else:
        # For single files, look for labels in parent directory structure
        parent_dir = input_path.parent
        if parent_dir.name.startswith('images_') or parent_dir.name == 'images':
            labels_dir = auto_detect_labels_dir(parent_dir, args.verbose)
    
    # Setup output directory structure
    # Use specified output directory directly for images
    output_images_dir = Path(args.output_dir)
    
    # Auto-infer corresponding labels directory from output images directory
    output_images_name = output_images_dir.name
    if output_images_name.startswith('images_'):
        labels_dir_name = output_images_name.replace('images_', 'labels_', 1)
        output_labels_dir = output_images_dir.parent / labels_dir_name
    elif output_images_name == 'images':
        output_labels_dir = output_images_dir.parent / 'labels'
    else:
        # For non-standard names, create 'labels' directory alongside
        output_labels_dir = output_images_dir.parent / 'labels'
    
    # Single file processing
    if not is_directory and len(file_list) == 1:
        if args.verbose:
            print("=" * 60)
            print("SAR Complex Data Extended Processing (sin/cos phase encoding)")
            print("=" * 60)
        
        file_path = file_list[0]
        slc_data, load_status = load_array(file_path, args.verbose)
        
        if load_status != "success":
            if load_status == "not_complex":
                print(f"Error: Input array is not complex-valued")
            elif load_status == "not_2d":
                print(f"Error: Input array must be 2D")
            else:
                print(f"Error: Failed to load input array")
            sys.exit(1)
        
        # Process the data
        if args.verbose:
            print("\nProcessing data...")
            print(f"NaN strategy: {args.nan_strategy}")
            print(f"Epsilon: {args.epsilon}")
            if args.amp_clip_params is not None:
                print("Using amplitude clipping parameters")
            else:
                print("Using adaptive normalization")
        
        processed_data = process_complex_data(
            slc_data,
            epsilon=args.epsilon,
            nan_strategy=args.nan_strategy,
            verbose=args.verbose,
            amp_clip_params=args.amp_clip_params
        )
        
        # Handle case where processing was skipped
        if processed_data is None:
            print("Processing skipped due to NaN values and nan_strategy='skip'")
            sys.exit(0)
        
        # Save processed data
        if args.verbose:
            print("\nSaving processed data...")
        
        save_success, save_info = save_as_png(processed_data, file_path, output_images_dir, args.verbose)
        
        if save_success:
            if args.verbose:
                print(f"Successfully saved: {save_info}")
            
            # Copy corresponding label file if it exists
            label_success, label_info = copy_label_file(file_path, labels_dir, output_labels_dir, args.verbose)
            
            if not label_success and label_info != "no_label":
                print(f"Warning: Failed to copy label file: {label_info}")
            
            if args.verbose:
                print("\nProcessing completed successfully!")
            else:
                print(f"Successfully processed: {save_info}")
        else:
            print(f"Error saving processed array: {save_info}")
            sys.exit(1)
    
    # Batch processing mode
    else:
        if args.verbose:
            print("=" * 60)
            print("SAR Extended Batch Processing Mode (sin/cos phase encoding)")
            print("=" * 60)
            print(f"Input path: {args.input_dir}")
            print(f"Output images directory: {args.output_dir}")
            print(f"Output labels directory: {output_labels_dir}")
            print(f"NaN strategy: {args.nan_strategy}")
            print(f"Epsilon: {args.epsilon}")
            if args.amp_clip_params is not None:
                print("Using amplitude clipping parameters")
            else:
                print("Using adaptive normalization per file")
        
        # Process all files
        stats = process_batch(file_list, args, labels_dir, output_images_dir, output_labels_dir, args.verbose)
        
        # Print summary
        print_batch_summary(stats, args.verbose)
        
        # Exit with appropriate code
        if stats['processed_successfully'] == 0:
            print("\nNo files were processed successfully!")
            sys.exit(1)
        elif stats['processed_successfully'] < stats['total_files']:
            print(f"\nPartial success: {stats['processed_successfully']}/{stats['total_files']} files processed")
            sys.exit(2)
        else:
            print("\nAll files processed successfully!")
            sys.exit(0)

if __name__ == "__main__":
    main()
