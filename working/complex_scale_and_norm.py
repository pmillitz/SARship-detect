#!/usr/bin/env python3

"""
complex_scale_and_norm.py

Author: Peter Milllitz
Date: 19-06-2025

Takes as input a complex-valued 2D SAR SLC product in the form of a numpy.ndarray of dtype=complex64
and shape (H, W). Extracts amplitude and phase, applies decibel scaling to amplitude and
sine/cosine transformation to phase. Normalises all components to [0, 1] range. Outputs
numpy.ndarray with shape (3, H, W) compatible with YOLO models. Memory-optimised.

usage: 

    complex_scale_and_norm.py [-h] [--output-dir OUTPUT_DIR] [--nan-strategy {skip,zero,mean,interpolate}]
                              [--epsilon EPSILON] [--verbose]
                              [--global-norm-params AMP_MIN AMP_MAX]
                              input_array
Parameters:
-----------
h: 
    Prints detailed program usage 
input_array: numpy.ndarray
    2D complex-valued input array (Single Look Complex data)
output-dir:
    output directory for processed arrays 
nan_strategy : str, optional
    How to handle NaN/invalid values: 'interpolate', 'zero', 'mean', or 'skip' (default: 'skip')
epsilon: float, optional
    Small value to prevent log(0) in amplitude processing (default: 1e-6)
verbose (v): bool, optional
    Print normalisation range information (default: False)
global_norm_params: list of float, optional
    Global normalisation parameters [amp_min, amp_max] for consistent
    amplitude normalisation across multiple images. amp_min/max are unscaled amplitude values 
    (will be converted to dB internally). Phase normalisation is automatic via sine/cosine 
    transformation. (default: None, use adaptive normalisation)

Example usage:

    python complex_scale_and_norm.py input_array.npy --output-dir processed/ --verbose
    python complex_scale_and_norm.py input_array.npy --global-norm-params 0.001 50.2

"""

import numpy as np
import argparse
import os
from pathlib import Path
import sys


def process_complex_data(slc_data, epsilon=1e-6, nan_strategy='skip', verbose=False, global_norm_params=None):
    """
    Processes a single complex-valued 2D SAR SLC data array. Extracts amplitude and
    phase, applies scaling and normalisation, then stacks into a 3-channel format.
    
    Returns:
    --------
    numpy.ndarray
        3D array of shape (3, H, W) with normalised amplitude, phase_sin, and phase_cos channels
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
    
    # Extract global normalisation parameters if provided
    if global_norm_params is not None:
        global_amp_min, global_amp_max = global_norm_params
        use_global_norm = True
        if verbose:
            print(f"Using global normalisation parameters:")
            print(f"  Amplitude (unscaled): [{global_amp_min:.6f}, {global_amp_max:.6f}]")
            print("  Phase normalisation: automatic via sine/cosine transformation")
    else:
        use_global_norm = False
        global_amp_min = global_amp_max = None
    
    # Memory-efficient amplitude processing
    VH_mag_norm = _process_amplitude_inplace(slc_data, epsilon, verbose, 
                                           use_global_norm, global_amp_min, global_amp_max)
    
    # Memory-efficient phase processing (sine/cosine components)
    VH_phase_sin_norm, VH_phase_cos_norm = _process_phase_sincos_inplace(slc_data, verbose)
    
    # Stack into (3, H, W) format for YOLO compatibility
    image_input = np.stack((VH_mag_norm, VH_phase_sin_norm, VH_phase_cos_norm), axis=0)
    
    return image_input


def _process_amplitude_inplace(complex_data, epsilon=1e-6, verbose=False, 
                             use_global_norm=False, global_amp_min=None, global_amp_max=None):
    """
    Process amplitude in-place to minimise memory usage.
    """
    # Extract magnitude (automatically float32 from complex64)
    VH_mag = np.abs(complex_data)
    
    # Use float32 constants to prevent upcasting
    epsilon_f32 = np.float32(epsilon)
    
    # Always apply logarithmic scaling (dB conversion)
    np.log10(VH_mag + epsilon_f32, out=VH_mag)
    VH_mag *= np.float32(20)
    
    if use_global_norm:
        # Convert global unscaled amplitude parameters to dB scale with float32
        global_mag_min_db = np.float32(20) * np.log10(np.float32(global_amp_min) + epsilon_f32)
        global_mag_max_db = np.float32(20) * np.log10(np.float32(global_amp_max) + epsilon_f32)
        
        if verbose:
            local_min = VH_mag.min()
            local_max = VH_mag.max()
            print(f"Amplitude range after scaling (dB): [{local_min:.3f}, {local_max:.3f}]")
        
        # Apply global normalisation using dB-scaled bounds
        VH_mag -= global_mag_min_db
        VH_mag /= (global_mag_max_db - global_mag_min_db)

        if verbose:
            # Check how many values are out of bounds before clipping
            out_of_bounds = (VH_mag < 0) | (VH_mag > 1)
            clipped_count = np.sum(out_of_bounds)

        # Clip values to [0, 1] range to handle out-of-range values
        np.clip(VH_mag, np.float32(0.0), np.float32(1.0), out=VH_mag)
        
        if verbose:
            final_min = VH_mag.min()
            final_max = VH_mag.max()
            print(f"Final amplitude range after normalisation: [{final_min:.3f}, {final_max:.3f}]")

            if clipped_count > 0:
                print(f"Clipped {clipped_count} out-of-range values ({clipped_count/VH_mag.size*100:.1f}%) to [0,1] range")

    else:
        # Use adaptive normalisation
        mag_min = VH_mag.min()
        mag_max = VH_mag.max()
        VH_mag -= mag_min
        VH_mag /= (mag_max - mag_min)
        
        if verbose:
            print(f"Adaptive amplitude normalisation: [{mag_min:.3f}, {mag_max:.3f}] dB → [0.000, 1.000]")
    
    return VH_mag


def _process_phase_sincos_inplace(complex_data, verbose=False):
    """
    Process phase into sine/cosine components and normalise to [0, 1] range.
    Normalisation is automatic - no global parameters needed.
    """
    # Extract phase (automatically float32 from complex64)
    VH_phase = np.angle(complex_data)
    
    # Convert to sine and cosine components with float32 constants
    VH_phase_sin = (np.sin(VH_phase) + np.float32(1)) / np.float32(2)
    VH_phase_cos = (np.cos(VH_phase) + np.float32(1)) / np.float32(2)
    
    if verbose:
        print(f"Phase sine range: [{VH_phase_sin.min():.3f}, {VH_phase_sin.max():.3f}]")
        print(f"Phase cosine range: [{VH_phase_cos.min():.3f}, {VH_phase_cos.max():.3f}]")
        print(f"Output dtype: {VH_phase_sin.dtype}")
    
    return VH_phase_sin, VH_phase_cos


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


def validate_global_norm_params(params, verbose=False):
    """
    Validate global normalisation parameters.
    
    Parameters:
    -----------
    params : list of float
        [amp_min, amp_max] (phase normalisation is automatic via sine/cosine)
    
    Returns:
    --------
    bool
        True if parameters are valid
    """
    if len(params) != 2:
        print(f"Error: global-norm-params requires exactly 2 values (amp_min, amp_max), got {len(params)}")
        return False
    
    amp_min, amp_max = params
    
    # Validate amplitude parameters (unscaled, must be non-negative)
    if amp_min < 0 or amp_max <= 0:
        print(f"Error: amplitude parameters must be non-negative with amp_max > 0, got [{amp_min:.6f}, {amp_max:.6f}]")
        return False
    
    # Handle amp_min = 0.0 case with float32-appropriate value
    if amp_min == 0.0:
        amp_min = 1e-6  # Float32-safe value, consistent with default epsilon
        params[0] = amp_min  # Update the original list
        if verbose:
            print(f"Note: amp_min of 0.0 converted to {amp_min:.2e} to avoid log(0) issues")
    
    if amp_min >= amp_max:
        print(f"Error: amp_min ({amp_min:.6f}) must be less than amp_max ({amp_max:.6f})")
        return False
    
    if verbose:
        print("Global normalisation parameters validated successfully:")
        print(f"  Amplitude (unscaled): [{amp_min:.6f}, {amp_max:.6f}]")
        print("  Phase normalisation: automatic via sine/cosine transformation")
    
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
            print(f"Array size: {data.size} elements ({data.nbytes / 1024:.1f} KB)")
        
        # Validate that it's complex data
        if not np.iscomplexobj(data):
            print(f"Warning: Input array is not complex-valued (dtype: {data.dtype})")
            print("Expected complex64 or complex128 for SAR data")
        
        # Validate that it's 2D
        if data.ndim != 2:
            print(f"Error: Input array must be 2D, got {data.ndim}D array")
            sys.exit(1)
            
        return data
        
    except FileNotFoundError:
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading array: {e}")
        sys.exit(1)


def save_processed_array(processed_data, input_path, output_dir, verbose=False):
    """
    Save processed array with appropriate naming convention.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    input_path = Path(input_path)
    base_name = input_path.stem  # filename without extension
    output_filename = f"{base_name}_proc.npy"
    output_path = output_dir / output_filename
    
    try:
        if verbose:
            print(f"Saving processed array to: {output_path}")
            print(f"Output shape: {processed_data.shape}")
            print(f"Output dtype: {processed_data.dtype}")
            print(f"Output size: {processed_data.nbytes / 1024:.1f} KB")
        
        np.save(output_path, processed_data)
        print(f"Successfully saved: {output_path}")
        
    except Exception as e:
        print(f"Error saving processed array: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Process complex-valued SAR data into normalised 3-channel format for YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.npy
  %(prog)s data.npy --output-dir processed/
  %(prog)s data.npy --nan-strategy zero --verbose
  %(prog)s data.npy --output-dir results/ --nan-strategy interpolate --epsilon 1e-8
  %(prog)s data.npy --global-norm-params 0.001 50.2 --verbose
  %(prog)s data.npy --global-norm-params 0.005 25.0 --output-dir processed/

Global Normalisation:
  The --global-norm-params option takes 2 values: [amp_min, amp_max]
  - amp_min/max: Unscaled amplitude values (will be converted to dB scale internally)
  - Phase normalisation is automatic via sine/cosine transformation to [0, 1] range
  Use this option to ensure consistent amplitude normalisation across multiple SAR images.
        """
    )
    
    parser.add_argument(
        'input_array',
        help='Path to input numpy array file (.npy)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory (default: current working directory)'
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
        '--global-norm-params',
        nargs=2,
        type=float,
        metavar=('AMP_MIN', 'AMP_MAX'),
        help='Global normalisation parameters: amp_min amp_max. '
             'Amplitude values are unscaled (converted to dB internally). '
             'Phase normalisation is automatic via sine/cosine transformation.'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_array):
        print(f"Error: Input file does not exist: {args.input_array}")
        sys.exit(1)
    
    # Validate global normalisation parameters if provided
    if args.global_norm_params is not None:
        if not validate_global_norm_params(args.global_norm_params, args.verbose):
            sys.exit(1)
    
    # Load input array
    if args.verbose:
        print("=" * 60)
        print("SAR Complex Data Scaling and Normalisation for YOLO")
        print("=" * 60)
    
    slc_data = load_array(args.input_array, args.verbose)
    
    # Process the data
    if args.verbose:
        print("\nProcessing data...")
        print(f"NaN strategy: {args.nan_strategy}")
        print(f"Epsilon: {args.epsilon}")
        if args.global_norm_params is None:
            print("Using adaptive normalisation")
    
    processed_data = process_complex_data(
        slc_data,
        epsilon=args.epsilon,
        nan_strategy=args.nan_strategy,
        verbose=args.verbose,
        global_norm_params=args.global_norm_params
    )
    
    # Handle case where processing was skipped
    if processed_data is None:
        print("Processing skipped due to NaN values and nan_strategy='skip'")
        sys.exit(0)
    
    # Save processed data
    if args.verbose:
        print("\nSaving processed data...")
    
    save_processed_array(processed_data, args.input_array, args.output_dir, args.verbose)
    
    if args.verbose:
        print("\nProcessing completed successfully!")
        print(f"Output format: {processed_data.shape} (channels, height, width)")
        print(f"Output dtype: {processed_data.dtype}")
        print("Channels: [0] Amplitude, [1] Phase Sine, [2] Phase Cosine")
        print(f"Memory usage: {processed_data.nbytes / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
