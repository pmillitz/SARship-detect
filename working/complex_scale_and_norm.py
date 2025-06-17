#!/usr/bin/env python3

"""
complex_scale_and_norm.py

Author: Peter Millitz
Date: 16-06-2025

Takes as input a complex-valued 2D SAR SLC product in the form of a numpy.ndarray of dtype=complex64
and shape (H, W). Extracts amplitude and phase, applies decibel scaling to ammplitude and
normalisation to amplitude and phase. Adds a third 'zeros' channel, then stacks the data. Outputs
numpy.ndarray with shape (H, W, 3). Memory-optimised.

usage: 

    complex_scale_and_norm.py [-h] [--output-dir OUTPUT_DIR] [--nan-strategy {skip,zero,mean,interpolate}]
                              [--epsilon EPSILON] [--verbose]
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
    Print normalization range information (default: False)

Example usage:

    python complex_scale_and_norm.py data/slc_data.npy --output-dir processed/ --verbose

"""

import numpy as np
import argparse
import os
from pathlib import Path
import sys


def process_complex_data(slc_data, epsilon=1e-6, nan_strategy='skip', verbose=False):
    """
    Processes a single complex-valued 2D SAR SLC data array. Extracts amplitude and
    phase, applies scaling and normalization, then stacks into a 3-channel format.
    
    
    Returns:
    --------
    numpy.ndarray
        3D array of shape (H, W, 3) with normalised amplitude, phase, and zeros channels
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
    
    # Memory-efficient amplitude processing
    VH_mag_norm = _process_amplitude_inplace(slc_data, epsilon, verbose)
    
    # Memory-efficient phase processing
    VH_phase_norm = _process_phase_inplace(slc_data, verbose)
    
    # Create zeros channel (reuse memory-efficient approach)
    zeros_channel = np.zeros(slc_data.shape)
    
    # Stack into (H, W, 3) format
    image_input = np.stack((VH_mag_norm, VH_phase_norm, zeros_channel), axis=-1)
    
    return image_input


def _process_amplitude_inplace(complex_data, epsilon=1e-6, verbose=False):
    """
    Process amplitude in-place to minimize memory usage.
    """
    # Extract magnitude
    VH_mag = np.abs(complex_data)
    
    # In-place logarithmic scaling
    np.log10(VH_mag + epsilon, out=VH_mag)
    VH_mag *= 20
    
    # In-place normalization
    mag_min = VH_mag.min()
    mag_max = VH_mag.max()
    VH_mag -= mag_min
    VH_mag /= (mag_max - mag_min)
    
    if verbose:
        print(f"VH_mag_norm range: [0.000, 1.000]")
    
    return VH_mag


def _process_phase_inplace(complex_data, verbose=False):
    """
    Process phase in-place to minimize memory usage.
    """
    # Extract phase
    VH_phase = np.angle(complex_data)
    
    # Phase normalization with floating-point tolerance
    phase_min = VH_phase.min()
    phase_max = VH_phase.max()
    tolerance = 1e-6
    
    if verbose:
        print(f"Raw phase range: [{phase_min:.10f}, {phase_max:.10f}]")
    
    # Determine the legitimate range based on data distribution
    if phase_min >= -tolerance and phase_max <= (2*np.pi + tolerance):
        # Data appears to be in [0, 2pi] range (with floating-point errors)
        legitimate_range = "[0, 2pi]"
        
        # Clean up floating-point precision issues in-place
        np.clip(VH_phase, 0, 2*np.pi, out=VH_phase)
        VH_phase /= (2*np.pi)
        
        if verbose:
            print(f"Detected {legitimate_range} range, applied direct normalization")
            if phase_max > 2*np.pi:
                print(f"Corrected floating-point error: max {phase_max:.10f} → 2pi")
    
    elif phase_min >= (-np.pi - tolerance) and phase_max <= (np.pi + tolerance):
        # Data appears to be in [-pi, pi] range (with floating-point errors)
        legitimate_range = "[-pi, pi]"
        
        # Clean up floating-point precision issues in-place
        np.clip(VH_phase, -np.pi, np.pi, out=VH_phase)
        VH_phase += np.pi
        VH_phase /= (2*np.pi)
        
        if verbose:
            print(f"Detected {legitimate_range} range, applied shift-then-divide normalization")
            if phase_max > np.pi:
                print(f"Corrected floating-point error: max {phase_max:.10f} → pi")
            if phase_min < -np.pi:
                print(f"Corrected floating-point error: min {phase_min:.10f} → -pi")
    
    else:
        # Truly unexpected range - this indicates a real data issue
        if verbose:
            print(f"Warning: Unexpected phase range [{phase_min:.6f}, {phase_max:.6f}]")
            print("This may indicate upstream processing issues")
        
        # Force wrap to [-pi, pi] and normalise in-place
        # Use temporary array only for the wrapping operation
        wrapped_phase = np.angle(np.exp(1j * VH_phase))
        VH_phase[:] = wrapped_phase  # Copy back in-place
        VH_phase += np.pi
        VH_phase /= (2*np.pi)
    
    if verbose:
        print(f"VH_phase_norm range: [{np.min(VH_phase):.3f}, {np.max(VH_phase):.3f}]")
    
    return VH_phase


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
            print(f"Output size: {processed_data.nbytes / 1024**2:.1f} MB")
        
        np.save(output_path, processed_data)
        print(f"Successfully saved: {output_path}")
        
    except Exception as e:
        print(f"Error saving processed array: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Process complex-valued SAR data into normalised 3-channel format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.npy
  %(prog)s data.npy --output-dir processed/
  %(prog)s data.npy --nan-strategy zero --verbose
  %(prog)s data.npy --output-dir results/ --nan-strategy interpolate --epsilon 1e-8
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
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_array):
        print(f"Error: Input file does not exist: {args.input_array}")
        sys.exit(1)
    
    # Load input array
    if args.verbose:
        print("=" * 60)
        print("SAR Complex Data Scaling and Normalisation")
        print("=" * 60)
    
    slc_data = load_array(args.input_array, args.verbose)
    
    # Process the data
    if args.verbose:
        print("\nProcessing data...")
        print(f"NaN strategy: {args.nan_strategy}")
        print(f"Epsilon: {args.epsilon}")
    
    processed_data = process_complex_data(
        slc_data,
        epsilon=args.epsilon,
        nan_strategy=args.nan_strategy,
        verbose=args.verbose
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


if __name__ == "__main__":
    main()

