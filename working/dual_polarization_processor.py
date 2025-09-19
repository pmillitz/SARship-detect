#!/usr/bin/env python3

"""
dual_polarisation_processor.py

Author: Peter Millitz
Created: 2025-09-13

This script processes dual-polarisation SAR crops (VH and VV) to create 3-channel RGB PNG images
suitable for YOLO training. The three channels are:
- Red Channel: VH amplitude (clipped, dB-scaled, normalized to [0,1])
- Green Channel: VV amplitude (clipped, dB-scaled, normalized to [0,1])  
- Blue Channel: Polarisation Coherence Features (PCF) normalized to [0,1]

The script takes VH and VV crop directories as input and outputs 3-channel PNG files.
Processing logic replicates and extends the approach from complex_scale_and_norm_png.py
for dual-polarisation data.
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
import sys
import shutil
from datetime import datetime
from tqdm import tqdm

class Logger:
    """Logger class to handle both file and console output."""
    
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        
        # Open log file for writing
        try:
            self.log_file = open(log_file_path, 'w', encoding='utf-8')
            # Write header to log file
            self.log_file.write(f"Dual Polarisation Processing Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("="*80 + "\n\n")
            self.log_file.flush()
        except Exception as e:
            print(f"Warning: Could not create log file {log_file_path}: {e}")
            self.log_file = None
    
    def print(self, message, force_screen=True):
        """Print message to both file and screen."""
        # Always write to log file if available
        if self.log_file:
            try:
                self.log_file.write(message + "\n")
                self.log_file.flush()
            except Exception:
                pass
        
        # Print to screen
        if force_screen:
            print(message)
    
    def close(self):
        """Close the log file."""
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass

def compute_pcf(vh_complex, vv_complex):
    """
    Compute Polarisation Coherence Features (PCF) from VH and VV complex arrays.
    
    PCF = |VV * conjugate(VH)|
    
    Args:
        vh_complex (np.ndarray): Complex VH data
        vv_complex (np.ndarray): Complex VV data
        
    Returns:
        np.ndarray: PCF values (real, non-negative)
    """
    # Calculate cross-polarisation coherence
    cross_pol_product = vv_complex * np.conj(vh_complex)
    
    # Take absolute value to get magnitude
    pcf = np.abs(cross_pol_product)
    
    return pcf

def process_amplitude(complex_array, amp_min, amp_max, epsilon=1e-6):
    """
    Process complex SAR data to amplitude with clipping, dB scaling, and normalization.
    
    Args:
        complex_array (np.ndarray): Complex SAR data
        amp_min (float): Minimum amplitude for clipping (99% coverage)
        amp_max (float): Maximum amplitude for clipping (99% coverage)
        epsilon (float): Small value to prevent log(0)
        
    Returns:
        np.ndarray: Processed amplitude normalized to [0,1]
    """
    # Extract amplitude
    amplitude = np.abs(complex_array)
    
    # Clip to remove extreme outliers (99% coverage)
    amplitude_clipped = np.clip(amplitude, amp_min, amp_max)
    
    # Add epsilon to prevent log(0)
    amplitude_safe = amplitude_clipped + epsilon
    
    # Convert to dB scale
    amplitude_db = 20 * np.log10(amplitude_safe)
    
    # Normalize to [0, 1] range
    db_min = 20 * np.log10(amp_min + epsilon)
    db_max = 20 * np.log10(amp_max + epsilon)
    amplitude_normalized = (amplitude_db - db_min) / (db_max - db_min)
    
    # Ensure values are in [0, 1] range
    amplitude_normalized = np.clip(amplitude_normalized, 0, 1)
    
    return amplitude_normalized

def process_pcf(pcf_array, pcf_min=4.24, pcf_max=11138.0, epsilon=1e-6):
    """
    Process PCF data with clipping, dB scaling, and normalization.
    
    Args:
        pcf_array (np.ndarray): Raw PCF values (real, non-negative)
        pcf_min (float): Minimum PCF for clipping (99% coverage: 4.24)
        pcf_max (float): Maximum PCF for clipping (99% coverage: 11138.0)
        epsilon (float): Small value to prevent log(0)
        
    Returns:
        np.ndarray: Processed PCF normalized to [0,1]
    """
    # Clip to remove extreme outliers (99% coverage)
    pcf_clipped = np.clip(pcf_array, pcf_min, pcf_max)
    
    # Add epsilon to prevent log(0)
    pcf_safe = pcf_clipped + epsilon
    
    # Convert to dB scale
    pcf_db = 20 * np.log10(pcf_safe)
    
    # Normalize to [0, 1] range
    db_min = 20 * np.log10(pcf_min + epsilon)
    db_max = 20 * np.log10(pcf_max + epsilon)
    pcf_normalized = (pcf_db - db_min) / (db_max - db_min)
    
    # Ensure values are in [0, 1] range
    pcf_normalized = np.clip(pcf_normalized, 0, 1)
    
    return pcf_normalized

def process_dual_polarisation_crop(vh_path, vv_path, output_path, logger=None):
    """
    Process a single VH/VV crop pair to create 3-channel RGB PNG.
    
    Args:
        vh_path (Path): Path to VH crop file (.npy)
        vv_path (Path): Path to VV crop file (.npy)
        output_path (Path): Path for output PNG file
        logger (Logger): Logger instance
        
    Returns:
        dict: Processing results and statistics
    """
    result = {
        'success': False,
        'vh_shape': None,
        'vv_shape': None,
        'pcf_min': None,
        'pcf_max': None,
        'error': None
    }
    
    try:
        # Load VH and VV crops
        vh_complex = np.load(vh_path)
        vv_complex = np.load(vv_path)
        
        # Validate shapes match
        if vh_complex.shape != vv_complex.shape:
            result['error'] = f"Shape mismatch: VH {vh_complex.shape} vs VV {vv_complex.shape}"
            return result
            
        result['vh_shape'] = vh_complex.shape
        result['vv_shape'] = vv_complex.shape
        
        # Validate data types
        if not np.iscomplexobj(vh_complex) or not np.iscomplexobj(vv_complex):
            result['error'] = f"Expected complex data, got VH: {vh_complex.dtype}, VV: {vv_complex.dtype}"
            return result
        
        # Process VH amplitude (Red channel) - clipping range [1.0, 71.5]
        vh_amplitude = process_amplitude(vh_complex, amp_min=1.0, amp_max=71.5)
        
        # Process VV amplitude (Green channel) - clipping range [2.0, 232.7]
        vv_amplitude = process_amplitude(vv_complex, amp_min=2.0, amp_max=232.7)
        
        # Compute PCF (Blue channel)
        pcf = compute_pcf(vh_complex, vv_complex)
        
        # Process PCF with clip-dB-scale-normalize (99% coverage: [4.24, 11138.0])
        pcf_processed = process_pcf(pcf, pcf_min=4.24, pcf_max=11138.0)
        
        # Store raw PCF statistics for analysis
        pcf_min = pcf.min()
        pcf_max = pcf.max()
        result['pcf_min'] = float(pcf_min)
        result['pcf_max'] = float(pcf_max)
        
        # Stack channels: Red=VH, Green=VV, Blue=Processed_PCF
        rgb_array = np.stack([vh_amplitude, vv_amplitude, pcf_processed], axis=-1)
        
        # Convert to uint8 for PNG format
        rgb_uint8 = (rgb_array * 255).astype(np.uint8)
        
        # Save as PNG using OpenCV (BGR format)
        bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(output_path), bgr_uint8)
        
        if not success:
            result['error'] = "Failed to write PNG file"
            return result
            
        # Verify file was created
        if not output_path.exists():
            result['error'] = "Output file was not created"
            return result
            
        result['success'] = True
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result

def find_corresponding_crops(vh_dir, vv_dir, logger=None):
    """
    Find corresponding VH and VV crop pairs based on filename matching.
    
    Args:
        vh_dir (Path): Directory containing VH crops
        vv_dir (Path): Directory containing VV crops
        logger (Logger): Logger instance
        
    Returns:
        list: List of (vh_path, vv_path) tuples for matching crops
    """
    vh_files = list(vh_dir.glob("*.npy"))
    vv_files = list(vv_dir.glob("*.npy"))
    
    # Create lookup dictionary for VV files
    vv_lookup = {f.name: f for f in vv_files}
    
    crop_pairs = []
    missing_vv = []
    
    for vh_file in vh_files:
        # Look for corresponding VV file with same name
        if vh_file.name in vv_lookup:
            crop_pairs.append((vh_file, vv_lookup[vh_file.name]))
        else:
            missing_vv.append(vh_file.name)
    
    if logger:
        logger.print(f"Found {len(vh_files)} VH crops")
        logger.print(f"Found {len(vv_files)} VV crops")
        logger.print(f"Matched {len(crop_pairs)} crop pairs")
        
        if missing_vv:
            logger.print(f"Warning: {len(missing_vv)} VH crops missing corresponding VV crops")
            if len(missing_vv) <= 10:  # Only show first 10 to avoid spam
                for missing in missing_vv:
                    logger.print(f"  Missing VV for: {missing}")
            else:
                logger.print(f"  (showing first 10 missing)")
                for missing in missing_vv[:10]:
                    logger.print(f"  Missing VV for: {missing}")
    
    return crop_pairs

def copy_label_file(vh_path, labels_dir, output_labels_dir, logger=None):
    """
    Copy corresponding label file with _proc suffix.

    Args:
        vh_path (Path): Path to the VH crop file (.npy) - used to determine base filename
        labels_dir (Path): Directory containing label files (None if no labels available)
        output_labels_dir (Path): Output directory for processed label files
        logger (Logger): Logger instance

    Returns:
        tuple: (success, info)
            success (bool): True if label was copied or doesn't exist, False if error
            info (str): Status message or error description
    """
    # Skip if no labels directory provided
    if labels_dir is None:
        return True, "no_labels_dir"

    labels_dir = Path(labels_dir)
    output_labels_dir = Path(output_labels_dir)

    if not labels_dir.exists():
        return True, "labels_dir_not_found"

    # Find corresponding label file in labels directory
    label_filename = f"{vh_path.stem}.txt"
    label_path = labels_dir / label_filename

    if not label_path.exists():
        return True, "no_label"  # Not an error - labels are optional

    try:
        # Create output labels directory if it doesn't exist
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        # Create output label filename with _proc suffix
        base_name = vh_path.stem
        output_label_name = f"{base_name}_proc.txt"
        output_label_path = output_labels_dir / output_label_name

        # Copy the label file
        shutil.copy2(label_path, output_label_path)

        if logger:
            logger.print(f"Copied label: {label_path.name} → {output_label_path.name}", force_screen=False)

        return True, str(output_label_path)

    except Exception as e:
        error_msg = f"Error copying label file: {e}"
        if logger:
            logger.print(error_msg)
        return False, error_msg

def main():
    """
    Main function to process dual-polarization crops.
    """
    parser = argparse.ArgumentParser(
        description="Process dual-polarisation SAR crops to 3-channel RGB PNGs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dual_polarisation_processor.py --vh-dir train/crops_vh --vv-dir train/crops_vv --output-dir train/crops_dual
  python dual_polarisation_processor.py --vh-dir val/crops_vh --vv-dir val/crops_vv --output-dir val/crops_dual --data-split val
  python dual_polarisation_processor.py --vh-dir train/crops_vh --vv-dir train/crops_vv --output-dir train/crops_dual --labels-dir train/labels --output-labels-dir train/labels_dual
        """
    )
    
    parser.add_argument("--vh-dir", required=True, type=Path,
                       help="Directory containing VH crop files (.npy)")
    parser.add_argument("--vv-dir", required=True, type=Path,
                       help="Directory containing VV crop files (.npy)")
    parser.add_argument("--output-dir", required=True, type=Path,
                       help="Output directory for 3-channel PNG files")
    parser.add_argument("--labels-dir", type=Path,
                       help="Directory containing corresponding label files (.txt) - optional")
    parser.add_argument("--output-labels-dir", type=Path,
                       help="Output directory for processed label files - optional (defaults to output-dir/../labels_dual if not specified)")
    parser.add_argument("--data-split", default="",
                       help="Data split name (for logging) - optional")
    
    args = parser.parse_args()
    
    # Validate input directories
    if not args.vh_dir.exists():
        print(f"Error: VH directory not found: {args.vh_dir}")
        sys.exit(1)
        
    if not args.vv_dir.exists():
        print(f"Error: VV directory not found: {args.vv_dir}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup output labels directory
    if args.output_labels_dir is None:
        # Default: create labels_dual directory alongside output_dir
        args.output_labels_dir = args.output_dir.parent / "labels_dual"

    # Validate labels directory if provided
    if args.labels_dir is not None and not args.labels_dir.exists():
        print(f"Warning: Labels directory not found: {args.labels_dir}")
        print("Label files will not be copied.")
        args.labels_dir = None
    
    # Initialize logger
    log_filename = f"dual_polarisation_processing_{args.data_split}.log" if args.data_split else "dual_polarisation_processing.log"
    logger = Logger(log_filename)
    
    # Log processing parameters
    split_info = f" ({args.data_split.upper()} split)" if args.data_split else ""
    logger.print(f"DUAL POLARISATION PROCESSING{split_info}")
    logger.print("="*60)
    logger.print(f"VH crops directory: {args.vh_dir}")
    logger.print(f"VV crops directory: {args.vv_dir}")
    logger.print(f"Output directory: {args.output_dir}")
    if args.labels_dir:
        logger.print(f"Labels directory: {args.labels_dir}")
        logger.print(f"Output labels directory: {args.output_labels_dir}")
    else:
        logger.print("No labels directory specified - inference mode")
    logger.print(f"Processing channels:")
    logger.print(f"  Red Channel: VH amplitude (clipped [1.0, 71.5], dB-scaled, normalized)")
    logger.print(f"  Green Channel: VV amplitude (clipped [2.0, 232.7], dB-scaled, normalized)")
    logger.print(f"  Blue Channel: PCF (clipped [4.24, 11138.0], dB-scaled, normalized)")
    logger.print("")
    
    # Find corresponding crop pairs
    logger.print("Finding corresponding VH/VV crop pairs...")
    crop_pairs = find_corresponding_crops(args.vh_dir, args.vv_dir, logger)
    
    if not crop_pairs:
        logger.print("Error: No matching VH/VV crop pairs found")
        logger.close()
        sys.exit(1)
    
    # Process statistics
    stats = {
        'processed': 0,
        'successful': 0,
        'failed': 0,
        'pcf_stats': [],
        'errors': [],
        'labels_copied': 0,
        'labels_not_found': 0,
        'labels_failed': 0
    }
    
    # Process each crop pair with progress bar
    with tqdm(crop_pairs, desc="Processing crops", unit="crop") as pbar:
        for vh_path, vv_path in pbar:
            # Generate output filename with _proc suffix for consistency
            output_filename = vh_path.stem + "_proc.png"
            output_path = args.output_dir / output_filename
            
            # Process the crop pair
            result = process_dual_polarisation_crop(vh_path, vv_path, output_path, logger)
            
            stats['processed'] += 1
            
            if result['success']:
                stats['successful'] += 1
                if result['pcf_min'] is not None and result['pcf_max'] is not None:
                    stats['pcf_stats'].append((result['pcf_min'], result['pcf_max']))

                # Copy corresponding label file if labels directory is provided
                if args.labels_dir:
                    label_success, label_info = copy_label_file(vh_path, args.labels_dir, args.output_labels_dir, logger)

                    if label_success:
                        if label_info == "no_label":
                            stats['labels_not_found'] += 1
                        elif label_info not in ["no_labels_dir", "labels_dir_not_found"]:
                            stats['labels_copied'] += 1
                    else:
                        stats['labels_failed'] += 1
                        logger.print(f"Label copy failed for {vh_path.name}: {label_info}")
            else:
                stats['failed'] += 1
                error_msg = f"Failed to process {vh_path.name}: {result['error']}"
                stats['errors'].append(error_msg)
                logger.print(f"Error: {error_msg}")
            
            # Update progress bar with current statistics
            pbar.set_postfix({
                'Success': stats['successful'],
                'Failed': stats['failed']
            })
    
    # Final statistics
    logger.print("\n" + "="*60)
    logger.print("PROCESSING SUMMARY")
    logger.print("="*60)
    logger.print(f"Total crop pairs processed: {stats['processed']}")
    logger.print(f"Successful: {stats['successful']}")
    logger.print(f"Failed: {stats['failed']}")
    
    if stats['successful'] > 0:
        success_rate = (stats['successful'] / stats['processed']) * 100
        logger.print(f"Success rate: {success_rate:.1f}%")
        
        # Verify output files
        actual_files = len(list(args.output_dir.glob("*.png")))
        logger.print(f"PNG files created: {actual_files}")
        
        if actual_files != stats['successful']:
            logger.print(f"Warning: Mismatch between successful processing ({stats['successful']}) and actual files ({actual_files})")

    # Label statistics
    if args.labels_dir:
        logger.print(f"Labels copied: {stats['labels_copied']}")
        logger.print(f"Labels not found: {stats['labels_not_found']}")
        logger.print(f"Label copy failures: {stats['labels_failed']}")

        if stats['labels_copied'] > 0:
            # Verify output label files
            actual_labels = len(list(args.output_labels_dir.glob("*.txt")))
            logger.print(f"TXT label files created: {actual_labels}")

            if actual_labels != stats['labels_copied']:
                logger.print(f"Warning: Mismatch between labels copied ({stats['labels_copied']}) and actual label files ({actual_labels})")
    else:
        logger.print("No labels directory specified - labels not processed")

    # PCF statistics
    if stats['pcf_stats']:
        pcf_mins = [s[0] for s in stats['pcf_stats']]
        pcf_maxs = [s[1] for s in stats['pcf_stats']]
        
        logger.print(f"\nPCF Statistics:")
        logger.print(f"  PCF min range: [{min(pcf_mins):.6f}, {max(pcf_mins):.6f}]")
        logger.print(f"  PCF max range: [{min(pcf_maxs):.6f}, {max(pcf_maxs):.6f}]")
        logger.print(f"  Overall PCF range: [{min(pcf_mins):.6f}, {max(pcf_maxs):.6f}]")
    
    # Error summary
    if stats['errors']:
        logger.print(f"\nError Summary ({len(stats['errors'])} errors):")
        # Group similar errors
        error_counts = {}
        for error in stats['errors']:
            error_type = error.split(':')[-1].strip() if ':' in error else error
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        for error_type, count in error_counts.items():
            logger.print(f"  {error_type}: {count} occurrences")
    
    print(f"\nProcessing complete. Log saved to: {log_filename}")
    logger.close()

if __name__ == "__main__":
    main()