#!/usr/bin/env python3

"""
batch_sar_processing.py

Author: Peter Millitz
Created: 2025-06-18

This module contains a set of functions for batch processesing all .npy files
in a directory using the complex_scale_and_norm.py script. Contains a number of
flexible configuration options.
"""

import os
import subprocess
import sys
from pathlib import Path
import time
from typing import Optional, List, Tuple
import glob

# Configuration
class SARProcessingConfig:
    def __init__(self):
        # Directory paths
        self.input_dir = "data/sar_crops"  # Directory containing .npy files
        self.output_dir = "data/processed_crops"  # Output directory
        self.script_path = "complex_scale_and_norm.py"  # Path to the processing script
        
        # Processing parameters
        self.nan_strategy = "skip"  # Options: 'skip', 'zero', 'mean', 'interpolate'
        self.epsilon = 1e-6
        self.verbose = True
        
        # Global normalisation parameters (set to None for adaptive normalisation)
        # Format: [amp_min, amp_max, phase_min, phase_max]
        self.global_norm_params = None  # Example: [0.001, 50.2, -3.14159, 3.14159]
        
        # Processing options
        self.max_workers = None  # Number of parallel processes (None = sequential)
        self.file_pattern = "*.npy"  # File pattern to match
        self.skip_existing = True  # Skip files that already have processed outputs
        
    def validate(self):
        import numpy as np
        """Validate configuration parameters"""
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        if not os.path.exists(self.script_path):
            raise ValueError(f"Processing script not found: {self.script_path}")
        
        if self.global_norm_params is not None:
            if len(self.global_norm_params) != 4:
                raise ValueError("global_norm_params must have exactly 4 values")

            amp_min, amp_max, phase_min, phase_max = self.global_norm_params
            
            # Handle amp_min = 0.0 case
            if amp_min == 0.0:
                self.global_norm_params[0] = 1e-10
                amp_min = 1e-10
            
            # INDENT THESE LINES - they should be inside the global_norm_params check
            if amp_min < 0 or amp_max <= 0 or amp_min >= amp_max:
                raise ValueError("Invalid amplitude parameters")
            
            if phase_min <= -3.141592653589793 or phase_max > 3.141592653589793 or phase_min >= phase_max:
                raise ValueError("Invalid phase parameters (must be in (-π, π] range)")


def find_sar_files(input_dir: str, pattern: str = "*.npy") -> List[Path]:
    """
    Find all SAR .npy files in the input directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory to search for files
    pattern : str
        File pattern to match (default: "*.npy")
    
    Returns:
    --------
    List[Path]
        List of file paths found
    """
    input_path = Path(input_dir)
    files = list(input_path.glob(pattern))
    files.sort()  # Sort for consistent processing order
    
    print(f"Found {len(files)} files matching pattern '{pattern}' in {input_dir}")
    
    return files


def get_output_path(input_file: Path, output_dir: str) -> Path:
    """
    Generate output path for processed file.
    
    Parameters:
    -----------
    input_file : Path
        Input file path
    output_dir : str
        Output directory
    
    Returns:
    --------
    Path
        Expected output file path
    """
    output_path = Path(output_dir)
    output_filename = f"{input_file.stem}_proc.npy"
    return output_path / output_filename


def build_command(input_file: Path, config: SARProcessingConfig) -> List[str]:
    """
    Build command line arguments for the processing script.
    
    Parameters:
    -----------
    input_file : Path
        Input file to process
    config : SARProcessingConfig
        Configuration object
    
    Returns:
    --------
    List[str]
        Command line arguments
    """
    cmd = [
        sys.executable,  # Use same Python interpreter
        config.script_path,
        str(input_file),
        "--output-dir", config.output_dir,
        "--nan-strategy", config.nan_strategy,
        "--epsilon", str(config.epsilon)
    ]
    
    if config.verbose:
        cmd.append("--verbose")
    
    if config.global_norm_params is not None:
        cmd.extend(["--global-norm-params"] + [str(x) for x in config.global_norm_params])
    
    return cmd


def process_single_file(input_file: Path, config: SARProcessingConfig) -> Tuple[bool, str]:
    """
    Process a single SAR file.
    
    Parameters:
    -----------
    input_file : Path
        Input file to process
    config : SARProcessingConfig
        Configuration object
    
    Returns:
    --------
    Tuple[bool, str]
        (success, message)
    """
    try:
        # Check if output already exists
        if config.skip_existing:
            output_file = get_output_path(input_file, config.output_dir)
            if output_file.exists():
                return True, f"Skipped (output exists): {input_file.name}"
        
        # Build and execute command
        cmd = build_command(input_file, config)
        
        if config.verbose:
            print(f"Processing: {input_file.name}")
            print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per file
        )
        
        if result.returncode == 0:
            return True, f"Success: {input_file.name}"
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return False, f"Failed: {input_file.name} - {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout: {input_file.name}"
    except Exception as e:
        return False, f"Error: {input_file.name} - {str(e)}"


def process_sequential(files: List[Path], config: SARProcessingConfig) -> Tuple[int, int, List[str]]:
    """
    Process files sequentially.
    
    Parameters:
    -----------
    files : List[Path]
        List of files to process
    config : SARProcessingConfig
        Configuration object
    
    Returns:
    --------
    Tuple[int, int, List[str]]
        (successful_count, failed_count, error_messages)
    """
    successful = 0
    failed = 0
    errors = []
    
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
        
        success, message = process_single_file(file_path, config)
        
        if success:
            successful += 1
            if config.verbose:
                print(f"✓ {message}")
        else:
            failed += 1
            errors.append(message)
            print(f"✗ {message}")
    
    return successful, failed, errors


def process_parallel(files: List[Path], config: SARProcessingConfig) -> Tuple[int, int, List[str]]:
    """
    Process files in parallel using multiprocessing.
    
    Parameters:
    -----------
    files : List[Path]
        List of files to process
    config : SARProcessingConfig
        Configuration object
    
    Returns:
    --------
    Tuple[int, int, List[str]]
        (successful_count, failed_count, error_messages)
    """
    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed
    except ImportError:
        print("Warning: multiprocessing not available, falling back to sequential processing")
        return process_sequential(files, config)
    
    successful = 0
    failed = 0
    errors = []
    
    max_workers = config.max_workers or min(4, os.cpu_count())
    
    print(f"Processing {len(files)} files using {max_workers} parallel workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_single_file, file_path, config): file_path 
            for file_path in files
        }
        
        # Process completed jobs
        for i, future in enumerate(as_completed(future_to_file), 1):
            file_path = future_to_file[future]
            
            try:
                success, message = future.result()
                
                if success:
                    successful += 1
                    print(f"[{i}/{len(files)}] ✓ {message}")
                else:
                    failed += 1
                    errors.append(message)
                    print(f"[{i}/{len(files)}] ✗ {message}")
                    
            except Exception as e:
                failed += 1
                error_msg = f"Exception: {file_path.name} - {str(e)}"
                errors.append(error_msg)
                print(f"[{i}/{len(files)}] ✗ {error_msg}")
    
    return successful, failed, errors


def batch_process_sar_data(config: SARProcessingConfig):
    """
    Main function to batch process SAR data.
    
    Parameters:
    -----------
    config : SARProcessingConfig
        Configuration object
    """
    print("=" * 60)
    print("Batch SAR Data Processing")
    print("=" * 60)
    
    # Validate configuration
    try:
        config.validate()
        print("✓ Configuration validated")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"✓ Output directory: {config.output_dir}")
    
    # Find input files
    files = find_sar_files(config.input_dir, config.file_pattern)
    
    if not files:
        print("No files found to process!")
        return
    
    # Display configuration
    print(f"\nProcessing Configuration:")
    print(f"  Input directory: {config.input_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  NaN strategy: {config.nan_strategy}")
    print(f"  Epsilon: {config.epsilon}")
    print(f"  Global normalisation: {'Yes' if config.global_norm_params else 'Adaptive'}")
    if config.global_norm_params:
        print(f"    Parameters: {config.global_norm_params}")
    print(f"  Parallel processing: {'Yes' if config.max_workers else 'Sequential'}")
    print(f"  Skip existing: {config.skip_existing}")
    
    # Start processing
    start_time = time.time()
    
    if config.max_workers:
        successful, failed, errors = process_parallel(files, config)
    else:
        successful, failed, errors = process_sequential(files, config)
    
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Total files: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    if errors:
        print(f"\nErrors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


# Example usage functions for Jupyter notebook

def setup_basic_processing():
    """Set up basic processing configuration"""
    config = SARProcessingConfig()
    config.input_dir = "data/sar_crops"
    config.output_dir = "data/processed_crops"
    config.nan_strategy = "skip"
    config.verbose = True
    config.skip_existing = True
    return config


def setup_global_norm_processing():
    """Set up processing with global normalisation"""
    config = SARProcessingConfig()
    config.input_dir = "data/sar_crops"
    config.output_dir = "data/processed_crops_global"
    config.nan_strategy = "interpolate"
    config.verbose = True
    config.skip_existing = True
    
    # Set global normalisation parameters
    # [amp_min, amp_max, phase_min, phase_max]
    config.global_norm_params = [0.001, 50.2, -3.14159, 3.14159]
    
    return config


def setup_parallel_processing():
    """Set up parallel processing configuration"""
    config = SARProcessingConfig()
    config.input_dir = "data/sar_crops"
    config.output_dir = "data/processed_crops"
    config.nan_strategy = "zero"
    config.verbose = False  # Less verbose for parallel processing
    config.skip_existing = True
    config.max_workers = 4  # Use 4 parallel workers
    
    return config


# Quick start examples for Jupyter notebook

def quick_process_adaptive():
    """Quick start: Process all files with adaptive normalisation"""
    config = setup_basic_processing()
    batch_process_sar_data(config)


def quick_process_global():
    """Quick start: Process all files with global normalisation"""
    config = setup_global_norm_processing()
    batch_process_sar_data(config)


def quick_process_parallel():
    """Quick start: Process all files in parallel"""
    config = setup_parallel_processing()
    batch_process_sar_data(config)


# Jupyter notebook example cells
"""
# =============================================================================
# JUPYTER NOTEBOOK USAGE EXAMPLES
# =============================================================================

# Cell 1: Import and basic setup
from batch_sar_processing import *

# Cell 2: Quick processing with adaptive normalisation
quick_process_adaptive()

# Cell 3: Processing with global normalisation
quick_process_global()

# Cell 4: Parallel processing
quick_process_parallel()

# Cell 5: Custom configuration
config = SARProcessingConfig()
config.input_dir = "path/to/your/sar/crops"
config.output_dir = "path/to/output"
config.nan_strategy = "interpolate"
config.global_norm_params = [0.005, 25.0, -2.5, 2.5]
config.max_workers = 2
batch_process_sar_data(config)

# Cell 6: Check specific directory contents
files = find_sar_files("data/sar_crops")
print(f"Found {len(files)} files to process")
for f in files[:5]:  # Show first 5 files
    print(f"  {f.name}")

# Cell 7: Process only specific files
config = setup_basic_processing()
config.file_pattern = "subset_*.npy"  # Only process files starting with "subset_"
batch_process_sar_data(config)
"""
