#!/usr/bin/env python3

"""
batch_sar_processing.py

Author: Peter Millitz
Created: 2025-06-29

Batch processing for SAR data. Processes all .npy files in a directory
using the complex_scale_and_norm.py script. Processing parameters are 
are referenced from a separate YAML file ('config.yaml').
"""

import os
import subprocess
import sys
from pathlib import Path
import time
import yaml
from typing import Optional, List, Tuple


def load_config(config_file: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_file : str
        Path to YAML configuration file
    
    Returns:
    --------
    dict
        Configuration dictionary
    """
    default_config = {
        'input_dir': 'data/sar_crops',
        'output_dir': 'data/processed_crops',
        'script_path': 'complex_scale_and_norm.py',
        'nan_strategy': 'skip',
        'epsilon': 1e-6,
        'verbose': True,
        'global_norm_params': None,
        'max_workers': None,
        'file_pattern': '*.npy',
        'skip_existing': True,
        'log_file': None
    }
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            uni_config = yaml.safe_load(f) or {}
        
        user_config = uni_config['batch_sar_processing']
        
        # Merge with defaults
        config = {**default_config, **user_config}
        print(f"✓ Configuration loaded from: {config_file}")
    else:
        config = default_config
        print(f"✓ Using default configuration (no {config_file} found)")
    
    return config


def validate_config(config: dict) -> bool:
    """
    Validate configuration parameters.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    bool
        True if configuration is valid
    """
    if not os.path.exists(config['input_dir']):
        print(f"✗ Input directory does not exist: {config['input_dir']}")
        return False
    
    if not os.path.exists(config['script_path']):
        print(f"✗ Processing script not found: {config['script_path']}")
        return False
    
    if config['global_norm_params'] is not None:
        if len(config['global_norm_params']) != 2:
            print(f"✗ global_norm_params must have exactly 2 values (amp_min, amp_max), got {len(config['global_norm_params'])}")
            return False

        amp_min, amp_max = config['global_norm_params']
        
        # Handle amp_min = 0.0 case with float32-appropriate value
        if amp_min == 0.0:
            config['global_norm_params'][0] = 1e-6
            amp_min = 1e-6
        
        # Validate amplitude parameters
        if amp_min < 0 or amp_max <= 0 or amp_min >= amp_max:
            print(f"✗ Invalid amplitude parameters: amp_min must be non-negative, amp_max > 0, and amp_min < amp_max")
            return False
    
    return True


def setup_logging(config: dict):
    """Set up logging configuration"""
    if config['log_file']:
        import logging
        logging.basicConfig(
            filename=config['log_file'],
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'
        )
        return logging.getLogger()
    return None


def find_sar_files(input_dir: str, pattern: str = "*.npy") -> List[Path]:
    """
    Find all SAR .npy files in the input directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory to search for files
    pattern : str
        File pattern to match
    
    Returns:
    --------
    List[Path]
        List of file paths found
    """
    input_path = Path(input_dir)
    files = list(input_path.glob(pattern))
    files.sort()
    
    print(f"Found {len(files)} files matching pattern '{pattern}' in {input_dir}")
    
    return files


def get_output_path(input_file: Path, output_dir: str) -> Path:
    """Generate output path for processed file."""
    output_path = Path(output_dir)
    output_filename = f"{input_file.stem}_proc.npy"
    return output_path / output_filename


def build_command(input_file: Path, config: dict) -> List[str]:
    """Build command line arguments for the processing script."""
    cmd = [
        sys.executable,
        config['script_path'],
        str(input_file),
        "--output-dir", config['output_dir'],
        "--nan-strategy", config['nan_strategy'],
        "--epsilon", str(config['epsilon'])
    ]
    
    if config['verbose']:
        cmd.append("--verbose")
    
    if config['global_norm_params'] is not None:
        cmd.extend(["--global-norm-params"] + [str(x) for x in config['global_norm_params']])
    
    return cmd


def process_single_file(input_file: Path, config: dict, logger=None) -> Tuple[bool, str]:
    """Process a single SAR file."""
    try:
        # Check if output already exists
        if config['skip_existing']:
            output_file = get_output_path(input_file, config['output_dir'])
            if output_file.exists():
                return True, f"Skipped (output exists): {input_file.name}"
        
        # Build and execute command
        cmd = build_command(input_file, config)
        
        # Existing screen output (unchanged)
        if config['verbose']:
            print(f"Processing: {input_file.name}")
            print(f"Command: {' '.join(cmd)}")
        
        # Additional logging
        if logger:
            logger.info(f"Processing: {input_file.name}")
            logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Log subprocess output
        if logger:
            if result.stdout:
                logger.info(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.info(f"STDERR: {result.stderr}")
            logger.info(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            success_msg = f"Success: {input_file.name}"
            if logger:
                logger.info(success_msg)
            return True, success_msg
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            failure_msg = f"Failed: {input_file.name} - {error_msg}"
            if logger:
                logger.error(failure_msg)
            return False, failure_msg
            
    except subprocess.TimeoutExpired:
        timeout_msg = f"Timeout: {input_file.name}"
        if logger:
            logger.error(timeout_msg)
        return False, timeout_msg
    except Exception as e:
        error_msg = f"Error: {input_file.name} - {str(e)}"
        if logger:
            logger.error(error_msg)
        return False, error_msg


def process_sequential(files: List[Path], config: dict, logger=None) -> Tuple[int, int, List[str]]:
    """Process files sequentially."""
    successful = 0
    failed = 0
    errors = []
    
    for i, file_path in enumerate(files, 1):
        if config['verbose']:
            print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
        
        if logger:
            logger.info(f"Starting file {i}/{len(files)}: {file_path.name}")
        
        success, message = process_single_file(file_path, config, logger)
        
        if success:
            successful += 1
            if config['verbose']:
                print(f"✓ {message}")
        else:
            failed += 1
            errors.append(message)
            print(f"✗ {message}")  # Always show errors
    
    if logger:
        logger.info(f"Sequential processing completed: {successful} successful, {failed} failed")
    
    return successful, failed, errors


def process_parallel(files: List[Path], config: dict, logger=None) -> Tuple[int, int, List[str]]:
    """Process files in parallel using multiprocessing."""
    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed
    except ImportError:
        print("Warning: multiprocessing not available, falling back to sequential processing")
        return process_sequential(files, config, logger)
    
    successful = 0
    failed = 0
    errors = []
    
    max_workers = config['max_workers'] or min(4, os.cpu_count())
    
    if config['verbose']:
        print(f"Processing {len(files)} files using {max_workers} parallel workers...")
    else:
        print(f"Processing {len(files)} files with {max_workers} workers (verbose=False)...")
    
    if logger:
        logger.info(f"Starting parallel processing with {max_workers} workers")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_single_file, file_path, config, logger): file_path 
            for file_path in files
        }
        
        # Process completed jobs
        for i, future in enumerate(as_completed(future_to_file), 1):
            file_path = future_to_file[future]
            
            try:
                success, message = future.result()
                
                if success:
                    successful += 1
                    if config['verbose']:
                        print(f"[{i}/{len(files)}] ✓ {message}")
                else:
                    failed += 1
                    errors.append(message)
                    print(f"[{i}/{len(files)}] ✗ {message}")  # Always show errors
                    
            except Exception as e:
                failed += 1
                error_msg = f"Exception: {file_path.name} - {str(e)}"
                errors.append(error_msg)
                print(f"[{i}/{len(files)}] ✗ {error_msg}")  # Always show errors
    
    if logger:
        logger.info(f"Parallel processing completed: {successful} successful, {failed} failed")
    
    return successful, failed, errors


def batch_process_sar_data(config_file: str = "config.yaml"):
    """
    Main function to batch process SAR data.
    
    Parameters:
    -----------
    config_file : str
        Path to YAML configuration file
    """
    print("=" * 60)
    print("Batch SAR Data Processing")
    print("=" * 60)
    
    # Load configuration
    config = load_config(config_file)
    
    # Setup logging
    logger = setup_logging(config)
    if logger:
        print(f"✓ Logging enabled: {config['log_file']}")
        logger.info("Batch SAR processing started")
        logger.info(f"Configuration: {config}")
    
    # Validate configuration
    if not validate_config(config):
        return
    
    print("✓ Configuration validated")
    if logger:
        logger.info("Configuration validated successfully")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    print(f"✓ Output directory: {config['output_dir']}")
    if logger:
        logger.info(f"Output directory created/verified: {config['output_dir']}")
    
    # Find input files
    files = find_sar_files(config['input_dir'], config['file_pattern'])
    
    if not files:
        no_files_msg = "No files found to process!"
        print(no_files_msg)
        if logger:
            logger.warning(no_files_msg)
        return
    
    if logger:
        logger.info(f"Found {len(files)} files to process")
        logger.info(f"Files: {[f.name for f in files]}")
    
    # Display configuration
    print(f"\nProcessing Configuration:")
    print(f"  Input directory: {config['input_dir']}")
    print(f"  Output directory: {config['output_dir']}")
    print(f"  NaN strategy: {config['nan_strategy']}")
    print(f"  Epsilon: {config['epsilon']}")
    print(f"  Global normalisation: {'Yes' if config['global_norm_params'] else 'Adaptive'}")
    if config['global_norm_params']:
        print(f"    Amplitude parameters: {config['global_norm_params']}")
        print(f"    Phase normalization: automatic via sine/cosine transformation")
    print(f"  Parallel processing: {'Yes' if config['max_workers'] else 'Sequential'}")
    print(f"  Skip existing: {config['skip_existing']}")
    
    # Start processing
    start_time = time.time()
    
    if logger:
        logger.info("Starting batch processing")
        logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    if config['max_workers']:
        successful, failed, errors = process_parallel(files, config, logger)
    else:
        successful, failed, errors = process_sequential(files, config, logger)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Total files: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Processing time: {processing_time:.2f} seconds")
    
    # Log summary
    if logger:
        logger.info("Batch processing completed")
        logger.info(f"Total files: {len(files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    
    if errors:
        print(f"\nErrors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        
        # Log all errors
        if logger:
            logger.info("Errors encountered:")
            for error in errors:
                logger.error(f"  {error}")


def create_sample_config(filename: str = "config.yaml"):
    """Create a sample configuration file."""
    sample_config = {
        'input_dir': 'data/sar_crops',
        'output_dir': 'data/processed_crops',
        'script_path': 'complex_scale_and_norm.py',
        'nan_strategy': 'skip',
        'epsilon': 1e-6,
        'verbose': True,
        'global_norm_params': None,  # [0.001, 50.2] for global normalization
        'max_workers': None,  # 4 for parallel processing
        'file_pattern': '*.npy',
        'skip_existing': True,
        'log_file': None  # 'processing.log' to enable logging
    }
    
    with open(filename, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Sample configuration created: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process SAR data")
    parser.add_argument("--config", default="config.yaml", help="Configuration file (default: config.yaml)")
    parser.add_argument("--create-config", action="store_true", help="Create sample configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config(args.config)
    else:
        batch_process_sar_data(args.config)

