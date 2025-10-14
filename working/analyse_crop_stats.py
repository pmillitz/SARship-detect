#!/usr/bin/env python3

"""
analyse_crop_stats.py

Author: Peter Millitz
Created: 2025-07-04

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm

def analyse_crop_statistics(base_dir, image_list=None, sample_size=None, 
                          compute_log=True, compute_real=True, percentiles=[1, 5, 25, 50, 75, 95, 99]):
    """
    Analyse statistics of SAR image crops to determine optimal normalisation parameters.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing 'images' subdirectory with .npy crop files
    image_list : list, optional
        Specific list of image filenames to analyse. If None, analyses all .npy files
    sample_size : int, optional
        Maximum number of files to analyse (for speed). If None, analyses all
    compute_log : bool
        Whether to compute log-magnitude statistics as well
    compute_real : bool
        Whether to compute real part statistics as well
    percentiles : list
        Percentiles to compute for the data distribution
    
    Returns:
    --------
    dict : Dictionary containing comprehensive statistics
    """
    
    def load_sar_magnitude(image_path):
        """Load SAR image and extract magnitude"""
        img_data = np.load(image_path)
        magnitude = np.abs(img_data)
        return magnitude
    
    def load_sar_log_magnitude(image_path):
        """Load SAR image and extract log-magnitude (dB)"""
        img_data = np.load(image_path)
        magnitude = np.abs(img_data)
        # Convert to dB, avoiding log(0)
        log_magnitude = 20 * np.log10(magnitude + 1e-10)
        return log_magnitude
    
    def load_sar_real_part(image_path):
        """Load SAR image and extract real part"""
        img_data = np.load(image_path)
        real_part = np.real(img_data)
        return real_part
    
    # Get list of files to analyze
    images_dir = Path(base_dir) / 'images'
    
    if image_list is not None:
        # Use provided list
        file_paths = [images_dir / filename for filename in image_list]
        file_paths = [p for p in file_paths if p.exists()]  # Filter existing files
    else:
        # Get all .npy files
        file_paths = list(images_dir.glob('*.npy'))
    
    # Sample if requested
    if sample_size is not None and len(file_paths) > sample_size:
        np.random.seed(42)  # For reproducible sampling
        file_paths = np.random.choice(file_paths, sample_size, replace=False)
    
    print(f"Analysing {len(file_paths)} crop files...")
    
    # Collect all pixel values
    magnitude_values = []
    log_magnitude_values = []
    real_part_values = []
    
    # Per-image statistics for distribution analysis
    per_image_stats = {
        'magnitude': {'mins': [], 'maxs': [], 'means': [], 'stds': []},
        'log_magnitude': {'mins': [], 'maxs': [], 'means': [], 'stds': []},
        'real_part': {'mins': [], 'maxs': [], 'means': [], 'stds': []}
    }
    
    # Process each file
    for file_path in tqdm(file_paths, desc="Processing crops"):
        try:
            # Load magnitude
            mag_img = load_sar_magnitude(file_path)
            magnitude_values.extend(mag_img.flatten())
            
            # Per-image stats for magnitude
            per_image_stats['magnitude']['mins'].append(np.min(mag_img))
            per_image_stats['magnitude']['maxs'].append(np.max(mag_img))
            per_image_stats['magnitude']['means'].append(np.mean(mag_img))
            per_image_stats['magnitude']['stds'].append(np.std(mag_img))
            
            if compute_log:
                # Load log-magnitude
                log_mag_img = load_sar_log_magnitude(file_path)
                log_magnitude_values.extend(log_mag_img.flatten())
                
                # Per-image stats for log-magnitude
                per_image_stats['log_magnitude']['mins'].append(np.min(log_mag_img))
                per_image_stats['log_magnitude']['maxs'].append(np.max(log_mag_img))
                per_image_stats['log_magnitude']['means'].append(np.mean(log_mag_img))
                per_image_stats['log_magnitude']['stds'].append(np.std(log_mag_img))
            
            if compute_real:
                # Load real part
                real_img = load_sar_real_part(file_path)
                real_part_values.extend(real_img.flatten())
                
                # Per-image stats for real part
                per_image_stats['real_part']['mins'].append(np.min(real_img))
                per_image_stats['real_part']['maxs'].append(np.max(real_img))
                per_image_stats['real_part']['means'].append(np.mean(real_img))
                per_image_stats['real_part']['stds'].append(np.std(real_img))
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Convert to numpy arrays for analysis
    magnitude_values = np.array(magnitude_values)
    if compute_log:
        log_magnitude_values = np.array(log_magnitude_values)
    if compute_real:
        real_part_values = np.array(real_part_values)
    
    # Compute comprehensive statistics
    stats = {
        'files_analyzed': len(file_paths),
        'total_pixels': len(magnitude_values)
    }
    
    # Magnitude statistics
    mag_percentiles = np.percentile(magnitude_values, percentiles)
    stats['magnitude'] = {
        'min': np.min(magnitude_values),
        'max': np.max(magnitude_values),
        'mean': np.mean(magnitude_values),
        'std': np.std(magnitude_values),
        'percentiles': dict(zip(percentiles, mag_percentiles)),
        'per_image_stats': per_image_stats['magnitude']
    }
    
    # Log-magnitude statistics
    if compute_log:
        log_percentiles = np.percentile(log_magnitude_values, percentiles)
        stats['log_magnitude'] = {
            'min': np.min(log_magnitude_values),
            'max': np.max(log_magnitude_values),
            'mean': np.mean(log_magnitude_values),
            'std': np.std(log_magnitude_values),
            'percentiles': dict(zip(percentiles, log_percentiles)),
            'per_image_stats': per_image_stats['log_magnitude']
        }
    
    # Real part statistics
    if compute_real:
        real_percentiles = np.percentile(real_part_values, percentiles)
        stats['real_part'] = {
            'min': np.min(real_part_values),
            'max': np.max(real_part_values),
            'mean': np.mean(real_part_values),
            'std': np.std(real_part_values),
            'percentiles': dict(zip(percentiles, real_percentiles)),
            'per_image_stats': per_image_stats['real_part']
        }
    
    return stats, magnitude_values, log_magnitude_values if compute_log else None, real_part_values if compute_real else None


def print_statistics_summary(stats):
    """Print a formatted summary of the statistics"""
    print("\n" + "="*60)
    print("SAR CROP STATISTICS SUMMARY")
    print("="*60)
    print(f"Files analyzed: {stats['files_analyzed']:,}")
    print(f"Total pixels: {stats['total_pixels']:,}")
    
    print(f"\nMAGNITUDE STATISTICS:")
    print(f"  Min: {stats['magnitude']['min']:.2f}")
    print(f"  Max: {stats['magnitude']['max']:.2f}")
    print(f"  Mean: {stats['magnitude']['mean']:.2f}")
    print(f"  Std: {stats['magnitude']['std']:.2f}")
    
    print(f"\n  Percentiles:")
    for p, val in stats['magnitude']['percentiles'].items():
        print(f"    {p:2d}%: {val:8.2f}")
    
    if 'log_magnitude' in stats:
        print(f"\nLOG-MAGNITUDE (dB) STATISTICS:")
        print(f"  Min: {stats['log_magnitude']['min']:.2f} dB")
        print(f"  Max: {stats['log_magnitude']['max']:.2f} dB")
        print(f"  Mean: {stats['log_magnitude']['mean']:.2f} dB")
        print(f"  Std: {stats['log_magnitude']['std']:.2f} dB")
        
        print(f"\n  Percentiles:")
        for p, val in stats['log_magnitude']['percentiles'].items():
            print(f"    {p:2d}%: {val:8.2f} dB")

    if 'real_part' in stats:
        print(f"\nREAL PART STATISTICS:")
        print(f"  Min: {stats['real_part']['min']:.2f}")
        print(f"  Max: {stats['real_part']['max']:.2f}")
        print(f"  Mean: {stats['real_part']['mean']:.2f}")
        print(f"  Std: {stats['real_part']['std']:.2f}")
        
        print(f"\n  Percentiles:")
        for p, val in stats['real_part']['percentiles'].items():
            print(f"    {p:2d}%: {val:8.2f}")
    
    print("\n" + "="*60)
    print("RECOMMENDED NORMALISATION PARAMETERS:")
    print("="*60)
    
    # Recommendations based on percentiles
    mag_99 = stats['magnitude']['percentiles'][99]
    mag_1 = stats['magnitude']['percentiles'][1]
    
    print(f"For MAGNITUDE (99% coverage):")
    print(f"  amp_min = {mag_1:.1f}")
    print(f"  amp_max = {mag_99:.1f}")
    
    if 'log_magnitude' in stats:
        log_99 = stats['log_magnitude']['percentiles'][99]
        log_1 = stats['log_magnitude']['percentiles'][1]
        log_95 = stats['log_magnitude']['percentiles'][95]
        log_5 = stats['log_magnitude']['percentiles'][5]
        
        print(f"\nFor LOG-MAGNITUDE (99% coverage):")
        print(f"  log_min = {log_1:.1f}")
        print(f"  log_max = {log_99:.1f}")
        
        print(f"\nFor LOG-MAGNITUDE (95% coverage - more conservative):")
        print(f"  log_min = {log_5:.1f}")
        print(f"  log_max = {log_95:.1f}")

    if 'real_part' in stats:
        real_99 = stats['real_part']['percentiles'][99]
        real_1 = stats['real_part']['percentiles'][1]
        real_95 = stats['real_part']['percentiles'][95]
        real_5 = stats['real_part']['percentiles'][5]
        
        print(f"\nFor REAL PART (99% coverage):")
        print(f"  real_min = {real_1:.1f}")
        print(f"  real_max = {real_99:.1f}")
        
        print(f"\nFor REAL PART (95% coverage - more conservative):")
        print(f"  real_min = {real_5:.1f}")
        print(f"  real_max = {real_95:.1f}")


def plot_distribution_analysis(stats, magnitude_values, log_magnitude_values=None, real_part_values=None):
    """Plot histograms and distribution analysis"""
    # Determine number of subplots needed
    n_plots = 1  # Always have magnitude
    if log_magnitude_values is not None:
        n_plots += 1
    if real_part_values is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(2, n_plots, figsize=(5*n_plots, 10))
    if n_plots == 1:
        axes = axes.reshape(-1, 1)  # Ensure 2D array
    
    # Magnitude histogram
    axes[0,0].hist(magnitude_values, bins=100, alpha=0.7, density=True)
    axes[0,0].set_xlabel('Magnitude')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Magnitude Distribution')
    axes[0,0].axvline(stats['magnitude']['percentiles'][99], color='red', linestyle='--', 
                     label=f"99%: {stats['magnitude']['percentiles'][99]:.1f}")
    axes[0,0].axvline(stats['magnitude']['percentiles'][1], color='red', linestyle='--',
                     label=f"1%: {stats['magnitude']['percentiles'][1]:.1f}")
    axes[0,0].legend()
    
    # Magnitude histogram (log scale)
    axes[1,0].hist(magnitude_values, bins=100, alpha=0.7, density=True)
    axes[1,0].set_xlabel('Magnitude')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Magnitude Distribution (Log Y-axis)')
    axes[1,0].set_yscale('log')
    
    plot_idx = 1
    
    if log_magnitude_values is not None:
        # Log-magnitude histogram
        axes[0,plot_idx].hist(log_magnitude_values, bins=100, alpha=0.7, density=True)
        axes[0,plot_idx].set_xlabel('Log-Magnitude (dB)')
        axes[0,plot_idx].set_ylabel('Density')
        axes[0,plot_idx].set_title('Log-Magnitude Distribution')
        axes[0,plot_idx].axvline(stats['log_magnitude']['percentiles'][99], color='red', linestyle='--',
                         label=f"99%: {stats['log_magnitude']['percentiles'][99]:.1f}")
        axes[0,plot_idx].axvline(stats['log_magnitude']['percentiles'][1], color='red', linestyle='--',
                         label=f"1%: {stats['log_magnitude']['percentiles'][1]:.1f}")
        axes[0,plot_idx].legend()
        
        # Per-image max values distribution
        axes[1,plot_idx].hist(stats['log_magnitude']['per_image_stats']['maxs'], bins=50, alpha=0.7)
        axes[1,plot_idx].set_xlabel('Per-Image Maximum (dB)')
        axes[1,plot_idx].set_ylabel('Count')
        axes[1,plot_idx].set_title('Distribution of Log-Magnitude Maximums')
        
        plot_idx += 1
    
    if real_part_values is not None:
        # Real part histogram
        axes[0,plot_idx].hist(real_part_values, bins=100, alpha=0.7, density=True)
        axes[0,plot_idx].set_xlabel('Real Part')
        axes[0,plot_idx].set_ylabel('Density')
        axes[0,plot_idx].set_title('Real Part Distribution')
        axes[0,plot_idx].axvline(stats['real_part']['percentiles'][99], color='red', linestyle='--',
                         label=f"99%: {stats['real_part']['percentiles'][99]:.1f}")
        axes[0,plot_idx].axvline(stats['real_part']['percentiles'][1], color='red', linestyle='--',
                         label=f"1%: {stats['real_part']['percentiles'][1]:.1f}")
        axes[0,plot_idx].legend()
        
        # Per-image real part max/min distribution
        axes[1,plot_idx].hist(stats['real_part']['per_image_stats']['maxs'], bins=50, alpha=0.7, 
                             label='Max values', color='blue')
        axes[1,plot_idx].hist(stats['real_part']['per_image_stats']['mins'], bins=50, alpha=0.7,
                             label='Min values', color='red')
        axes[1,plot_idx].set_xlabel('Per-Image Extrema')
        axes[1,plot_idx].set_ylabel('Count')
        axes[1,plot_idx].set_title('Distribution of Real Part Extrema')
        axes[1,plot_idx].legend()
    
    plt.tight_layout()

    # Add figure caption below the figure
    caption_text = (
        'Statistical distributions of SLC crop pixel values. Top row: relative frequency density '
        'distributions with 1st and 99th percentile markers (red dashed lines). Bottom left: '
        'magnitude distribution on logarithmic y-axis. Bottom centre: distribution of per-image '
        'log-magnitude maximum values. Bottom right: distribution of per-image real part extrema '
        '(max and min values).'
    )

    fig.text(0.1, 0.01, caption_text, ha='left', va='bottom', fontsize=10,
             wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.subplots_adjust(bottom=0.12)  # Make room for caption

    plt.show()


# Example usage:
if __name__ == "__main__":
    # Analyse all crops in directory
    stats, mag_vals, log_vals, real_vals = analyse_crop_statistics('cropping', sample_size=1000)
    
    # Print summary
    print_statistics_summary(stats)
    
    # Plot distributions
    plot_distribution_analysis(stats, mag_vals, log_vals, real_vals)

