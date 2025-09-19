#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from tqdm import tqdm

def analyze_raw_pcf(npy_dir):
    """Analyze raw PCF values from channel 2 of numpy arrays."""

    npy_files = list(npy_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} numpy files")

    all_pcf_values = []

    # Load all PCF values (channel 2)
    for npy_file in tqdm(npy_files, desc="Loading arrays"):
      try:
          array = np.load(npy_file)  # Shape: (96, 96, 3)
          pcf_channel = array[:, :, 2]  # Extract blue channel (raw PCF)
          all_pcf_values.extend(pcf_channel.flatten())
      except Exception as e:
          print(f"Error loading {npy_file}: {e}")

    # Convert to numpy array
    all_pcf = np.array(all_pcf_values)

    # Compute statistics
    print(f"\nRAW PCF STATISTICS:")
    print(f"Total pixels: {len(all_pcf):,}")
    print(f"Min: {all_pcf.min():.6f}")
    print(f"Max: {all_pcf.max():.6f}")
    print(f"Mean: {all_pcf.mean():.6f}")
    print(f"Std: {all_pcf.std():.6f}")
    print(f"Zeros: {np.sum(all_pcf == 0):,} ({np.sum(all_pcf == 0)/len(all_pcf)*100:.1f}%)")

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    print(f"\nPercentiles:")
    for p in percentiles:
      value = np.percentile(all_pcf, p)
      print(f"  {p:5.1f}th: {value:.6f}")

    # Non-zero percentiles if many zeros
    non_zero = all_pcf[all_pcf > 0]
    if len(non_zero) != len(all_pcf):
      print(f"\nNon-zero percentiles:")
      for p in percentiles:
          value = np.percentile(non_zero, p)
          print(f"  {p:5.1f}th: {value:.6f}")

    # 99% coverage recommendation
    p1 = np.percentile(all_pcf, 1)
    p99 = np.percentile(all_pcf, 99)
    print(f"\n99% Coverage Range: [{p1:.6f}, {p99:.6f}]")

    return all_pcf
