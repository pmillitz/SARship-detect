#!/usr/bin/env python3
"""
IoU Analysis Script for YOLO Label Comparison

Runs label comparison across a range of IoU thresholds and outputs results
for plotting correct detection rates vs IoU threshold.
"""

import os
import sys
import csv
import argparse
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Import the comparison function from the label comparison script
from label_comparison import compare_labels

# Optional plotting imports - only import if available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def run_iou_analysis(gt_dir: str, pred_dir: str, iou_start: float = 0.1,
                    iou_end: float = 0.9, iou_step: float = 0.1,
                    output_csv: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
    """
    Run label comparison across a range of IoU thresholds.

    Args:
        gt_dir: Directory containing ground truth label files
        pred_dir: Directory containing predicted label files
        iou_start: Starting IoU threshold
        iou_end: Ending IoU threshold (inclusive)
        iou_step: Step size for IoU threshold
        output_csv: Path to save CSV results (optional)
        verbose: Print progress to screen (default: True)

    Returns:
        pandas DataFrame containing results for each IoU threshold
    """

    # Generate IoU thresholds
    iou_thresholds = np.arange(iou_start, iou_end + iou_step/2, iou_step)
    iou_thresholds = [round(iou, 1) for iou in iou_thresholds]  # Round to avoid float precision issues

    all_results = []

    if verbose:
        print(f"Running IoU analysis from {iou_start} to {iou_end} with step {iou_step}")
        print("="*80)
        print(f"{'IoU':<5} {'Total':<6} {'Correct':<8} {'Rate%':<8} {'Class Breakdown':<30}")
        print("-"*80)

    for iou_threshold in iou_thresholds:
        # Run comparison for this IoU threshold
        results = compare_labels(gt_dir, pred_dir, iou_threshold)

        # Extract overall metrics
        total_gt = results['total_gt_objects']
        correct_detections = results['correct_detections']
        overall_rate = (correct_detections / total_gt * 100) if total_gt > 0 else 0.0

        # Extract class-wise metrics
        class_results = {}
        class_breakdown_str = ""
        for class_id, stats in results['class_breakdown'].items():
            if stats['gt_count'] > 0:
                class_rate = (stats['correct'] / stats['gt_count']) * 100
                class_results[f'class_{class_id}_count'] = stats['correct']
                class_results[f'class_{class_id}_total'] = stats['gt_count']
                class_results[f'class_{class_id}_rate'] = class_rate
                class_breakdown_str += f"C{class_id}:{class_rate:.1f}% "
            else:
                class_results[f'class_{class_id}_count'] = 0
                class_results[f'class_{class_id}_total'] = 0
                class_results[f'class_{class_id}_rate'] = 0.0

        # Store results
        result_dict = {
            'iou_threshold': iou_threshold,
            'total_gt_objects': total_gt,
            'correct_detections': correct_detections,
            'overall_rate': overall_rate,
            **class_results
        }
        all_results.append(result_dict)

        # Print to screen
        if verbose:
            print(f"{iou_threshold:<5.1f} {total_gt:<6} {correct_detections:<8} {overall_rate:<8.1f} {class_breakdown_str:<30}")

    if verbose:
        print("-"*80)

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)

    # Save to CSV if specified
    if output_csv:
        df_results.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nResults saved to: {output_csv}")

    return df_results


def plot_iou_analysis(df_results: pd.DataFrame, save_path: Optional[str] = None,
                     show_classes: bool = True, figsize: tuple = (10, 6)):
    """
    Plot IoU analysis results.

    Args:
        df_results: DataFrame from run_iou_analysis()
        save_path: Path to save plot (optional)
        show_classes: Whether to show per-class curves (default: True)
        figsize: Figure size tuple (default: (10, 6))
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create plots.")
        return

    plt.figure(figsize=figsize)

    # Plot overall detection rate
    plt.plot(df_results['iou_threshold'], df_results['overall_rate'],
             'b-o', linewidth=2, markersize=6, label='Overall')

    # Plot per-class rates if requested
    if show_classes:
        # Find all class columns
        class_rate_cols = [col for col in df_results.columns if col.endswith('_rate') and col.startswith('class_')]
        colors = ['r', 'g', 'orange', 'purple', 'brown']
        markers = ['s', '^', 'D', 'v', '*']

        for i, col in enumerate(class_rate_cols):
            class_id = col.split('_')[1]  # Extract class ID
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.plot(df_results['iou_threshold'], df_results[col],
                    '--', color=color, marker=marker, linewidth=1.5, markersize=4, label=f'Class {class_id}')

    plt.xlabel('IoU Threshold')
    plt.ylabel('Correct Detection Rate (%)')
    plt.title('Detection Performance vs IoU Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(df_results['iou_threshold'].min(), df_results['iou_threshold'].max())
    plt.ylim(0, 100)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def save_to_csv(results: List[Dict], csv_path: str):
    """Save results to CSV file."""
    if not results:
        print("No results to save")
        return

    # Get all possible column names from the first result
    fieldnames = list(results[0].keys())

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description='Analyze YOLO label comparison across IoU thresholds')
    parser.add_argument('gt_dir', help='Directory containing ground truth label files')
    parser.add_argument('pred_dir', help='Directory containing predicted label files')
    parser.add_argument('--iou-start', type=float, default=0.1,
                       help='Starting IoU threshold (default: 0.1)')
    parser.add_argument('--iou-end', type=float, default=0.9,
                       help='Ending IoU threshold (default: 0.9)')
    parser.add_argument('--iou-step', type=float, default=0.1,
                       help='IoU threshold step size (default: 0.1)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output CSV file path (optional)')

    args = parser.parse_args()

    # Validate directories
    if not os.path.exists(args.gt_dir):
        print(f"Error: Ground truth directory '{args.gt_dir}' does not exist")
        sys.exit(1)

    if not os.path.exists(args.pred_dir):
        print(f"Error: Predictions directory '{args.pred_dir}' does not exist")
        sys.exit(1)

    # Set default output file if not specified
    output_csv = args.output
    if not output_csv:
        output_csv = "iou_analysis_results.csv"

    # Run analysis
    results = run_iou_analysis(
        args.gt_dir,
        args.pred_dir,
        args.iou_start,
        args.iou_end,
        args.iou_step,
        output_csv
    )

    print(f"\nAnalysis complete! Results for {len(results)} IoU thresholds.")

    # Show summary
    if results:
        best_iou = max(results, key=lambda x: x['overall_rate'])
        print(f"Best overall rate: {best_iou['overall_rate']:.1f}% at IoU {best_iou['iou_threshold']}")


if __name__ == "__main__":
    main()