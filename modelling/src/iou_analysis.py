#!/usr/bin/env python3
"""
IoU Analysis Script for YOLO Label Comparison

Runs label comparison across a range of IoU thresholds and outputs results
for plotting correct detection rates vs IoU threshold.

Example usage:
    # Run analysis from command line
    python iou_analysis.py ground_truth_labels/ predicted_labels/ --output results.csv

    # Use in Python code
    from iou_analysis import run_iou_analysis, plot_iou_analysis

    df = run_iou_analysis('gt_labels/', 'pred_labels/', iou_start=0.1, iou_end=0.9)
    plot_iou_analysis(df, save_path='iou_plot.png', show_plot=False)
"""

import os
import sys
import argparse
from typing import Dict, Optional
import numpy as np
import pandas as pd


def compute_auc_all(df, normalize=True):
    """
    Compute AUC for all rate columns (overall + per-class) in the given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
          - 'iou_threshold'
          - any number of rate columns (e.g., 'overall_rate', 'class_0_rate', 'class_1_rate', ...)
    normalize : bool, default=True
        If True, normalize by IoU range so the AUC is in [0, 1].

    Returns
    -------
    auc_dict : dict
        Mapping from column name → AUC value (normalized if normalize=True)
    """

    # Extract IoU thresholds and compute the range
    x = df["iou_threshold"].to_numpy()
    dx = x.max() - x.min()

    # Find all columns that end with '_rate'
    rate_cols = [col for col in df.columns if col.endswith("_rate")]

    auc_dict = {}
    for col in rate_cols:
        y = df[col].to_numpy()

        # Convert from percent to fraction if necessary
        if y.max() > 1:
            y = y / 100.0

        auc = np.trapz(y, x)
        if normalize and dx != 0:
            auc /= dx

        auc_dict[col] = auc

    return auc_dict

# Import the comparison function from the label comparison script
try:
    from label_comparison import compare_labels
except ImportError:
    print("Error: label_comparison.py not found. Please ensure it exists in the same directory.")
    sys.exit(1)

# Optional plotting imports - only import if available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def run_iou_analysis(gt_dir: str, pred_dir: str, iou_start: float = 0.05,
                    iou_end: float = 0.95, iou_step: float = 0.05,
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

    # Determine appropriate rounding precision based on step size
    # Count decimal places in step size to preserve precision
    step_str = f"{iou_step:.10f}".rstrip('0')  # Remove trailing zeros
    if '.' in step_str:
        decimal_places = len(step_str.split('.')[1])
    else:
        decimal_places = 0

    iou_thresholds = [round(iou, decimal_places) for iou in iou_thresholds]

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
            # Format IoU threshold with appropriate precision
            iou_str = f"{iou_threshold:.{decimal_places}f}"
            print(f"{iou_str:<5} {total_gt:<6} {correct_detections:<8} {overall_rate:<8.1f} {class_breakdown_str:<30}")

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
                     show_classes: bool = True, figsize: tuple = (10, 6),
                     title_suffix: str = "", show_plot: bool = True,
                     class_names: Optional[Dict[int, str]] = None):
    """
    Plot IoU analysis results.

    Args:
        df_results: DataFrame from run_iou_analysis()
        save_path: Path to save plot (optional)
        show_classes: Whether to show per-class curves (default: True)
        figsize: Figure size tuple (default: (10, 6))
        title_suffix: Additional text to append to plot title (default: "")
        show_plot: Whether to display plot interactively (default: True)
        class_names: Optional dict mapping class IDs to names (e.g., {0: 'is_vessel', 1: 'is_fishing'})
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create plots.")
        return

    plt.figure(figsize=figsize)

    # Plot overall detection rate
    plt.plot(df_results['iou_threshold'], df_results['overall_rate'],
             'b-o', linewidth=2, markersize=6, label='all', markevery=1)

    # Plot per-class rates if requested
    if show_classes:
        # Find all class columns
        class_rate_cols = [col for col in df_results.columns if col.endswith('_rate') and col.startswith('class_')]
        colors = ['r', 'g', 'orange', 'purple', 'brown']
        markers = ['s', '^', 'D', 'v', '*']

        # Default class names
        default_names = {0: 'is_vessel', 1: 'is_fishing'}

        for i, col in enumerate(class_rate_cols):
            class_id = int(col.split('_')[1])  # Extract class ID
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            # Use provided class names, fall back to defaults, then generic label
            if class_names:
                label = class_names.get(class_id, default_names.get(class_id, f'Class {class_id}'))
            else:
                label = default_names.get(class_id, f'Class {class_id}')

            plt.plot(df_results['iou_threshold'], df_results[col],
                    '--', color=color, marker=marker, linewidth=1.5, markersize=4, label=label, markevery=1)

    plt.xlabel('IoU Threshold', fontsize=16, fontweight='bold')
    plt.ylabel('Detection Rate (%)', fontsize=16, fontweight='bold')
    # No title - user will add figure caption in document
    plt.grid(True, alpha=0.3)

    # Place legend in bottom left quadrant with larger font
    plt.legend(loc='lower left', fontsize=14, framealpha=0.95,
               edgecolor='black', fancybox=True)

    # Set x and y limits based on data
    iou_min = df_results['iou_threshold'].min()
    iou_max = df_results['iou_threshold'].max()
    plt.xlim(iou_min, iou_max)
    plt.ylim(0, 100)

    # Set x-axis ticks at 0.1 intervals within data range, y-axis at 10% intervals
    x_tick_start = np.floor(iou_min * 10) / 10
    x_ticks = np.arange(x_tick_start, iou_max + 0.05, 0.1)
    plt.xticks(x_ticks, fontsize=14)
    plt.yticks(np.arange(0, 101, 10), fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_multimodel_comparison(model_results: Dict[str, pd.DataFrame],
                               class_id: Optional[int] = None,
                               save_path: Optional[str] = None,
                               figsize: tuple = (10, 6),
                               show_plot: bool = True,
                               class_names: Optional[Dict[int, str]] = None) -> None:
    """
    Plot IoU analysis comparing multiple models for a specific class or overall performance.

    Args:
        model_results: Dictionary mapping model names to DataFrames from run_iou_analysis()
                      Example: {'Model A': df_a, 'Model B': df_b, 'Model C': df_c}
        class_id: Class ID to plot (0, 1, etc.) or None for overall performance
        save_path: Path to save plot (optional)
        figsize: Figure size tuple (default: (10, 6))
        show_plot: Whether to display plot interactively (default: True)
        class_names: Optional dict mapping class IDs to names (e.g., {0: 'is_vessel', 1: 'is_fishing'})

    Example:
        # Run analysis for three models
        df_model1 = run_iou_analysis(gt_dir, pred_dir1, output_csv='model1.csv')
        df_model2 = run_iou_analysis(gt_dir, pred_dir2, output_csv='model2.csv')
        df_model3 = run_iou_analysis(gt_dir, pred_dir3, output_csv='model3.csv')

        models = {
            'YOLOv8n': df_model1,
            'YOLOv8s': df_model2,
            'YOLOv8m': df_model3
        }

        class_names = {0: 'is_vessel', 1: 'is_fishing'}

        # Plot for each class and overall
        plot_multimodel_comparison(models, class_id=0, save_path='class0_comparison.png',
                                   class_names=class_names)
        plot_multimodel_comparison(models, class_id=1, save_path='class1_comparison.png',
                                   class_names=class_names)
        plot_multimodel_comparison(models, class_id=None, save_path='overall_comparison.png')
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create plots.")
        return

    if not model_results:
        print("Error: No model results provided.")
        return

    plt.figure(figsize=figsize)

    # Define colors and markers for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '*']
    linestyles = ['-', '--', '-.', ':', '-', '--']

    # Determine what to plot
    if class_id is None:
        # Plot overall performance
        y_column = 'overall_rate'
        class_name = 'all'
    else:
        # Plot specific class performance
        y_column = f'class_{class_id}_rate'
        # Use class names with defaults
        default_names = {0: 'is_vessel', 1: 'is_fishing'}
        if class_names:
            class_name = class_names.get(class_id, default_names.get(class_id, f'Class {class_id}'))
        else:
            class_name = default_names.get(class_id, f'Class {class_id}')

    # Plot each model
    for i, (model_name, df_results) in enumerate(model_results.items()):
        if y_column not in df_results.columns:
            print(f"Warning: Column '{y_column}' not found in results for {model_name}")
            continue

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]

        plt.plot(df_results['iou_threshold'], df_results[y_column],
                linestyle=linestyle, color=color, marker=marker,
                linewidth=2, markersize=6, label=model_name, markevery=1)

    # Configure plot
    plt.xlabel('IoU Threshold', fontsize=16, fontweight='bold')
    plt.ylabel('Detection Rate (%)', fontsize=16, fontweight='bold')
    # No title - user will add figure caption in document
    plt.grid(True, alpha=0.3, linestyle='--')

    # Place legend in bottom left quadrant with larger font
    plt.legend(loc='lower left', fontsize=14, framealpha=0.95,
               edgecolor='black', fancybox=True)

    # Add class name box in top right corner
    # Format class name for display
    if class_id is None:
        display_name = 'All classes'
    else:
        display_name = class_name

    # Create text box with subtle styling
    ax = plt.gca()
    textstr = display_name
    props = dict(boxstyle='round,pad=0.5,rounding_size=0.15', facecolor='white',
                 edgecolor='black', linewidth=1.5, alpha=0.95)
    ax.text(0.95, 0.94, textstr, transform=ax.transAxes,
            fontsize=15, verticalalignment='top', horizontalalignment='right',
            bbox=props)

    # Get IoU range from first model
    first_df = next(iter(model_results.values()))
    iou_min = first_df['iou_threshold'].min()
    iou_max = first_df['iou_threshold'].max()
    plt.xlim(iou_min, iou_max)
    plt.ylim(0, 100)

    # Set x-axis ticks at 0.1 intervals within data range, y-axis at 10% intervals
    x_tick_start = np.floor(iou_min * 10) / 10
    x_ticks = np.arange(x_tick_start, iou_max + 0.05, 0.1)
    plt.xticks(x_ticks, fontsize=14)
    plt.yticks(np.arange(0, 101, 10), fontsize=14)

    # Add minor gridlines
    plt.grid(True, which='minor', alpha=0.1, linestyle=':')
    plt.minorticks_on()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_all_classes_multimodel_combined(model_results: Dict[str, pd.DataFrame],
                                         save_path: Optional[str] = None,
                                         figsize: tuple = (18, 5),
                                         show_plot: bool = True,
                                         class_names: Optional[Dict[int, str]] = None) -> None:
    """
    Plot all classes (overall, class 0, class 1) side-by-side in a single figure.
    Optimized for A4 PDF thesis documents with shared y-axis and readable fonts.

    Args:
        model_results: Dictionary mapping model names to DataFrames from run_iou_analysis()
        save_path: Path to save the combined plot (optional)
        figsize: Figure size tuple (default: (18, 6) for side-by-side layout)
        show_plot: Whether to display plot interactively (default: True)
        class_names: Optional dict mapping class IDs to names

    Example:
        models = {
            'Model A': df_model1,
            'Model B': df_model2,
            'Model C': df_model3
        }

        plot_all_classes_multimodel_combined(
            models,
            save_path='./thesis_figures/iou_comparison_combined.pdf',
            show_plot=False
        )
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create plots.")
        return

    if not model_results:
        print("Error: No model results provided.")
        return

    # Create figure with 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    # Define colors and markers for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '*']
    linestyles = ['-', '--', '-.', ':', '-', '--']

    # Default class names
    default_names = {0: 'is_vessel', 1: 'is_fishing'}

    # Define what to plot for each subplot
    plot_configs = [
        {'class_id': None, 'y_column': 'overall_rate', 'display_name': 'All classes'},
        {'class_id': 0, 'y_column': 'class_0_rate',
         'display_name': (class_names or default_names).get(0, 'is_vessel')},
        {'class_id': 1, 'y_column': 'class_1_rate',
         'display_name': (class_names or default_names).get(1, 'is_fishing')}
    ]

    # Get data range from first model to automatically set x-axis limits
    first_df = next(iter(model_results.values()))
    iou_min = first_df['iou_threshold'].min()
    iou_max = first_df['iou_threshold'].max()

    # Plot each class in its own subplot
    for ax_idx, config in enumerate(plot_configs):
        ax = axes[ax_idx]
        y_column = config['y_column']
        display_name = config['display_name']

        # Plot each model and compute AUC
        for i, (model_name, df_results) in enumerate(model_results.items()):
            if y_column not in df_results.columns:
                print(f"Warning: Column '{y_column}' not found in results for {model_name}")
                continue

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            linestyle = linestyles[i % len(linestyles)]

            # Compute AUC for this model and column
            auc_dict = compute_auc_all(df_results, normalize=True)
            auc_value = auc_dict.get(y_column, 0.0)

            # Create label with AUC in parentheses
            label_with_auc = f"{model_name} ({auc_value:.3f})"

            ax.plot(df_results['iou_threshold'], df_results[y_column],
                    linestyle=linestyle, color=color, marker=marker,
                    linewidth=2, markersize=6, label=label_with_auc, markevery=1)

        # Configure subplot
        ax.set_xlabel('IoU Threshold', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(iou_min, iou_max)
        ax.set_ylim(0, 100)

        # Set x-axis ticks at 0.1 intervals within the data range
        # Round down to nearest 0.1 for start tick
        x_tick_start = np.floor(iou_min * 10) / 10
        x_ticks = np.arange(x_tick_start, iou_max + 0.05, 0.1)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis='both', labelsize=14)

        # Add minor gridlines
        ax.grid(True, which='minor', alpha=0.1, linestyle=':')
        ax.minorticks_on()

        # Add class name box in top right corner
        textstr = display_name
        props = dict(boxstyle='round,pad=0.5,rounding_size=0.15', facecolor='white',
                     edgecolor='black', linewidth=1.5, alpha=0.95)
        ax.text(0.95, 0.94, textstr, transform=ax.transAxes,
                fontsize=15, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                bbox=props)

        # Add legend to each subplot (bottom left)
        ax.legend(loc='lower left', framealpha=0.95,
                 edgecolor='black', fancybox=True, prop={'size': 16, 'weight': 'bold'})

    # Set y-axis label only on the leftmost subplot
    axes[0].set_ylabel('Detection Rate (%)', fontsize=16, fontweight='bold')

    # Set y-axis ticks: major at 20% (labeled), minor at 10% (tick marks only)
    axes[0].set_yticks(np.arange(0, 101, 20))  # Major ticks with labels every 20%
    axes[0].set_yticks(np.arange(0, 101, 10), minor=True)  # Minor tick marks every 10%

    # Adjust spacing between subplots
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_all_classes_multimodel(model_results: Dict[str, pd.DataFrame],
                                save_dir: Optional[str] = None,
                                figsize: tuple = (10, 6),
                                show_plot: bool = True,
                                class_names: Optional[Dict[int, str]] = None,
                                file_prefix: str = 'iou_comparison') -> None:
    """
    Generate comparison plots for all classes and overall performance.

    Creates three plots:
    1. Overall performance comparison
    2. Class 0 comparison
    3. Class 1 comparison
    (Extends automatically if more classes exist)

    Args:
        model_results: Dictionary mapping model names to DataFrames from run_iou_analysis()
        save_dir: Directory to save plots (optional, will use current dir if None but save_path needed)
        figsize: Figure size tuple (default: (10, 6))
        show_plot: Whether to display plots interactively (default: True)
        class_names: Optional dict mapping class IDs to names
        file_prefix: Prefix for saved plot filenames (default: 'iou_comparison')

    Example:
        models = {
            'YOLOv8n': df_model1,
            'YOLOv8s': df_model2,
            'YOLOv8m': df_model3
        }

        plot_all_classes_multimodel(
            models,
            save_dir='./plots',
            class_names={0: 'is_vessel', 1: 'is_fishing'},
            file_prefix='yolo_comparison'
        )

        # Creates:
        # ./plots/yolo_comparison_overall.png
        # ./plots/yolo_comparison_class_0_is_vessel.png
        # ./plots/yolo_comparison_class_1_is_fishing.png
    """
    from pathlib import Path

    if not model_results:
        print("Error: No model results provided.")
        return

    # Determine available classes from first model
    first_df = next(iter(model_results.values()))
    class_rate_cols = [col for col in first_df.columns if col.endswith('_rate') and col.startswith('class_')]
    class_ids = sorted([int(col.split('_')[1]) for col in class_rate_cols])

    # Prepare save directory
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    # Plot overall performance (all classes)
    overall_save = f"{save_dir}/{file_prefix}_all.png" if save_dir else None
    print(f"\nGenerating all classes comparison...")
    plot_multimodel_comparison(model_results, class_id=None, save_path=overall_save,
                               figsize=figsize, show_plot=show_plot, class_names=class_names)

    # Plot each class
    default_names = {0: 'is_vessel', 1: 'is_fishing'}
    for class_id in class_ids:
        # Determine class name with defaults
        if class_names:
            class_name = class_names.get(class_id, default_names.get(class_id, f'class_{class_id}'))
        else:
            class_name = default_names.get(class_id, f'class_{class_id}')

        class_save = f"{save_dir}/{file_prefix}_{class_name}.png" if save_dir else None

        print(f"Generating comparison for {class_name}...")
        plot_multimodel_comparison(model_results, class_id=class_id, save_path=class_save,
                                   figsize=figsize, show_plot=show_plot, class_names=class_names)

    print(f"\nGenerated {len(class_ids) + 1} comparison plots.")


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
    df_results = run_iou_analysis(
        args.gt_dir,
        args.pred_dir,
        args.iou_start,
        args.iou_end,
        args.iou_step,
        output_csv
    )

    print(f"\nAnalysis complete! Results for {len(df_results)} IoU thresholds.")

    # Show summary
    if not df_results.empty:
        best_idx = df_results['overall_rate'].idxmax()
        best_iou = df_results.loc[best_idx, 'iou_threshold']
        best_rate = df_results.loc[best_idx, 'overall_rate']
        print(f"Best overall rate: {best_rate:.1f}% at IoU {best_iou:.1f}")


if __name__ == "__main__":
    main()