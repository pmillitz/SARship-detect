import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List

def plot_total_loss(csv_path: Union[str, Path], 
                   figsize: Tuple[int, int] = (8, 5),
                   save_path: Optional[Union[str, Path]] = None,
                   show_plot: bool = True,
                   ylim: Optional[Tuple[float, float]] = None) -> None:
    """
    Plot total training vs validation loss from YOLO results.csv file.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the results.csv file
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save the plot. If None, plot is not saved.
    show_plot : bool
        Whether to display the plot
    
    Returns:
    --------
    None
    """
    
    # Convert to Path object for consistent handling
    csv_path = Path(csv_path)
    
    # Load the results
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Calculate total losses
    df['total_train_loss'] = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
    df['total_val_loss'] = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(df['epoch'], df['total_train_loss'], label='Training Loss', color='red', linewidth=2)
    plt.plot(df['epoch'], df['total_val_loss'], label='Validation Loss', color='blue', linewidth=2)
    
    # Extract run info from path for title
    run_info = ""
    if len(csv_path.parts) >= 2:
        # Get the parent directory name (e.g., "20250808_1712")
        run_number = csv_path.parent.name
        run_info = f" - Run {run_number}"
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.title(f'Training vs Validation Loss Curves{run_info}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_loss_curves_combined(csv_paths: Dict[str, Union[str, Path]],
                              save_path: Optional[Union[str, Path]] = None,
                              figsize: Optional[Tuple[float, float]] = None,
                              show_plot: bool = True,
                              ylim: Optional[Tuple[float, float]] = None,
                              stack: bool = False) -> None:
    """
    Plot three training loss curves side-by-side or stacked in a single figure.
    Optimized for A4 PDF thesis documents with shared y-axis and readable fonts.

    Parameters:
    -----------
    csv_paths : dict
        Dictionary mapping subplot labels to results.csv file paths.
        Example: {'Model A': 'run1/results.csv', 'Model B': 'run2/results.csv', 'Model C': 'run3/results.csv'}
    save_path : str or Path, optional
        Path to save the combined plot (e.g., 'loss_curves_combined.pdf')
    figsize : tuple, optional
        Figure size (width, height). If None, defaults to (18, 4.2) for side-by-side or (6, 12) for stacked
    show_plot : bool
        Whether to display the plot interactively
    ylim : tuple, optional
        Y-axis limits (min, max). If None, automatically determined from data.
    stack : bool
        If True, stack plots vertically; if False, arrange side-by-side (default: False)

    Returns:
    --------
    None

    Example:
    --------
    csv_paths = {
        'Model A': 'runs/run1/results.csv',
        'Model B': 'runs/run2/results.csv',
        'Model C': 'runs/run3/results.csv'
    }

    # Side-by-side layout
    plot_loss_curves_combined(
        csv_paths,
        save_path='./thesis_figures/loss_curves_combined.pdf',
        show_plot=False
    )

    # Stacked layout
    plot_loss_curves_combined(
        csv_paths,
        save_path='./thesis_figures/loss_curves_stacked.pdf',
        show_plot=False,
        stack=True
    )
    """

    if len(csv_paths) != 3:
        raise ValueError(f"Expected exactly 3 csv paths, got {len(csv_paths)}")

    # Set default figsize based on layout
    if figsize is None:
        figsize = (6, 12) if stack else (18, 4.2)

    # Create figure with 3 subplots
    if stack:
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    # Load and plot each training run
    for ax_idx, (label, csv_path) in enumerate(csv_paths.items()):
        ax = axes[ax_idx]
        csv_path = Path(csv_path)

        # Load the results
        if not csv_path.exists():
            print(f"Warning: Results file not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Calculate total losses
        df['total_train_loss'] = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
        df['total_val_loss'] = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']

        # Plot training and validation loss
        ax.plot(df['epoch'], df['total_train_loss'],
               label='Training', color='#d62728', linewidth=2, marker='o',
               markersize=4, markevery=max(1, len(df)//10))
        ax.plot(df['epoch'], df['total_val_loss'],
               label='Validation', color='#1f77b4', linewidth=2, marker='s',
               markersize=4, markevery=max(1, len(df)//10))

        # Configure subplot
        # Only add x-label to bottom plot when stacked, or all plots when side-by-side
        if stack:
            if ax_idx == len(csv_paths) - 1:  # Bottom plot
                ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        else:
            ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=14)

        # Add minor gridlines
        ax.grid(True, which='minor', alpha=0.1, linestyle=':')
        ax.minorticks_on()

        # Add model label box in top right corner
        textstr = label
        props = dict(boxstyle='round,pad=0.5,rounding_size=0.15', facecolor='white',
                     edgecolor='black', linewidth=1.5, alpha=0.95)
        ax.text(0.95, 0.94, textstr, transform=ax.transAxes,
                fontsize=15, verticalalignment='top', horizontalalignment='right',
                bbox=props)

        # Only add legend to the first subplot (upper left of center, avoiding model label)
        if ax_idx == 0:
            # Position: top left of center with right edge at vertical center
            ax.legend(loc='upper center', bbox_to_anchor=(0.35, 1.0),
                     fontsize=14, framealpha=0.95,
                     edgecolor='black', fancybox=True)

    # Set y-axis label
    if stack:
        # For stacked layout, add y-label to all subplots
        for ax in axes:
            ax.set_ylabel('Total Loss', fontsize=16, fontweight='bold')
    else:
        # For side-by-side layout, add y-label to leftmost subplot
        axes[0].set_ylabel('Total Loss', fontsize=16, fontweight='bold')

    # Set y-axis limits if provided
    if ylim is not None:
        if stack:
            for ax in axes:
                ax.set_ylim(ylim)
        else:
            axes[0].set_ylim(ylim)

    # Adjust spacing between subplots
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


# Example usage:
if __name__ == "__main__":
    # Single plot - basic usage
    plot_total_loss('results.csv')

    # Single plot - save without showing
    plot_total_loss('results.csv', save_path='loss_curves.png', show_plot=False)

    # Single plot - custom figure size
    plot_total_loss('results.csv', figsize=(15, 10))

    # Combined plot - three training runs side-by-side for publication
    csv_paths = {
        'YOLOv8n': 'runs/yolov8n/20250915_1358_30042/results.csv',
        'YOLOv8s': 'runs/yolov8s/20250916_1045_12345/results.csv',
        'YOLOv8m': 'runs/yolov8m/20250917_0930_67890/results.csv'
    }

    plot_loss_curves_combined(
        csv_paths,
        save_path='./thesis_figures/loss_curves_combined.pdf',
        show_plot=False,
        ylim=(0, 2.0)  # Optional: set consistent y-axis range
    )
