import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Union

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


# Example usage:
if __name__ == "__main__":
    # Basic usage
    plot_total_loss('results.csv')
    
    # Save without showing
    plot_total_loss('results.csv', save_path='loss_curves.png', show_plot=False)
    
    # Custom figure size
    plot_total_loss('results.csv', figsize=(15, 10))
