#!/usr/bin/env python3

"""
verify_crops_complex.py

Author: Peter Millitz
Created: 2025-06-20

# ----------------------------------------------------------------------
# Visualizes 2D complex64 SAR image crops stored as .npy files with optional
# YOLO-format bounding box overlays. 
#
# Navigation:
#   → : next crop
#   ← : previous crop
#   ↑ : next display mode
#   ↓ : previous display mode
#   Esc : quit
#
# The viewer uses a persistent figure window to maintain screen position.
# ----------------------------------------------------------------------
"""

import numpy as np
import matplotlib

# Try to use interactive backend
try:
    matplotlib.use('Qt5Agg')  # or 'TkAgg' 
except:
    try:
        matplotlib.use('TkAgg')
    except:
        print("Warning: No interactive backend available. GUI may not work properly.")


# Set rcParams to hide toolbar by default  
matplotlib.rcParams['toolbar'] = 'None'

import matplotlib.pyplot as plt
from pathlib import Path

def visualize_crop(crop_path, label_path=None, title=None, display_mode='magnitude', fig=None, ax=None):
    """
    Display a 2D complex64 SAR crop and optional YOLO bounding boxes.

    Args:
        crop_path (Path): Path to .npy file with shape (H, W) and dtype=complex64
        label_path (Path or None): Corresponding YOLO .txt label file
        title (str): Title to show above the image
        display_mode (str): 'magnitude', 'phase', 'real', 'imag', or 'log_magnitude'
        fig (matplotlib.figure.Figure): Reusable figure object
        ax (matplotlib.axes.Axes): Reusable axes object
    """
    crop = np.load(crop_path)
    assert crop.dtype == np.complex64, f"Expected complex64 input, got {crop.dtype}"
    assert len(crop.shape) == 2, f"Expected 2D input (H, W), got shape {crop.shape}"

    # Select visual representation based on display mode
    if display_mode == 'magnitude':
        image = np.abs(crop)
        cmap = 'gray'
        mode_title = "Magnitude"
    elif display_mode == 'log_magnitude':
        magnitude = np.abs(crop)
        # Add small epsilon to avoid log(0)
        image = 20 * np.log10(magnitude + 1e-10)  # Convert to dB
        cmap = 'gray'
        mode_title = "Log Magnitude (dB)"
    elif display_mode == 'phase':
        image = np.angle(crop)
        cmap = 'hsv'  # Phase is circular, HSV colormap works well
        mode_title = "Phase"
    elif display_mode == 'sin_phase':
        phase = np.angle(crop)
        image = (np.sin(phase) + 1) / 2  # Normalize from [-1,1] to [0,1]
        cmap = 'gray'
        mode_title = "sin(phase) [0,1]"
    elif display_mode == 'cos_phase':
        phase = np.angle(crop)
        image = (np.cos(phase) + 1) / 2  # Normalize from [-1,1] to [0,1]
        cmap = 'gray'
        mode_title = "cos(phase) [0,1]"
    elif display_mode == 'real':
        image = np.real(crop)
        cmap = 'gray'
        mode_title = "Real Part"
    elif display_mode == 'imag':
        image = np.imag(crop)
        cmap = 'gray'
        mode_title = "Imaginary Part"
    else:
        raise ValueError(f"Unknown display_mode: {display_mode}")

    # Normalize image for display (except for phase, sin_phase, cos_phase which are handled above)
    if display_mode not in ['phase', 'sin_phase', 'cos_phase']:
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        else:
            image = np.zeros_like(image)  # Handle constant images

    # Use supplied figure or create new one
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    else:
        # Clear the entire figure to remove any existing colorbars
        fig.clear()
        ax = fig.add_subplot(111)

    # Show the processed image
    im = ax.imshow(image, origin="upper", cmap=cmap)
    
    # Add appropriate colorbar for each display mode
    if display_mode == 'phase':
        cbar = plt.colorbar(im, ax=ax, label='Phase (radians)', shrink=0.8)
    elif display_mode == 'sin_phase':
        cbar = plt.colorbar(im, ax=ax, label='sin(phase) [0,1]', shrink=0.8)
    elif display_mode == 'cos_phase':
        cbar = plt.colorbar(im, ax=ax, label='cos(phase) [0,1]', shrink=0.8)
    elif display_mode == 'log_magnitude':
        cbar = plt.colorbar(im, ax=ax, label='Normalized Log Magnitude (dB)', shrink=0.8)
    elif display_mode == 'magnitude':
        cbar = plt.colorbar(im, ax=ax, label='Normalized Magnitude [0,1]', shrink=0.8)
    elif display_mode == 'real':
        cbar = plt.colorbar(im, ax=ax, label='Normalized Real Component [0,1]', shrink=0.8)
    elif display_mode == 'imag':
        cbar = plt.colorbar(im, ax=ax, label='Normalized Imaginary Component [0,1]', shrink=0.8)

    # Draw YOLO-style bounding boxes if labels exist
    if label_path and label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, xc, yc, w, h = map(float, parts[:5])
                    crop_h, crop_w = crop.shape
                    
                    # Convert from normalized YOLO format to pixel coordinates
                    box_w = w * crop_w
                    box_h = h * crop_h
                    box_x = (xc * crop_w) - box_w / 2
                    box_y = (yc * crop_h) - box_h / 2

                    # Draw rectangle
                    rect = plt.Rectangle((box_x, box_y), box_w, box_h,
                                         edgecolor='lime', facecolor='none', linewidth=2)
                    ax.add_patch(rect)

                    # Label with class name positioned off NW corner of BBox
                    label_str = "is_fishing" if int(class_id) == 1 else "is_vessel"
                    label_x = box_x - 2   # 2 pixels left
                    label_y = box_y + box_h
                    ax.text(
                        label_x, label_y, label_str,
                        color='lime',
                        fontsize=10,
                        verticalalignment='bottom',  # Top of label aligns with (x, y)
                        horizontalalignment='right'  # Right end of label aligns with (x, y)
                    )

    # Set title with mode information
    full_title = f"{title or crop_path.stem} - {mode_title}"
    ax.set_title(full_title)
    ax.axis("off")                # Hide axes ticks and labels
    fig.canvas.draw()             # Trigger re-render
    fig.canvas.flush_events()     # Safe GUI update across backends

def visualize_all(image_dir, label_dir, display_mode='magnitude'):
    """
    Visualize all crop-label pairs with keyboard navigation.

    Args:
        image_dir (Path): Directory containing .npy image crops
        label_dir (Path): Directory containing YOLO .txt label files
        display_mode (str): 'magnitude', 'phase', 'real', 'imag', or 'log_magnitude'
    """
    image_files = sorted(list(Path(image_dir).glob("*.npy")))
    if len(image_files) == 0:
        print("No .npy files found in:", image_dir)
        return

    # Use all available image files
    all_files = image_files

    # Create reusable figure window with constrained layout
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    # Set custom window title
    fig.canvas.manager.set_window_title('SAR Image Browser')

    index = [0]         # Mutable image index (wrapped in list for closure)
    mode_index = [0]    # Current display mode index
    done = [False]      # Control flag to end event loop
    
    # Available display modes
    modes = ['magnitude', 'log_magnitude', 'phase', 'sin_phase', 'cos_phase', 'real', 'imag']

    def on_key(event):
        """
        Handle keyboard input for image navigation and mode switching.
        """
        if event.key == 'right':
            if index[0] < len(all_files) - 1:
                index[0] += 1
                update()
        elif event.key == 'left':
            if index[0] > 0:
                index[0] -= 1
                update()
        elif event.key == 'up':
            # Cycle through display modes
            mode_index[0] = (mode_index[0] + 1) % len(modes)
            update()
        elif event.key == 'down':
            # Cycle through display modes (reverse)
            mode_index[0] = (mode_index[0] - 1) % len(modes)
            update()
        elif event.key == 'escape':
            done[0] = True
            plt.close(fig)
            return  # Exit immediately

    def update():
        """
        Display current crop and label based on index and mode.
        """
        try:
            crop_path = all_files[index[0]]
            label_path = Path(label_dir) / (crop_path.stem + ".txt")
            current_mode = modes[mode_index[0]]
            visualize_crop(crop_path, label_path, 
                          title=f"{crop_path.stem} ({index[0]+1}/{len(all_files)})",
                          display_mode=current_mode, fig=fig, ax=ax)
        except Exception as e:
            print(f"Error updating display: {e}")

    # Hook up keyboard handler
    cid = fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial display
    update()
    print("Navigation:")
    print("  → : next crop")
    print("  ← : previous crop") 
    print("  ↑ : next display mode")
    print("  ↓ : previous display mode")
    print("  Esc : quit")
    print(f"\nDisplay modes: {', '.join(modes)}")
    print(f"Total files: {len(all_files)}")

    # Show the plot
    plt.show(block=False)
    
    # Keep GUI responsive and running until user exits
    try:
        while not done[0] and plt.fignum_exists(fig.number):
            plt.pause(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up
        try:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
        except:
            pass

# Example usage if run as script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize 2D complex64 SAR crops with optional YOLO labels')
    parser.add_argument('--images', '-i', required=True, 
                       help='Directory containing .npy image files (complex64, 2D)')
    parser.add_argument('--labels', '-l', required=True, 
                       help='Directory containing .txt label files (YOLO format)')
    parser.add_argument('--mode', '-m', default='log_magnitude', 
                       choices=['magnitude', 'log_magnitude', 'phase', 'sin_phase', 'cos_phase', 'real', 'imag'],
                       help='Initial display mode (default: log_magnitude)')
    
    args = parser.parse_args()
    
    visualize_all(
        image_dir=args.images,
        label_dir=args.labels,
        display_mode=args.mode
    )
