#!/usr/bin/env python3

"""
verify_crops.py

Author: Peter Millitz
Created: 2025-06-23

Visualises 2D complex64 SAR image crops stored as .npy files with optional
YOLO-format bounding box overlays. 

Options:
  -h, --help: show this help message and exit
  -i, --images IMAGES, -i IMAGES: directory containing .npy image files (complex64, 2D)
  -l, --labels LABELS, -l LABELS: directory containing .txt label files (YOLO format)
  -m, --mode {magnitude,log_magnitude,phase,sin_phase,cos_phase,real,imag}: Initial
             display mode (default: magnitude)
  -n, --samples SAMPLES: number of randomly sampled images to view (default = 50)
  -s, --single IMAGE: view single image by filename (e.g., "image_001" or "image_001.npy")

Navigation:
 ->  : move to next crop
 <-  : move to previous crop
 ↑   : next display mode
 ↓   : previous display mode
 Esc : exit the viewer

The viewer uses a persistent figure window to maintain screen position.
"""

import numpy as np
import matplotlib
import random

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

def select_sample_images(image_files, n):
    """
    Select a sample of images based on specified number.
    
    Args:
        image_files (list): List of all image file paths
        n (int): Number of images to sample
        
    Returns:
        list: Selected sample of image files
    """
    total_count = len(image_files)
    
    if n >= total_count:
        # Use all images if n is greater than or equal to total
        selected_files = image_files
        sample_info = f"Using all {total_count} images"
    else:
        # Sample n images
        selected_files = random.sample(image_files, n)
        sample_info = f"Randomly sampled {n} images from {total_count} total"
    
    print(f"Sample selection: {sample_info}")
    return sorted(selected_files)

def visualise_crop(crop_path, label_path=None, title=None, display_mode='magnitude', fig=None, ax=None):
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
        image = (np.sin(phase) + 1) / 2  # Normalise from [-1,1] to [0,1]
        cmap = 'gray'
        mode_title = "sin(Phase) [0,1]"
    elif display_mode == 'cos_phase':
        phase = np.angle(crop)
        image = (np.cos(phase) + 1) / 2  # Normalise from [-1,1] to [0,1]
        cmap = 'gray'
        mode_title = "cos(Phase) [0,1]"
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

    # Normalise image for display (except for phase, sin_phase, cos_phase which are handled above)
    if display_mode not in ['phase', 'sin_phase', 'cos_phase']:
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        else:
            image = np.zeros_like(image)  # Handle constant images

    # Use supplied figure or create new one
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
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
        cbar = plt.colorbar(im, ax=ax, label='sin(Phase) [0,1]', shrink=0.8)
    elif display_mode == 'cos_phase':
        cbar = plt.colorbar(im, ax=ax, label='cos(Phase) [0,1]', shrink=0.8)
    elif display_mode == 'log_magnitude':
        cbar = plt.colorbar(im, ax=ax, label='Normalised Log Magnitude (dB)', shrink=0.8)
    elif display_mode == 'magnitude':
        cbar = plt.colorbar(im, ax=ax, label='Normalised Magnitude [0,1]', shrink=0.8)
    elif display_mode == 'real':
        cbar = plt.colorbar(im, ax=ax, label='Normalised Real Component [0,1]', shrink=0.8)
    elif display_mode == 'imag':
        cbar = plt.colorbar(im, ax=ax, label='Normalised Imaginary Component [0,1]', shrink=0.8)

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
                    label_y = box_y
                    ax.text(
                        label_x, label_y, label_str,
                        color='lime',
                        fontsize=10,
                        verticalalignment='top',     # Top of label aligns with (x, y)
                        horizontalalignment='right'  # Right end of label aligns with (x, y)
                    )

    # Set title with mode information
    full_title = f"{title or crop_path.stem} - {mode_title}"
    ax.set_title(full_title)
    ax.axis("off")                # Hide axes ticks and labels
    fig.canvas.draw()             # Trigger re-render
    fig.canvas.flush_events()     # Safe GUI update across backends

def visualise_single(image_dir, label_dir=None, image_name=None, display_mode='magnitude'):
    """
    Visualise a single specific image with display mode switching.

    Args:
        image_dir (Path): Directory containing .npy image crops
        label_dir (Path or None): Directory containing YOLO .txt label files (optional)
        image_name (str): Specific image filename to display
        display_mode (str): Initial display mode
    """
    # Check if image directory exists
    if not Path(image_dir).exists():
        print(f"Error: Image directory does not exist: {image_dir}")
        return
    
    # Check if labels directory exists (warn but continue if not)
    labels_available = True
    if label_dir is None:
        print("No labels directory specified - displaying images without labels")
        labels_available = False
    elif not Path(label_dir).exists():
        print(f"Warning: Labels directory does not exist: {label_dir}")
        print("Continuing to display images without labels")
        labels_available = False

    # Find the specific image file
    image_path = Path(image_dir) / image_name
    if not image_path.exists():
        # Try adding .npy extension if not present
        if not image_name.endswith('.npy'):
            image_path = Path(image_dir) / (image_name + '.npy')
    
    if not image_path.exists():
        print(f"Image not found: {image_name}")
        print(f"Searched in: {image_dir}")
        return

    print(f"Displaying: {image_path.name}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    fig.canvas.manager.set_window_title(f'SAR Viewer - {image_path.name}')
    
    # Center the window on screen
    try:
        # Get screen dimensions and calculate center position
        mngr = fig.canvas.manager
        mngr.window.wm_geometry("+400+200")  # x_offset+y_offset from top-left
    except:
        pass  # Ignore if positioning fails

    mode_index = [0]    # Current display mode index
    done = [False]      # Control flag to end event loop
    
    # Available display modes
    modes = ['magnitude', 'log_magnitude', 'phase', 'sin_phase', 'cos_phase', 'real', 'imag']
    
    # Set initial mode index
    if display_mode in modes:
        mode_index[0] = modes.index(display_mode)

    def on_key(event):
        """
        Handle keyboard input for display mode switching.
        """
        if event.key == 'up':
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
            return

    def update():
        """
        Display the image with current mode.
        """
        try:
            label_path = None
            if labels_available:
                label_path = Path(label_dir) / (image_path.stem + ".txt")
            current_mode = modes[mode_index[0]]
            visualise_crop(image_path, label_path, 
                          title=f"{image_path.name}",
                          display_mode=current_mode, fig=fig, ax=ax)
        except Exception as e:
            print(f"Error updating display: {e}")

    # Hook up keyboard handler
    cid = fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial display
    update()
    print("Single Image Mode:")
    print("  ↑ : next display mode")
    print("  ↓ : previous display mode")
    print("  Esc : quit")
    print(f"\nDisplay modes: {', '.join(modes)}")
    print(f"Current mode: {modes[mode_index[0]]}")

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

def visualise_all(image_dir, label_dir=None, display_mode='magnitude', n_samples=50):
    """
    Visualise crop-label pairs with sampling and keyboard navigation.

    Args:
        image_dir (Path): Directory containing .npy image crops
        label_dir (Path or None): Directory containing YOLO .txt label files (optional)
        display_mode (str): 'magnitude', 'phase', 'real', 'imag', or 'log_magnitude'
        n_samples (int): Number of images to sample for viewing
    """
    # Check if image directory exists
    if not Path(image_dir).exists():
        print(f"Error: Image directory does not exist: {image_dir}")
        return
    
    all_image_files = sorted(list(Path(image_dir).glob("*.npy")))
    if len(all_image_files) == 0:
        print("No .npy files found in:", image_dir)
        return

    # Check if labels directory exists (warn but continue if not)
    labels_available = True
    if label_dir is None:
        print("No labels directory specified - displaying images without labels")
        labels_available = False
    elif not Path(label_dir).exists():
        print(f"Warning: Labels directory does not exist: {label_dir}")
        print("Continuing to display images without labels")
        labels_available = False

    # Apply sampling
    sample_files = select_sample_images(all_image_files, n_samples)

    # Create reusable figure window with constrained layout
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    
    # Set custom window title
    fig.canvas.manager.set_window_title('SAR Crop Viewer')
    
    # Center the window on screen
    try:
        # Get screen dimensions and calculate center position
        mngr = fig.canvas.manager
        mngr.window.wm_geometry("+400+200")  # x_offset+y_offset from top-left
    except:
        pass  # Ignore if positioning fails

    index = [0]         # Mutable image index (wrapped in list for closure)
    mode_index = [0]    # Current display mode index
    done = [False]      # Control flag to end event loop
    
    # Available display modes
    modes = ['magnitude', 'log_magnitude', 'phase', 'sin_phase', 'cos_phase', 'real', 'imag']
    
    # Set initial mode index based on the requested display_mode
    if display_mode in modes:
        mode_index[0] = modes.index(display_mode)
    else:
        print(f"Unknown mode '{display_mode}', defaulting to magnitude")

    def on_key(event):
        """
        Handle keyboard input for image navigation and mode switching.
        """
        if event.key == 'right':
            if index[0] < len(sample_files) - 1:
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
            crop_path = sample_files[index[0]]
            label_path = None
            if labels_available:
                label_path = Path(label_dir) / (crop_path.stem + ".txt")
            current_mode = modes[mode_index[0]]
            visualise_crop(crop_path, label_path, 
                          title=f"{crop_path.stem} ({index[0]+1}/{len(sample_files)})",
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
    print(f"Sample size: {len(sample_files)} images")

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
    parser.add_argument('--labels', '-l', default=None,
                       help='Directory containing .txt label files (YOLO format, optional)')
    parser.add_argument('--mode', '-m', default='magnitude', 
                       choices=['magnitude', 'log_magnitude', 'phase', 'sin_phase', 'cos_phase', 'real', 'imag'],
                       help='Initial display mode (default: magnitude)')
    parser.add_argument('--samples', '-n', type=int, default=50,
                       help='Number of images to randomly sample for viewing (default: 50, use large number to view all)')
    parser.add_argument('--single', '-s', 
                       help='View single image by filename (e.g., "image_001" or "image_001.npy")')
    
    args = parser.parse_args()
    
    if args.single:
        # Single image mode
        visualise_single(
            image_dir=args.images,
            label_dir=args.labels,
            image_name=args.single,
            display_mode=args.mode
        )
    else:
        # Browse mode with sampling
        visualise_all(
            image_dir=args.images,
            label_dir=args.labels,
            display_mode=args.mode,
            n_samples=args.samples
        )

