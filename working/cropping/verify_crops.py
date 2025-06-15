# verify_crops.py
# ----------------------------------------------------------------------
# Visualizes 4-channel SAR image crops stored as .npy files with optional
# YOLO-format bounding box overlays. 
#
# Navigation:
#  ->  : move to next crop
#  <-  : move to previous crop
#  Esc : exit the viewer
#
# The viewer uses a persistent figure window to maintain screen position.
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def visualize_crop(crop_path, label_path=None, title=None, grayscale=None, fig=None, ax=None):
    """
    Display a 4-channel SAR crop and optional YOLO bounding boxes.

    Args:
        crop_path (Path): Path to .npy file with shape (4, H, W)
        label_path (Path or None): Corresponding YOLO .txt label file
        title (str): Title to show above the image
        grayscale (str or None): 'vv', 'vh', or None for RGB-style
        fig (matplotlib.figure.Figure): Reusable figure object
        ax (matplotlib.axes.Axes): Reusable axes object
    """
    crop = np.load(crop_path)
    assert crop.shape[0] == 4, "Expected 4-channel input (VH_mag, VH_phase, VV_mag, VV_phase)"

    # Select visual representation
    if grayscale == 'vv':
        image = crop[2]  # VV magnitude
        image = (image - image.min()) / (image.max() - image.min())
        cmap = 'gray'
    elif grayscale == 'vh':
        image = crop[0]  # VH magnitude
        image = (image - image.min()) / (image.max() - image.min())
        cmap = 'gray'
    else:
        # RGB-style: [VH_mag, VV_mag, avg]
        vh_mag = crop[0]
        vv_mag = crop[2]
        combined = np.stack([vh_mag, vv_mag, (vh_mag + vv_mag) / 2], axis=-1)
        image = (combined - combined.min()) / (combined.max() - combined.min())
        cmap = None

    # Use supplied figure or create new one
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        ax.clear()  # Reuse panel by clearing previous content

    # Show the normalized image
    ax.imshow(image, origin="upper", cmap=cmap)

    # Draw YOLO-style bounding boxes if labels exist
    if label_path and label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                class_id, xc, yc, w, h = map(float, line.strip().split())
                crop_size = crop.shape[1]  # Assume square (H == W)
                box_w = w * crop_size
                box_h = h * crop_size
                box_x = (xc * crop_size) - box_w / 2
                box_y = (yc * crop_size) - box_h / 2

                # Draw rectangle
                rect = plt.Rectangle((box_x, box_y), box_w, box_h,
                                     edgecolor='lime', facecolor='none', linewidth=2)
                ax.add_patch(rect)

                # Label with class name postioned off NW corner of BBox
                label_str = "is_fishing" if int(class_id) == 1 else "is_vessel"
                label_x = box_x - 2   # 2 pixels left
                label_y = box_y + box_h
                ax.text(
                    label_x, label_y, label_str,
                    color='lime',
                    fontsize=10,
                    verticalalignment='bottom', # Top of label aligns with (x, y)
                    horizontalalignment='right' # Right end of label aligns with (x, y)
                )

    ax.set_title(title or crop_path.stem)
    ax.axis("off")                # Hide axes ticks and labels
    fig.canvas.draw()             # Trigger re-render
    fig.canvas.flush_events()     # Safe GUI update across backends

def sample_and_visualize(image_dir, label_dir, n=5, grayscale=None):
    """
    Sample N crop-label pairs and visualize them with keyboard navigation.

    Args:
        image_dir (Path): Directory containing .npy image crops
        label_dir (Path): Directory containing YOLO .txt label files
        n (int): Number of examples to show
        grayscale (str or None): 'vv', 'vh', or None for RGB-style
    """
    image_files = sorted(list(Path(image_dir).glob("*.npy")))
    if len(image_files) == 0:
        print("No .npy files found in:", image_dir)
        return

    # Random sample of available image files
    sample_files = random.sample(image_files, min(n, len(image_files)))

    # Create reusable figure window
    fig, ax = plt.subplots(figsize=(5, 5))

    index = [0]         # Mutable image index (wrapped in list for closure)
    done = [False]      # Control flag to end event loop

    def on_key(event):
        """
        Handle keyboard input for image navigation.
        """
        if event.key == 'right':
            if index[0] < len(sample_files) - 1:
                index[0] += 1
                update()
        elif event.key == 'left':
            if index[0] > 0:
                index[0] -= 1
                update()
        elif event.key == 'escape':
            done[0] = True
            plt.close(fig)

    def update():
        """
        Display current crop and label based on index.
        """
        crop_path = sample_files[index[0]]
        label_path = Path(label_dir) / (crop_path.stem + ".txt")
        visualize_crop(crop_path, label_path, title=f"{crop_path.stem} ({index[0]+1}/{len(sample_files)})",
                       grayscale=grayscale, fig=fig, ax=ax)

    # Hook up keyboard handler
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial display
    update()
    print("Navigate with → (next), ← (back), Esc (quit)")

    # Keep GUI responsive and running until user exits
    while not done[0]:
        plt.pause(0.1)

# Example usage if run as script
if __name__ == "__main__":
    sample_and_visualize(image_dir='extracted_slc_crops/images', label_dir='extracted_slc_crops/labels', n=10, grayscale='vh')

