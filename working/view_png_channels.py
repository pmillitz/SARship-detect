#!/usr/bin/env python3

"""
view_png_channels.py

Author: Peter Millitz
Created: 2025-09-18

A visualization tool for displaying individual channels of PNG-processed SAR image crops.
Each column represents a different image, and each row represents a channel (red, green, blue).
Shows bounding boxes and class labels on all channels.

This tool displays up to 5 PNG images in a 3x5 grid:
- Row 1: Red channel (VH-polarisation magnitude) for all selected images
- Row 2: Green channel (VV-polarisation magnitude) for all selected images
- Row 3: Blue channel (Polarisation coherence magnitude) for all selected images

Images can be selected randomly or specified via a text file list.

Expected directory structure:
- images/: contains .png files processed from complex SAR data
- labels/: contains YOLO-format .txt files with corresponding stems

PNG Format (Dual-Polarisation):
- Red channel: VH-polarisation magnitude (0-255) → divide by 255 to get [0,1] range
- Green channel: VV-polarisation magnitude (0-255) → divide by 255 to get [0,1] range
- Blue channel: Polarisation coherence magnitude (0-255) → divide by 255 to get [0,1] range
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from pathlib import Path
import cv2
import random
import re


def load_png_channels(img_path):
    """Load PNG image and extract all three channels

    Args:
        img_path: Path to .png file

    Returns:
        tuple: (vh_magnitude, vv_magnitude, coherence_magnitude) as float32 arrays normalized to [0,1]
    """
    img_path = Path(img_path)

    if img_path.suffix.lower() != '.png':
        raise ValueError(f"Unsupported file format: {img_path.suffix}. Only .png files are supported.")

    # Load PNG image using OpenCV (BGR format)
    bgr_data = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr_data is None:
        raise ValueError(f"Could not load PNG file: {img_path}")

    # Convert BGR to RGB
    rgb_data = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2RGB)

    # Extract all three channels and normalize to [0,1]
    vh_magnitude = rgb_data[:, :, 0].astype(np.float32) / 255.0  # Red channel
    vv_magnitude = rgb_data[:, :, 1].astype(np.float32) / 255.0  # Green channel
    coherence_magnitude = rgb_data[:, :, 2].astype(np.float32) / 255.0  # Blue channel

    return vh_magnitude, vv_magnitude, coherence_magnitude


def find_matching_label(image_filename, label_filenames):
    """Find matching label file for PNG image (_proc suffix handling)"""
    if image_filename.endswith('_proc.png'):
        candidate = image_filename.replace('_proc.png', '_proc.txt')
    elif image_filename.endswith('.png'):
        candidate = image_filename.replace('.png', '.txt')
    else:
        # Fallback for other naming patterns
        candidate = image_filename.rsplit('.', 1)[0] + '.txt'

    return candidate if candidate in label_filenames else None


def load_all_bounding_boxes(label_path, img_height, img_width):
    """Load all bounding boxes from YOLO format label file"""
    boxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 5:
                    class_id, cx, cy, w, h = map(float, parts)
                elif len(parts) == 4:
                    cx, cy, w, h = map(float, parts)
                    class_id = 0
                else:
                    continue
                x = cx * img_width - (w * img_width) / 2
                y = cy * img_height - (h * img_height) / 2
                boxes.append((x, y, w * img_width, h * img_height, int(class_id)))
    except:
        pass
    return boxes


def plot_channel_with_boxes(ax, channel_data, label_path, channel_name, channel_cmap='gray'):
    """Plot single channel with bounding boxes"""
    img_height, img_width = channel_data.shape

    # Display channel data
    ax.imshow(channel_data, cmap=channel_cmap, vmin=0.0, vmax=1.0, aspect='equal')

    # Add bounding boxes
    bboxes = load_all_bounding_boxes(label_path, img_height, img_width)
    for x, y, w, h, class_id in bboxes:
        edge_col = 'lime' if class_id == 1 else 'tomato'
        rect = Rectangle((x, y), w, h, edgecolor=edge_col, facecolor='none', linewidth=2)
        ax.add_patch(rect)
        label_str = "is_fishing" if class_id == 1 else "is_vessel"
        color = 'lime' if class_id == 1 else 'tomato'
        ax.text(x - 2, y, label_str, color=color, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1))

    ax.set_xticks([])
    ax.set_yticks([])
    return True


def get_valid_png_images(all_imgs, all_lbls, lbl_dir):
    """Get valid PNG images that have corresponding label files"""
    valid_images = []
    for img in all_imgs:
        if img.endswith('.png'):
            label_match = find_matching_label(img, all_lbls)
            if label_match:
                valid_images.append((img, label_match))

    return valid_images


def load_image_list_from_file(file_path, valid_images):
    """Load specific image filenames from a text file"""
    try:
        with open(file_path, 'r') as f:
            requested_images = [line.strip() for line in f if line.strip()]

        # Filter valid_images to only include those in the requested list
        valid_dict = {img: lbl for img, lbl in valid_images}
        selected_images = []

        for img_name in requested_images:
            # Handle both with and without _proc.png suffix
            if not img_name.endswith('.png'):
                img_name += '_proc.png'
            elif img_name.endswith('.png') and not img_name.endswith('_proc.png'):
                img_name = img_name.replace('.png', '_proc.png')

            if img_name in valid_dict:
                selected_images.append((img_name, valid_dict[img_name]))
            else:
                print(f"Warning: Image {img_name} not found in valid images")

        return selected_images

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def view_png_channels(base_dir='.', max_images=5, image_list_file=None, save_path=None,
                     img_subdir='images', lbl_subdir='labels'):
    """
    Visualize PNG channels in a grid layout.

    Parameters:
    -----------
    base_dir : str, default '.'
        Base directory containing image and label subdirectories
    max_images : int, default 5
        Maximum number of images to display (max 5)
    image_list_file : str, optional
        Path to text file containing specific image filenames to display
    save_path : str, optional
        Path to save figure instead of displaying
    img_subdir : str, default 'images'
        Subdirectory name containing .png image files
    lbl_subdir : str, default 'labels'
        Subdirectory name containing .txt label files
    """
    img_dir = Path(base_dir) / img_subdir
    lbl_dir = Path(base_dir) / lbl_subdir

    # Collect only .png image files
    all_imgs = sorted([p.name for p in img_dir.glob('*.png')])
    all_lbls = sorted([p.name for p in lbl_dir.glob('*.txt')])

    if not all_imgs:
        print(f"No .png files found in {img_dir}")
        return

    # Get valid image-label pairs
    valid_images = get_valid_png_images(all_imgs, all_lbls, lbl_dir)
    if not valid_images:
        print("No valid PNG image-label pairs found.")
        return

    # Select images based on input method
    if image_list_file:
        selected_images = load_image_list_from_file(image_list_file, valid_images)
        if not selected_images:
            print("No valid images from list file. Falling back to random selection.")
            selected_images = random.sample(valid_images, min(max_images, len(valid_images)))
    else:
        selected_images = random.sample(valid_images, min(max_images, len(valid_images)))

    # Limit to maximum 5 images
    selected_images = selected_images[:5]
    num_images = len(selected_images)

    # Set up 3xN grid (3 channels x N images)
    num_cols = num_images
    num_rows = 3

    fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows), dpi=100)
    gs = GridSpec(num_rows, num_cols + 1, figure=fig,
                  width_ratios=[1]*num_cols + [0.05],
                  wspace=0.1, hspace=0.3)

    fig.suptitle("PNG Channel Visualisation - Dual Polarisation SAR", fontsize=18, y=0.95)

    # Channel names and colormaps
    channel_info = [
        ("VH Magnitude", 'gray'),
        ("VV Magnitude", 'gray'),
        ("Polarisation Coherence", 'gray')
    ]

    # Process each image
    for col, (img_file, lbl_file) in enumerate(selected_images):
        img_path = img_dir / img_file
        label_path = lbl_dir / lbl_file

        try:
            vh_magnitude, vv_magnitude, coherence_magnitude = load_png_channels(img_path)
            channels = [vh_magnitude, vv_magnitude, coherence_magnitude]

            # Plot each channel in its respective row
            for row, (channel_data, (channel_name, cmap)) in enumerate(zip(channels, channel_info)):
                ax = fig.add_subplot(gs[row, col])

                plot_channel_with_boxes(ax, channel_data, label_path, channel_name, cmap)

                # Add column title only for the first row
                if row == 0:
                    title_base = img_file.replace('_proc.png', '')
                    first_underscore = title_base.find('_')
                    if first_underscore != -1:
                        title_base = title_base[:first_underscore + 1] + '\n' + title_base[first_underscore + 1:]
                    title_wrapped = title_base.replace('swath', '\nswath')
                    ax.set_title(title_wrapped, fontsize=7, loc='center')

                # Add row label only for the first column
                if col == 0:
                    ax.set_ylabel(channel_name, fontsize=10, rotation=90, va='center')

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    # Add colorbars for each row - match the height of the images
    # Calculate the position and height based on the subplot positions
    if num_images > 0:
        # Get the first column's subplot positions to calculate row heights
        first_col_axes = [fig.add_subplot(gs[row, 0]) for row in range(num_rows)]

        for row, (channel_name, cmap) in enumerate(channel_info):
            # Get the position of the subplot in this row
            ax_pos = first_col_axes[row].get_position()
            row_bottom = ax_pos.y0
            row_height = ax_pos.height

            # Position colorbar to match the image height exactly
            cbar_ax = fig.add_axes([0.92, row_bottom, 0.02, row_height])
            norm = Normalize(vmin=0.0, vmax=1.0)
            cb = ColorbarBase(cbar_ax, cmap=cm.gray, norm=norm, orientation='vertical')
            cb.set_label("Normalised Magnitude", fontsize=8)
            cb.ax.tick_params(labelsize=8)

        # Remove the temporary axes we created for positioning
        for ax in first_col_axes:
            ax.remove()

    # Add figure caption
    caption = f"Figure: Channel-wise visualisation of {num_images} dual-polarisation PNG SAR images. " \
              f"Top row: VH magnitude. Second row: VV magnitude. Third row: Polarisation coherence features."
    fig.text(0.5, 0.02, caption, ha='center', fontsize=10, wrap=True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualise individual channels of dual-polarisation PNG-processed SAR images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View 5 random PNG images showing all channels
  %(prog)s data/train_png_processed

  # View specific images from a list file
  %(prog)s data/train_png_processed --image-list-file selected_images.txt

  # Save visualization
  %(prog)s data/train_png_processed --max-images 3 --save-path channels_output.png

  # Use custom subdirectory names
  %(prog)s data/processed --img-subdir images_dual --lbl-subdir labels_dual

Image list file format (one filename per line):
  filename1_swath1_original
  filename2_swath2_aug0_hflip
  filename3_swath1_original

Channel Information:
  Red Channel:   VH-polarisation magnitude
  Green Channel: VV-polarisation magnitude
  Blue Channel:  Polarisation coherence magnitude
        """
    )

    parser.add_argument('base_dir', default='.', nargs='?',
                       help='Base directory containing PNG image and label subdirectories (default: current directory)')
    parser.add_argument('--max-images', type=int, default=5,
                       help='Maximum number of images to display (max 5, default: 5)')
    parser.add_argument('--image-list-file',
                       help='Path to text file containing specific image filenames to display')
    parser.add_argument('--save-path',
                       help='Path to save figure instead of displaying')
    parser.add_argument('--img-subdir', default='images',
                       help='Subdirectory name containing .png image files (default: images)')
    parser.add_argument('--lbl-subdir', default='labels',
                       help='Subdirectory name containing .txt label files (default: labels)')

    args = parser.parse_args()

    # Enforce maximum of 5 images
    if args.max_images > 5:
        print("Warning: max_images capped at 5 for optimal display")
        args.max_images = 5

    # Call the main function
    view_png_channels(
        base_dir=args.base_dir,
        max_images=args.max_images,
        image_list_file=args.image_list_file,
        save_path=args.save_path,
        img_subdir=args.img_subdir,
        lbl_subdir=args.lbl_subdir
    )