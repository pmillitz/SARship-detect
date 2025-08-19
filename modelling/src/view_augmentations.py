#!/usr/bin/env python3

# Version 25 - Reverted to original intent: unprocessed augmented images only
# (Note: Formerly named 'view_scaled_magnitude.py')

"""
view_augmentations.py

A visualization tool for inspecting augmented SAR SLC image crops.
Supports only numpy arrays with shape (H, W) and dtype=complex64 for unprocessed augmented data.
The magnitude is computed and displayed in grayscale.

Expected directory structure:
- images/: contains .npy files with shape (H, W) and dtype=complex64
- labels/: contains YOLO-format .txt files with same stem (no '_proc' suffix).

Displays up to 5 randomly selected image-label pairs in two rows (max 5 per row).
Each original-augmented image pair is aligned vertically. Handles multiple bounding
boxes per image and supports clipping using raw dB values.

Supports mosaic image visualization (up to 10 examples) when select_mosaics=True.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from pathlib import Path
import os
import random
import re

_label_cache = {}

def load_image_data(img_path):
    """Load image data from .npy file and return magnitude
    
    Args:
        img_path: Path to .npy image file
        
    Returns:
        numpy.ndarray: Magnitude from complex64 array with shape (H, W)
    """
    img_path = Path(img_path)
    
    if img_path.suffix.lower() != '.npy':
        raise ValueError(f"Unsupported file format: {img_path.suffix}. Only .npy files are supported.")
    
    # Load numpy array with shape (H, W) and dtype=complex64
    img_data = np.load(img_path)
    if img_data.ndim != 2:
        raise ValueError(f"Invalid numpy shape: {img_data.shape}, expected (H, W)")
    if not np.iscomplexobj(img_data):
        raise ValueError(f"Invalid numpy dtype: {img_data.dtype}, expected complex64")
    
    # Return magnitude
    return np.abs(img_data)

def find_matching_label(image_filename, label_filenames):
    """Find matching label file for image (no '_proc' suffix for unprocessed data)"""
    if image_filename.endswith('.npy'):
        candidate = image_filename.replace('.npy', '.txt')
    else:
        # Fallback for other naming patterns
        candidate = image_filename.rsplit('.', 1)[0] + '.txt'
    
    return candidate if candidate in label_filenames else None

def find_matching_mosaic_label(image_filename, label_filenames):
    """Find matching label file for mosaic images"""
    # Extract X from mosaic_minority_X.npy or mosaic_majority_X.npy
    match = re.match(r'mosaic_(minority|majority)_(\d+)\.npy$', image_filename)
    if match:
        mosaic_type = match.group(1)
        x = match.group(2)
        candidate = f'mosaic_{mosaic_type}_{x}.txt'
        return candidate if candidate in label_filenames else None
    return None

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


def preload_label_cache(lbl_dir, all_lbls):
    """Pre-load all label files into cache for fast class filtering"""
    global _label_cache
    
    # Only reload cache if it's empty or has different files
    if len(_label_cache) == len(all_lbls):
        return
        
    _label_cache.clear()
    
    for lbl_file in all_lbls:
        lbl_path = lbl_dir / lbl_file
        file_key = str(lbl_path)
        
        classes = set()
        try:
            with open(lbl_path, 'r') as f:
                content = f.read().strip()
                if content:
                    for line in content.split('\n'):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            class_id = int(parts[0]) if len(parts) == 5 else 0
                            classes.add(class_id)
        except (IOError, ValueError, IndexError):
            pass
        
        _label_cache[file_key] = classes

def cached_label_contains_class(label_file, class_filter):
    """Fast cached version - assumes cache is pre-loaded"""
    if class_filter is None:
        return True
    
    file_key = str(label_file)
    return class_filter in _label_cache.get(file_key, set())

def validate_and_convert_clipping(clip_db_min, clip_db_max, data_db_min, data_db_max):
    """Validate dB clipping parameters and convert to normalized values"""
    clip_min = clip_max = None
    
    if clip_db_min is not None:
        if clip_db_min < data_db_min:
            print(f"Warning: clip_db_min={clip_db_min:.2f} is below data range ({data_db_min:.2f} dB). Clamping to {data_db_min:.2f} dB.")
            clip_db_min = data_db_min
        elif clip_db_min > data_db_max:
            print(f"Warning: clip_db_min={clip_db_min:.2f} exceeds data range ({data_db_max:.2f} dB). Clamping to {data_db_max:.2f} dB.")
            clip_db_min = data_db_max
        clip_min = db_to_normalized(clip_db_min, data_db_min, data_db_max)
    
    if clip_db_max is not None:
        if clip_db_max < data_db_min:
            print(f"Warning: clip_db_max={clip_db_max:.2f} is below data range ({data_db_min:.2f} dB). Clamping to {data_db_min:.2f} dB.")
            clip_db_max = data_db_min
        elif clip_db_max > data_db_max:
            print(f"Warning: clip_db_max={clip_db_max:.2f} exceeds data range ({data_db_max:.2f} dB). Clamping to {data_db_max:.2f} dB.")
            clip_db_max = data_db_max
        clip_max = db_to_normalized(clip_db_max, data_db_min, data_db_max)
    
    # Ensure clip_min <= clip_max
    if clip_min is not None and clip_max is not None and clip_min > clip_max:
        print(f"Warning: clip_db_min > clip_db_max. Swapping values.")
        clip_min, clip_max = clip_max, clip_min
    
    return clip_min, clip_max

def setup_figure_and_grid(num_cols, num_rows, title):
    """Set up matplotlib figure and grid"""
    fig = plt.figure(figsize=(4 * num_cols + 1.5, 5 * num_rows), dpi=100)
    gs = GridSpec(num_rows + 1, num_cols + 1, figure=fig, 
                  width_ratios=[1]*num_cols + [0.05], 
                  height_ratios=[0.2] + [1]*num_rows, 
                  wspace=0.1, hspace=0.3)
    fig.suptitle(title, fontsize=18, y=0.90)
    return fig, gs

def plot_image_with_boxes(ax, img_path, label_path, amp_clip_min, amp_clip_max):
    """Plot single image with bounding boxes"""
    try:
        mag = load_image_data(img_path)  # Raw amplitude/magnitude values
    except (ValueError, IOError) as e:
        print(f"Skipping invalid image: {img_path} - {e}")
        return False
    
    # Default clipping values (1% and 99% percentiles of train dataset)
    default_amp_min = 1.00
    default_amp_max = 71.51
    
    # Use provided clipping values or defaults
    clip_min = amp_clip_min if amp_clip_min is not None else default_amp_min
    clip_max = amp_clip_max if amp_clip_max is not None else default_amp_max
    
    # Step 1: Clip raw amplitudes
    mag_clipped = np.clip(mag, clip_min, clip_max)
    
    # Step 2: Convert clipped amplitudes to dB
    mag_db = 20 * np.log10(mag_clipped + 1e-10)  # Add epsilon to avoid log(0)
    
    # Step 3: Normalize dB values to [0,1] for display
    db_min = 20 * np.log10(clip_min + 1e-10)
    db_max = 20 * np.log10(clip_max + 1e-10)
    
    if db_max > db_min:
        display_data = (mag_db - db_min) / (db_max - db_min)
    else:
        display_data = np.zeros_like(mag_db)
    
    # Step 4: Display with fixed [0,1] range
    ax.imshow(display_data, cmap='gray', vmin=0.0, vmax=1.0, aspect='equal')
    
    # Add bounding boxes
    bboxes = load_all_bounding_boxes(label_path, *mag.shape)
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

def add_colorbars_to_rows(fig, row_axes, amp_clip_min, amp_clip_max):
    """Add colorbars for each row that has images"""
    # Use default clipping values if not provided
    default_amp_min = 1.00
    default_amp_max = 71.51
    
    clip_min = amp_clip_min if amp_clip_min is not None else default_amp_min
    clip_max = amp_clip_max if amp_clip_max is not None else default_amp_max
    
    # Convert amplitude clipping bounds to dB for colorbar labels
    db_min = 20 * np.log10(clip_min + 1e-10)
    db_max = 20 * np.log10(clip_max + 1e-10)
    
    norm = Normalize(vmin=0.0, vmax=1.0)  # Display data is normalized to [0,1]
    
    for row, axes in enumerate(row_axes):
        if axes:
            row_top = min(ax.get_position().y0 for ax in axes)
            row_bottom = max(ax.get_position().y1 for ax in axes)
            row_right = max(ax.get_position().x1 for ax in axes)
            
            cbar_ax = fig.add_axes([row_right + 0.01, row_top, 0.02, row_bottom - row_top])
            cb = ColorbarBase(cbar_ax, cmap=cm.gray, norm=norm, orientation='vertical')
            cb.set_label("dB Magnitude", fontsize=10)
            
            # Set endpoint ticks in dB showing the actual clipping range
            tick_vals = [0.0, 1.0]
            tick_labels = [f"{db_min:.1f}", f"{db_max:.1f}"]
            cb.set_ticks(tick_vals)
            cb.set_ticklabels(tick_labels)
            cb.ax.tick_params(labelsize=8)

def get_valid_mosaic_pairs(all_imgs, all_lbls, lbl_dir, class_filter):
    """Get valid mosaic image-label pairs"""
    mosaic_imgs = [img for img in all_imgs if re.match(r'mosaic_(minority|majority)_\d+\.npy$', img)]
    
    if not mosaic_imgs:
        return []
    
    valid_mosaics = []
    for img in mosaic_imgs:
        label_match = find_matching_mosaic_label(img, all_lbls)
        if label_match:
            label_path = lbl_dir / label_match
            if class_filter is None or cached_label_contains_class(label_path, class_filter):
                valid_mosaics.append((img, label_match))
    
    return valid_mosaics

def get_valid_augmented_pairs(all_imgs, all_lbls, lbl_dir, class_filter):
    """Get valid original-augmented image pairs"""
    pairs = {}
    for img in all_imgs:
        # Match pattern for unprocessed augmented files (no '_proc' suffix)
        match = re.match(r'(.+_swath\d+)_(original|aug\d+_[^_]+(?:_[^_]+)*)\.npy$', img)
        if match:
            prefix = match.group(1)
            variant_full = match.group(2)
            
            # Extract the actual variant name
            if variant_full == 'original':
                variant = 'original'
            else:
                # Use the full augmentation string as the variant key to ensure unique matching
                variant = variant_full  # Keep full string like 'aug4_hflip_vflip_rotate_translate'
            
            if prefix not in pairs:
                pairs[prefix] = {}
            pairs[prefix][variant] = img
            
    valid_pairs = []
    for prefix, variants in pairs.items():
        if 'original' not in variants:
            continue
        orig_img = variants['original']
        orig_lbl = find_matching_label(orig_img, all_lbls)
        if not orig_lbl:
            continue
        orig_lbl_path = lbl_dir / orig_lbl
        if class_filter is not None and not cached_label_contains_class(orig_lbl_path, class_filter):
            continue
        for variant_name, aug_img in variants.items():
            if variant_name == 'original':
                continue
            aug_lbl = find_matching_label(aug_img, all_lbls)
            if not aug_lbl:
                continue
            aug_lbl_path = lbl_dir / aug_lbl
            if class_filter is not None and not cached_label_contains_class(aug_lbl_path, class_filter):
                continue
            valid_pairs.append(((orig_img, orig_lbl), (aug_img, aug_lbl)))
    
    return valid_pairs

def view_mosaic_images(img_dir, lbl_dir, valid_mosaics, max_images, amp_clip_min, amp_clip_max, save_path):
    """Handle mosaic image visualization"""
    sample = random.sample(valid_mosaics, min(max_images, len(valid_mosaics)))
    num_images = len(sample)
    
    # Set up 2x5 grid for mosaics
    num_cols = 5
    num_rows = 2
    
    fig, gs = setup_figure_and_grid(num_cols, num_rows, "Mosaic Augmentation Examples")
    
    row_axes = [[] for _ in range(num_rows)]
    
    for i, (img_file, lbl_file) in enumerate(sample):
        row = i // num_cols
        col = i % num_cols
        
        ax = fig.add_subplot(gs[row + 1, col])
        row_axes[row].append(ax)
        
        img_path = img_dir / img_file
        label_path = lbl_dir / lbl_file
        
        if plot_image_with_boxes(ax, img_path, label_path, amp_clip_min, amp_clip_max):
            # Set title as filename without .npy extension
            title = img_file.replace('.npy', '')
            ax.set_title(title, fontsize=8)
    
    # Add colorbars for each row that has images
    add_colorbars_to_rows(fig, row_axes, amp_clip_min, amp_clip_max)
    
    fig.text(0.5, 0.03, "Figure: Randomly selected mosaic images with bounding box annotations.",
             ha='center', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def view_augmented_pairs(img_dir, lbl_dir, valid_pairs, max_images, amp_clip_min, amp_clip_max, save_path):
    """Handle augmented image pair visualization"""
    sample = random.sample(valid_pairs, min(max_images // 2, len(valid_pairs)))
    num_pairs = len(sample)
    num_images = num_pairs * 2
    num_cols = min(5, num_pairs)
    num_rows = 2

    fig, gs = setup_figure_and_grid(num_cols, num_rows, "Raw Data Augmentation Examples")
    row_axes = [[] for _ in range(num_rows)]

    for col, ((orig_img, orig_lbl), (aug_img, aug_lbl)) in enumerate(sample):
        for row, (img_file, lbl_file) in enumerate([(orig_img, orig_lbl), (aug_img, aug_lbl)]):
            ax = fig.add_subplot(gs[row + 1, col])
            row_axes[row].append(ax)
            img_path = img_dir / img_file
            label_path = lbl_dir / lbl_file

            if plot_image_with_boxes(ax, img_path, label_path, amp_clip_min, amp_clip_max):
                if row == 0:
                    title_base = img_file.replace('.npy', '')
                    first_underscore = title_base.find('_')
                    if first_underscore != -1:
                        title_base = title_base[:first_underscore + 1] + '\n' + title_base[first_underscore + 1:]
                    title_wrapped = title_base.replace('swath', '\nswath')
                    ax.set_title(title_wrapped, fontsize=7, loc='center')
                elif row == 1:
                    # Extract the full augmentation description
                    aug_match = re.search(r'(aug\d+_[^.]+)', lbl_file.replace('.txt', ''))
                    if aug_match:
                        full_aug = aug_match.group(1)
                        # Show the full augmentation but limit length for display
                        if len(full_aug) > 25:
                            display_title = full_aug[:22] + '...'
                        else:
                            display_title = full_aug
                    else:
                        display_title = ''
                    ax.set_title(display_title, fontsize=7)
               
    add_colorbars_to_rows(fig, row_axes, amp_clip_min, amp_clip_max)

    fig.text(0.5, 0.03, "Figure: Top row: original images. Bottom row: corresponding augmented versions.",
             ha='center', fontsize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def view_augmentations(base_dir='.', max_images=10, class_filter=None, save_path=None, 
                       amp_clip_min=None, amp_clip_max=None, select_mosaics=False,
                       img_subdir='images', lbl_subdir='labels'):
    """
    Visualize SAR augmentation examples for unprocessed augmented data.
    
    Parameters:
    -----------
    base_dir : str, default '.'
        Base directory containing image and label subdirectories
    max_images : int, default 10
        Maximum number of images to display
    class_filter : int, optional
        Filter images containing specific class ID
    save_path : str, optional
        Path to save figure instead of displaying
    amp_clip_min, amp_clip_max : float, optional
        Raw amplitude clipping values. Defaults: 1.00 (1%) and 71.51 (99%) percentiles
    select_mosaics : bool, default False
        Whether to display mosaic images instead of augmented pairs
    img_subdir : str, default 'images'
        Subdirectory name containing .npy image files (shape: H, W, dtype: complex64)
    lbl_subdir : str, default 'labels'
        Subdirectory name containing .txt label files
    """
    img_dir = Path(base_dir) / img_subdir
    lbl_dir = Path(base_dir) / lbl_subdir

    # Collect only .npy image files
    all_imgs = sorted([p.name for p in img_dir.glob('*.npy')])
    all_lbls = sorted([p.name for p in lbl_dir.glob('*.txt')])

    if not all_imgs:
        print(f"No .npy files found in {img_dir}")
        return

    # Pre-load all label files into cache if class filtering is needed
    if class_filter is not None:
        preload_label_cache(lbl_dir, all_lbls)

    if select_mosaics:
        valid_mosaics = get_valid_mosaic_pairs(all_imgs, all_lbls, lbl_dir, class_filter)
        if not valid_mosaics:
            print("No valid mosaic image-label pairs found.")
            return
        view_mosaic_images(img_dir, lbl_dir, valid_mosaics, max_images, 
                          amp_clip_min, amp_clip_max, save_path)
    else:
        valid_pairs = get_valid_augmented_pairs(all_imgs, all_lbls, lbl_dir, class_filter)
        if not valid_pairs:
            print("No valid original-augmented image-label pairs found.")
            return
        view_augmented_pairs(img_dir, lbl_dir, valid_pairs, max_images,
                           amp_clip_min, amp_clip_max, save_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize SAR augmentation examples for unprocessed augmented data (complex64 .npy files)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View augmentation pairs (uses default amplitude clipping: 1.00 to 71.51)
  %(prog)s data/train_hvrt_bal_alt
  
  # Save visualization
  %(prog)s data/train_hvrt_bal_alt --max-images 4 --save-path output.png
  
  # View with custom amplitude clipping
  %(prog)s data/train_hvrt_bal_alt --amp-clip-min 2.0 --amp-clip-max 50.0
  
  # View mosaic images
  %(prog)s data/train_hvrt_bal_alt --select-mosaics
        """
    )
    
    parser.add_argument('base_dir', default='.', nargs='?',
                       help='Base directory containing image and label subdirectories (default: current directory)')
    parser.add_argument('--max-images', type=int, default=10,
                       help='Maximum number of images to display (default: 10)')
    parser.add_argument('--class-filter', type=int, choices=[0, 1],
                       help='Filter images containing specific class ID (0=is_vessel, 1=is_fishing)')
    parser.add_argument('--save-path',
                       help='Path to save figure instead of displaying')
    parser.add_argument('--amp-clip-min', type=float,
                       help='Minimum raw amplitude value for clipping (default: 1.00)')
    parser.add_argument('--amp-clip-max', type=float,
                       help='Maximum raw amplitude value for clipping (default: 71.51)')
    parser.add_argument('--select-mosaics', action='store_true',
                       help='Display mosaic images instead of augmented pairs')
    parser.add_argument('--img-subdir', default='images',
                       help='Subdirectory name containing .npy image files (shape: H, W, dtype: complex64)')
    parser.add_argument('--lbl-subdir', default='labels',
                       help='Subdirectory name containing .txt label files (default: labels)')
    
    args = parser.parse_args()
    
    # Call the main function
    view_augmentations(
        base_dir=args.base_dir,
        max_images=args.max_images,
        class_filter=args.class_filter,
        save_path=args.save_path,
        amp_clip_min=args.amp_clip_min,
        amp_clip_max=args.amp_clip_max,
        select_mosaics=args.select_mosaics,
        img_subdir=args.img_subdir,
        lbl_subdir=args.lbl_subdir
    )