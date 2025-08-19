#!/usr/bin/env python3

"""
view_png_augmentations.py

Author: Peter Millitz
Created: 2025-08-17

A visualization tool for QC of PNG-processed SAR image crops.
Mimics view_augmentations.py but works with PNG files that have been processed from .npy files.

This tool compares original vs augmented PNG versions, displays amplitude with optional clipping,
and supports non-augmentation mode for inference scenarios.

Expected directory structure:
- images/: contains .png files processed from complex SAR data
- labels/: contains YOLO-format .txt files with corresponding stems

PNG Format:
- Red channel: Normalized amplitude (0-255) → divide by 255 to get [0,1] range
- Green channel: Normalized phase (0-255) 
- Blue channel: Zeros

Displays up to 5 randomly selected image-label pairs in two rows (max 5 per row).
Each original-augmented image pair is aligned vertically. Handles multiple bounding
boxes per image and supports amplitude clipping using actual amplitude bounds.

Supports non-augmentation mode (10 random images) when no augmented pairs are found.
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

_label_cache = {}

def load_png_amplitude(img_path):
    """Load PNG image and extract amplitude from red channel
    
    Args:
        img_path: Path to .png file
        
    Returns:
        numpy.ndarray: Amplitude data normalized to [0,1] from red channel
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
    
    # Extract amplitude from red channel and normalize to [0,1]
    amplitude = rgb_data[:, :, 0].astype(np.float32) / 255.0
    
    return amplitude

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

def find_matching_mosaic_label(image_filename, label_filenames):
    """Find matching label file for mosaic PNG images"""
    # Extract X from mosaic_minority_X_proc.png or mosaic_majority_X_proc.png
    match = re.match(r'mosaic_(minority|majority)_(\d+)_proc\.png$', image_filename)
    if match:
        mosaic_type = match.group(1)
        x = match.group(2)
        candidate = f'mosaic_{mosaic_type}_{x}_proc.txt'
        return candidate if candidate in label_filenames else None
    return None

def get_valid_mosaic_png_pairs(all_imgs, all_lbls, lbl_dir, class_filter):
    """Get valid mosaic PNG image-label pairs"""
    mosaic_imgs = [img for img in all_imgs if re.match(r'mosaic_(minority|majority)_\d+_proc\.png$', img)]
    
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

def setup_figure_and_grid(num_cols, num_rows, title):
    """Set up matplotlib figure and grid"""
    fig = plt.figure(figsize=(4 * num_cols + 1.5, 5 * num_rows), dpi=100)
    gs = GridSpec(num_rows + 1, num_cols + 1, figure=fig, 
                  width_ratios=[1]*num_cols + [0.05], 
                  height_ratios=[0.2] + [1]*num_rows, 
                  wspace=0.1, hspace=0.3)
    fig.suptitle(title, fontsize=18, y=0.90)
    return fig, gs

def plot_png_image_with_boxes(ax, img_path, label_path, amp_clip_min, amp_clip_max):
    """Plot single PNG image with bounding boxes"""
    try:
        amplitude = load_png_amplitude(img_path)  # Normalized amplitude values [0,1]
    except (ValueError, IOError) as e:
        print(f"Skipping invalid image: {img_path} - {e}")
        return False
    
    # Default clipping values (1% and 99% percentiles of train dataset)
    default_amp_min = 1.00
    default_amp_max = 71.51
    
    # Use provided clipping values or defaults
    clip_min = amp_clip_min if amp_clip_min is not None else default_amp_min
    clip_max = amp_clip_max if amp_clip_max is not None else default_amp_max
    
    # PNG files store normalized dB values, not amplitude values!
    # PNG creation process was: amplitude [1.00, 71.51] → dB → normalize to [0,1] → PNG
    # So PNG red channel contains normalized dB values in [0,1] range
    
    PNG_AMP_MIN = 1.00  # Original amplitude clipping used during PNG creation
    PNG_AMP_MAX = 71.51
    
    # Step 1: PNG red channel contains normalized dB values [0,1]
    normalized_db = amplitude  # This is actually normalized dB, not amplitude!
    
    # Step 2: Reconstruct the actual dB values used during PNG creation
    png_db_min = 20 * np.log10(PNG_AMP_MIN + 1e-10)
    png_db_max = 20 * np.log10(PNG_AMP_MAX + 1e-10) 
    actual_db = normalized_db * (png_db_max - png_db_min) + png_db_min
    
    # Step 3: Convert back to amplitude values
    reconstructed_amplitude = 10 ** (actual_db / 20)
    
    # Step 4: Now apply any additional clipping (same as .npy workflow)
    clipped_amplitude = np.clip(reconstructed_amplitude, clip_min, clip_max)
    
    # Step 5: Convert to dB for display
    amplitude_db = 20 * np.log10(clipped_amplitude + 1e-10)
    
    # Step 6: Normalize for display using the clipping bounds
    db_min = 20 * np.log10(clip_min + 1e-10)
    db_max = 20 * np.log10(clip_max + 1e-10)
    
    if db_max > db_min:
        display_data = (amplitude_db - db_min) / (db_max - db_min)
    else:
        display_data = np.zeros_like(amplitude_db)
    
    # Display with fixed [0,1] range
    ax.imshow(display_data, cmap='gray', vmin=0.0, vmax=1.0, aspect='equal')
    
    # Add bounding boxes
    bboxes = load_all_bounding_boxes(label_path, *amplitude.shape)
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

def get_valid_augmented_png_pairs(all_imgs, all_lbls, lbl_dir, class_filter):
    """Get valid original-augmented PNG image pairs"""
    pairs = {}
    for img in all_imgs:
        # Match pattern for processed PNG files (with _proc suffix)
        # Look for: filename_swath[1-3]_original_proc.png or filename_swath[1-3]_aug[0-9]+_augtype_proc.png
        match = re.match(r'(.+_swath\d+)_(original|aug\d+_[^_]+(?:_[^_]+)*)_proc\.png$', img)
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

def get_random_png_images(all_imgs, all_lbls, lbl_dir, class_filter, max_images=10):
    """Get random PNG images for non-augmentation mode"""
    valid_images = []
    for img in all_imgs:
        if img.endswith('_proc.png'):
            label_match = find_matching_label(img, all_lbls)
            if label_match:
                label_path = lbl_dir / label_match
                if class_filter is None or cached_label_contains_class(label_path, class_filter):
                    valid_images.append((img, label_match))
    
    return random.sample(valid_images, min(max_images, len(valid_images)))

def view_mosaic_png_images(img_dir, lbl_dir, valid_mosaics, max_images, amp_clip_min, amp_clip_max, save_path):
    """Handle mosaic PNG image visualization"""
    sample = random.sample(valid_mosaics, min(max_images, len(valid_mosaics)))
    num_images = len(sample)
    
    # Set up 2x5 grid for mosaics
    num_cols = 5
    num_rows = 2
    
    fig, gs = setup_figure_and_grid(num_cols, num_rows, "PNG Mosaic Augmentation Examples")
    
    row_axes = [[] for _ in range(num_rows)]
    
    for i, (img_file, lbl_file) in enumerate(sample):
        row = i // num_cols
        col = i % num_cols
        
        if row >= num_rows:
            break
            
        ax = fig.add_subplot(gs[row + 1, col])
        row_axes[row].append(ax)
        
        img_path = img_dir / img_file
        label_path = lbl_dir / lbl_file
        
        if plot_png_image_with_boxes(ax, img_path, label_path, amp_clip_min, amp_clip_max):
            # Set title as filename without _proc.png extension with proper wrapping
            title_base = img_file.replace('_proc.png', '')
            # Apply same wrapping logic as other modes
            first_underscore = title_base.find('_')
            if first_underscore != -1:
                title_base = title_base[:first_underscore + 1] + '\n' + title_base[first_underscore + 1:]
            title_wrapped = title_base.replace('minority', '\nminority')
            ax.set_title(title_wrapped, fontsize=7, loc='center')
    
    # Add colorbars for each row that has images
    add_colorbars_to_rows(fig, row_axes, amp_clip_min, amp_clip_max)
    
    fig.text(0.5, 0.03, "Figure: Randomly selected PNG mosaic images with bounding box annotations.",
             ha='center', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def view_augmented_png_pairs(img_dir, lbl_dir, valid_pairs, max_images, amp_clip_min, amp_clip_max, save_path):
    """Handle augmented PNG image pair visualization"""
    sample = random.sample(valid_pairs, min(max_images // 2, len(valid_pairs)))
    num_pairs = len(sample)
    num_images = num_pairs * 2
    num_cols = min(5, num_pairs)
    num_rows = 2

    fig, gs = setup_figure_and_grid(num_cols, num_rows, "PNG Data Augmentation Examples")
    row_axes = [[] for _ in range(num_rows)]

    for col, ((orig_img, orig_lbl), (aug_img, aug_lbl)) in enumerate(sample):
        for row, (img_file, lbl_file) in enumerate([(orig_img, orig_lbl), (aug_img, aug_lbl)]):
            ax = fig.add_subplot(gs[row + 1, col])
            row_axes[row].append(ax)
            img_path = img_dir / img_file
            label_path = lbl_dir / lbl_file

            if plot_png_image_with_boxes(ax, img_path, label_path, amp_clip_min, amp_clip_max):
                if row == 0:
                    title_base = img_file.replace('_proc.png', '')
                    first_underscore = title_base.find('_')
                    if first_underscore != -1:
                        title_base = title_base[:first_underscore + 1] + '\n' + title_base[first_underscore + 1:]
                    title_wrapped = title_base.replace('swath', '\nswath')
                    ax.set_title(title_wrapped, fontsize=7, loc='center')
                elif row == 1:
                    # Extract the full augmentation description, removing the _proc.txt suffix
                    aug_match = re.search(r'(aug\d+_[^.]+)', lbl_file.replace('_proc.txt', ''))
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

    fig.text(0.5, 0.03, "Figure: Top row: original PNG images. Bottom row: corresponding augmented PNG versions.",
             ha='center', fontsize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def view_random_png_images(img_dir, lbl_dir, random_images, max_images, amp_clip_min, amp_clip_max, save_path):
    """Handle random PNG image visualization (non-augmentation mode)"""
    sample = random_images[:max_images]
    num_images = len(sample)
    
    # Set up 2x5 grid for random images
    num_cols = 5
    num_rows = 2
    
    fig, gs = setup_figure_and_grid(num_cols, num_rows, "Random PNG Image Examples")
    
    row_axes = [[] for _ in range(num_rows)]
    
    for i, (img_file, lbl_file) in enumerate(sample):
        row = i // num_cols
        col = i % num_cols
        
        if row >= num_rows:
            break
            
        ax = fig.add_subplot(gs[row + 1, col])
        row_axes[row].append(ax)
        
        img_path = img_dir / img_file
        label_path = lbl_dir / lbl_file
        
        if plot_png_image_with_boxes(ax, img_path, label_path, amp_clip_min, amp_clip_max):
            # Set title as filename without _proc.png extension with proper wrapping
            title_base = img_file.replace('_proc.png', '')
            # Apply same wrapping logic as augmentation mode
            first_underscore = title_base.find('_')
            if first_underscore != -1:
                title_base = title_base[:first_underscore + 1] + '\n' + title_base[first_underscore + 1:]
            title_wrapped = title_base.replace('swath', '\nswath')
            ax.set_title(title_wrapped, fontsize=7, loc='center')
    
    # Add colorbars for each row that has images
    add_colorbars_to_rows(fig, row_axes, amp_clip_min, amp_clip_max)
    
    fig.text(0.5, 0.03, "Figure: Randomly selected PNG images with bounding box annotations.",
             ha='center', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def view_png_augmentations(base_dir='.', max_images=10, class_filter=None, save_path=None, 
                          amp_clip_min=None, amp_clip_max=None, select_mosaics=False,
                          img_subdir='images', lbl_subdir='labels'):
    """
    Visualize PNG-processed SAR augmentation examples.
    
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
        Amplitude clipping values for consistent display scaling
    select_mosaics : bool, default False
        Whether to display mosaic images instead of augmented pairs
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

    # Pre-load all label files into cache if class filtering is needed
    if class_filter is not None:
        preload_label_cache(lbl_dir, all_lbls)

    if select_mosaics:
        valid_mosaics = get_valid_mosaic_png_pairs(all_imgs, all_lbls, lbl_dir, class_filter)
        if not valid_mosaics:
            print("No valid mosaic PNG image-label pairs found.")
            return
        view_mosaic_png_images(img_dir, lbl_dir, valid_mosaics, max_images, 
                              amp_clip_min, amp_clip_max, save_path)
    else:
        # Try to find augmented pairs first
        valid_pairs = get_valid_augmented_png_pairs(all_imgs, all_lbls, lbl_dir, class_filter)
        
        if valid_pairs:
            print(f"Found {len(valid_pairs)} augmented pairs")
            view_augmented_png_pairs(img_dir, lbl_dir, valid_pairs, max_images,
                                    amp_clip_min, amp_clip_max, save_path)
        else:
            print("No augmented pairs found, switching to non-augmentation mode")
            random_images = get_random_png_images(all_imgs, all_lbls, lbl_dir, class_filter, max_images)
            if not random_images:
                print("No valid PNG image-label pairs found.")
                return
            view_random_png_images(img_dir, lbl_dir, random_images, max_images,
                                 amp_clip_min, amp_clip_max, save_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize PNG-processed SAR augmentation examples for QC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View PNG augmentation pairs
  %(prog)s data/train_hvrt_bal_alt_processed
  
  # Save visualization
  %(prog)s data/train_hvrt_bal_alt_processed --max-images 4 --save-path output.png
  
  # View with amplitude clipping
  %(prog)s data/train_hvrt_bal_alt_processed --amp-clip-min 2.0 --amp-clip-max 50.0
  
  # Filter by class
  %(prog)s data/train_hvrt_bal_alt_processed --class-filter 1
  
  # View PNG mosaic images
  %(prog)s data/train_hvrt_bal_alt_processed --select-mosaics
        """
    )
    
    parser.add_argument('base_dir', default='.', nargs='?',
                       help='Base directory containing PNG image and label subdirectories (default: current directory)')
    parser.add_argument('--max-images', type=int, default=10,
                       help='Maximum number of images to display (default: 10)')
    parser.add_argument('--class-filter', type=int, choices=[0, 1],
                       help='Filter images containing specific class ID (0=is_vessel, 1=is_fishing)')
    parser.add_argument('--save-path',
                       help='Path to save figure instead of displaying')
    parser.add_argument('--amp-clip-min', type=float,
                       help='Minimum amplitude value for clipping (scaled to PNG range)')
    parser.add_argument('--amp-clip-max', type=float,
                       help='Maximum amplitude value for clipping (scaled to PNG range)')
    parser.add_argument('--select-mosaics', action='store_true',
                       help='Display mosaic images instead of augmented pairs')
    parser.add_argument('--img-subdir', default='images',
                       help='Subdirectory name containing .png image files (default: images)')
    parser.add_argument('--lbl-subdir', default='labels',
                       help='Subdirectory name containing .txt label files (default: labels)')
    
    args = parser.parse_args()
    
    # Call the main function
    view_png_augmentations(
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