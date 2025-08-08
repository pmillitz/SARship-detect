#!/usr/bin/env python3

# Version 24 (<=> v35) stable version for augmented processed images
# (Note: Formerly named 'view_scaled_magnitude.py')

"""
view_augmentations.py

A visualization tool for inspecting augmented SAR SLC (3, H, W)-shaped image crops.
Only the first channel (scaled + normalized dB magnitude) is displayed in grayscale.

Expected directory structure:
- images/: contains .npy files with shape (3, H, W)
- labels/: contains YOLO-format .txt files with same stem, excluding '_proc' string.

Displays up to 5 randomly selected image-label pairs in two rows (max 5 per row).
Each original-augmented image pair is aligned vertically.Handles multiple bounding
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
import textwrap

_label_cache = {}

def find_matching_label(image_filename, label_filenames):
    candidate = image_filename.replace('_proc.npy', '.txt')
    return candidate if candidate in label_filenames else None

def find_matching_mosaic_label(image_filename, label_filenames):
    """Find matching label file for mosaic images"""
    # Extract X from mosaic_minority_X.npy
    match = re.match(r'mosaic_minority_(\d+)_proc\.npy$', image_filename)
    if match:
        x = match.group(1)
        candidate = f'mosaic_minority_{x}.txt'
        return candidate if candidate in label_filenames else None
    return None

def load_all_bounding_boxes(label_path, img_height, img_width):
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

def db_to_normalized(db_value, db_min=0.00, db_max=37.09):
    return (db_value - db_min) / (db_max - db_min)

def normalized_to_db(norm_value, db_min=0.00, db_max=37.09):
    return norm_value * (db_max - db_min) + db_min

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

def plot_image_with_boxes(ax, img_path, label_path, display_min, display_max):
    """Plot single image with bounding boxes"""
    img_data = np.load(img_path)
    if img_data.ndim != 3 or img_data.shape[0] != 3:
        print(f"Skipping invalid image shape: {img_path}")
        return False
    
    # Apply clipping if specified
    mag = img_data[0]
    if display_min is not None or display_max is not None:
        mag = np.clip(mag, display_min or 0.0, display_max or 1.0)
    
    ax.imshow(mag, cmap='gray', vmin=display_min or 0.0, vmax=display_max or 1.0, aspect='equal')
    
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

def add_colorbars_to_rows(fig, row_axes, display_min, display_max, data_db_min, data_db_max):
    """Add colorbars for each row that has images"""
    norm = Normalize(vmin=display_min, vmax=display_max)
    
    for row, axes in enumerate(row_axes):
        if axes:
            row_top = min(ax.get_position().y0 for ax in axes)
            row_bottom = max(ax.get_position().y1 for ax in axes)
            row_right = max(ax.get_position().x1 for ax in axes)
            
            cbar_ax = fig.add_axes([row_right + 0.01, row_top, 0.02, row_bottom - row_top])
            cb = ColorbarBase(cbar_ax, cmap=cm.gray, norm=norm, orientation='vertical')
            cb.set_label("dB Magnitude", fontsize=10)
            
            # Set endpoint ticks in dB
            tick_vals = [display_min, display_max]
            tick_labels = [f"{normalized_to_db(display_min, data_db_min, data_db_max):.2f}", 
                          f"{normalized_to_db(display_max, data_db_min, data_db_max):.2f}"]
            cb.set_ticks(tick_vals)
            cb.set_ticklabels(tick_labels)
            cb.ax.tick_params(labelsize=8)

def get_valid_mosaic_pairs(all_imgs, all_lbls, lbl_dir, class_filter):
    """Get valid mosaic image-label pairs"""
    mosaic_imgs = [img for img in all_imgs if re.match(r'mosaic_minority_\d+_proc\.npy$', img)]
    
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
        match = re.match(r'(.+_swath\d+)_(original|aug\d+_\w+)_proc\.npy$', img)
        if match:
            prefix = match.group(1)
            variant_full = match.group(2)
            
            # Extract the actual variant name
            if variant_full == 'original':
                variant = 'original'
            else:
                # For aug0_vflip, aug1_hflip, etc., extract the last part
                variant = variant_full.split('_')[-1]  # Gets 'vflip', 'hflip', etc.
            
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

def view_mosaic_images(img_dir, lbl_dir, valid_mosaics, max_images, display_min, display_max, 
                      data_db_min, data_db_max, save_path):
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
        
        if plot_image_with_boxes(ax, img_path, label_path, display_min, display_max):
            # Set title as filename without .npy extension
            title = img_file.replace('.npy', '')
            ax.set_title(title, fontsize=8)
    
    # Add colorbars for each row that has images
    add_colorbars_to_rows(fig, row_axes, display_min, display_max, data_db_min, data_db_max)
    
    fig.text(0.5, 0.03, "Figure: Randomly selected mosaic images with bounding box annotations.",
             ha='center', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    print(f"Displayed {num_images} mosaic image crops with bounding boxes.")

def view_augmented_pairs(img_dir, lbl_dir, valid_pairs, max_images, display_min, display_max,
                        data_db_min, data_db_max, save_path):
    """Handle augmented image pair visualization"""
    sample = random.sample(valid_pairs, min(max_images // 2, len(valid_pairs)))
    num_pairs = len(sample)
    num_images = num_pairs * 2
    num_cols = min(5, num_pairs)
    num_rows = 2

    fig, gs = setup_figure_and_grid(num_cols, num_rows, "Data Augmentation Examples")
    row_axes = [[] for _ in range(num_rows)]

    for col, ((orig_img, orig_lbl), (aug_img, aug_lbl)) in enumerate(sample):
        for row, (img_file, lbl_file) in enumerate([(orig_img, orig_lbl), (aug_img, aug_lbl)]):
            ax = fig.add_subplot(gs[row + 1, col])
            row_axes[row].append(ax)
            img_path = img_dir / img_file
            label_path = lbl_dir / lbl_file

            if plot_image_with_boxes(ax, img_path, label_path, display_min, display_max):
                if row == 0:
                    title_base = img_file.replace('.npy', '')
                    first_underscore = title_base.find('_')
                    if first_underscore != -1:
                        title_base = title_base[:first_underscore + 1] + '\n' + title_base[first_underscore + 1:]
                    title_wrapped = title_base.replace('swath', '\nswath')
                    ax.set_title(title_wrapped, fontsize=7, loc='center')
                elif row == 1:
                    aug_title = re.search(r'(aug[^.]+)', lbl_file)
                    ax.set_title(aug_title.group(1) if aug_title else '', fontsize=8)
               
    add_colorbars_to_rows(fig, row_axes, display_min, display_max, data_db_min, data_db_max)

    fig.text(0.5, 0.03, "Figure: Top row: original images. Bottom row: corresponding augmented versions.",
             ha='center', fontsize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    print(f"Displayed {num_images} image crops with bounding boxes.")

def view_augmentations(base_dir='.', max_images=10, class_filter=None, save_path=None, 
                       clip_db_min=None, clip_db_max=None, select_mosaics=False, 
                       data_db_min=0.00, data_db_max=37.09,
                       img_subdir='images', lbl_subdir='labels'):
    """
    Visualize SAR augmentation examples.
    
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
    clip_db_min, clip_db_max : float, optional
        Clipping values in dB units
    select_mosaics : bool, default False
        Whether to display mosaic images instead of augmented pairs
    data_db_min, data_db_max : float, default (0.00, 37.09)
        The actual dB range of your input data for conversion calculations
    img_subdir : str, default 'images'
        Subdirectory name containing .npy image files
    lbl_subdir : str, default 'labels'
        Subdirectory name containing .txt label files
    """
    img_dir = Path(base_dir) / img_subdir
    lbl_dir = Path(base_dir) / lbl_subdir

    all_imgs = sorted([p.name for p in img_dir.glob('*.npy')])
    all_lbls = sorted([p.name for p in lbl_dir.glob('*.txt')])

    # Pre-load all label files into cache if class filtering is needed
    if class_filter is not None:
        preload_label_cache(lbl_dir, all_lbls)

    # Validate and convert dB clipping parameters
    clip_min_norm, clip_max_norm = validate_and_convert_clipping(clip_db_min, clip_db_max, data_db_min, data_db_max)
    
    # Set display range
    display_min = clip_min_norm if clip_min_norm is not None else 0.0
    display_max = clip_max_norm if clip_max_norm is not None else 1.0

    if select_mosaics:
        valid_mosaics = get_valid_mosaic_pairs(all_imgs, all_lbls, lbl_dir, class_filter)
        if not valid_mosaics:
            print("No valid mosaic image-label pairs found.")
            return
        view_mosaic_images(img_dir, lbl_dir, valid_mosaics, max_images, 
                          display_min, display_max, data_db_min, data_db_max, save_path)
    else:
        valid_pairs = get_valid_augmented_pairs(all_imgs, all_lbls, lbl_dir, class_filter)
        if not valid_pairs:
            print("No valid original-augmented image-label pairs found.")
            return
        view_augmented_pairs(img_dir, lbl_dir, valid_pairs, max_images,
                           display_min, display_max, data_db_min, data_db_max, save_path)