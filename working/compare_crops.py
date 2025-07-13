#!/usr/bin/env python3

"""
compare_crops.py

Author: Peter Millitz
Created: 2025-07-12

"""
    
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def crop_compare(image_list1, image_list2=None, base_dir='.', 
                amp_min=1.0, amp_max=69.4, log_min=9.0, log_max=30.4, real_min=-20, real_max=20,
                mode_row1='magnitude', mode_row2='magnitude'):
    """
    Compare SAR image crops showing magnitude, log-magnitude, and real parts with normalization.
    
    Parameters:
    -----------
    image_list1 : list
        List of 5 image filenames (without path, in 'raw' format)
    image_list2 : list, optional
        Second list of 5 image filenames. If None, uses image_list1 for both rows
    base_dir : str
        Base directory path containing 'images' and 'labels' subdirectories
    amp_min, amp_max : float
        Min/max values for magnitude normalization (default: 1.0, 69.4)
    log_min, log_max : float
        Min/max values for log-magnitude (dB) normalization (default: 9.0, 30.4)
    real_min, real_max : float
        Min/max values for real part normalization (default: -20, 20)
    mode_row1, mode_row2 : str
        Display mode for each row: 'magnitude', 'log-magnitude', or 'real'
    """
    
    def load_sar_image_magnitude(image_path):
        """Load SAR image and extract magnitude"""
        img_data = np.load(image_path)
        magnitude = np.abs(img_data)
        return magnitude

    def load_sar_image_log_magnitude(image_path):
        """Load SAR image and extract log-magnitude (dB)"""
        img_data = np.load(image_path)
        magnitude = np.abs(img_data)
        # Convert to dB, avoiding log(0) by adding small epsilon
        log_magnitude = 20 * np.log10(magnitude + 1e-10)
        return log_magnitude

    def load_sar_image_real(image_path):
        """Load SAR image and extract real part"""
        img_data = np.load(image_path)
        real_part = np.real(img_data)
        return real_part

    def load_bounding_box(label_path, img_height, img_width):
        """Load bounding box from YOLO format and convert to pixel coordinates"""
        try:
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    
                    if len(parts) >= 4:
                        if len(parts) == 5:
                            # Format: class_id center_x center_y width height
                            class_id, center_x, center_y, width, height = map(float, parts)
                        else:
                            # Format: center_x center_y width height
                            center_x, center_y, width, height = map(float, parts)
                            class_id = 0  # Default class if not provided
                        
                        # Convert from normalized coordinates to pixel coordinates
                        pixel_center_x = center_x * img_width
                        pixel_center_y = center_y * img_height
                        pixel_width = width * img_width
                        pixel_height = height * img_height
                        
                        # Convert from center coordinates to top-left corner
                        x = pixel_center_x - pixel_width / 2
                        y = pixel_center_y - pixel_height / 2
                        
                        return (x, y, pixel_width, pixel_height, int(class_id))
            return None
        except Exception as e:
            print(f"Error loading bounding box from {label_path}: {e}")
            return None

    def normalize_image(img_data, min_val, max_val):
        """Normalize image to [0, 1] range using dataset min/max values"""
        normalized = (img_data - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)

    def process_image(image_path, mode):
        """Process image according to specified mode"""
        if mode == 'magnitude':
            img_data = load_sar_image_magnitude(image_path)
            img_normalized = normalize_image(img_data, amp_min, amp_max)
            return img_data, img_normalized
        elif mode == 'log-magnitude':
            img_data = load_sar_image_log_magnitude(image_path)
            img_normalized = normalize_image(img_data, log_min, log_max)
            return img_data, img_normalized
        elif mode == 'real':
            img_data = load_sar_image_real(image_path)
            img_normalized = normalize_image(img_data, real_min, real_max)
            return img_data, img_normalized
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'magnitude', 'log-magnitude', or 'real'")

    # Build full paths
    def build_paths(image_list):
        image_paths = [f"{base_dir}/images/{filename}" for filename in image_list]
        label_paths = [f"{base_dir}/labels/{filename.replace('.npy', '.txt')}" for filename in image_list]
        return image_paths, label_paths

    # Validate input lists
    if len(image_list1) < 1:
        print("Error: image_list1 must contain at least 1 image.")
        return
    if len(image_list1) > 5:
        print("Error: image_list1 cannot contain more than 5 images.")
        return
    
    if image_list2 is not None:
        if len(image_list2) < 1:
            print("Error: image_list2 must contain at least 1 image.")
            return
        if len(image_list2) > 5:
            print("Error: image_list2 cannot contain more than 5 images.")
            return
        if len(image_list1) != len(image_list2):
            print("Error: image_list1 and image_list2 must have the same number of images.")
            return
    
    # Determine number of images to display
    num_images = len(image_list1)
    
    # Determine display mode
    single_list_mode = image_list2 is None
    
    # Get paths
    image_paths1, label_paths1 = build_paths(image_list1)
    if not single_list_mode:
        image_paths2, label_paths2 = build_paths(image_list2)
    else:
        image_paths2, label_paths2 = image_paths1, label_paths1

    # Create figure with proper spacing for vertical colorbars
    fig = plt.figure(figsize=(4*num_images + 2, 10))
    
    # Use GridSpec for better control over layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, num_images + 1, figure=fig, 
                  width_ratios=[1]*num_images + [0.1],  # Last column narrower for colorbar
                  hspace=0.3, wspace=0.1)

    # Create normalization objects for colorbars
    def get_norm_and_label(mode):
        if mode == 'magnitude':
            return Normalize(vmin=amp_min, vmax=amp_max), 'Magnitude'
        elif mode == 'log-magnitude':
            return Normalize(vmin=log_min, vmax=log_max), 'Log-Magnitude (dB)'
        elif mode == 'real':
            return Normalize(vmin=real_min, vmax=real_max), 'Real Part'
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    norm1, label1 = get_norm_and_label(mode_row1)
    norm2, label2 = get_norm_and_label(mode_row2)
    
    # First row
    for i in range(num_images):
        ax = fig.add_subplot(gs[0, i])  # Use GridSpec positioning
        
        # Process image according to mode
        img_data, img_normalized = process_image(image_paths1[i], mode_row1)
        img_height, img_width = img_data.shape
        
        # Display normalized image
        ax.imshow(img_normalized, cmap='gray', vmin=0, vmax=1, aspect='equal')
        
        # Load and display bounding box
        bbox = load_bounding_box(label_paths1[i], img_height, img_width)
        if bbox:
            x, y, width, height, class_id = bbox
            rect = Rectangle((x, y), width, height, 
                            linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            
            # Add class label positioned off NW corner of BBox (exact same as visualise_crop)
            label_str = "is_fishing" if class_id == 1 else "is_vessel"
            label_x = x - 2   # 2 pixels left
            label_y = y
            ax.text(
                label_x, label_y, label_str,
                color='lime',
                fontsize=8,  # Slightly smaller for crop_compare grid layout
                verticalalignment='top',     # Top of label aligns with (x, y)
                horizontalalignment='right'  # Right end of label aligns with (x, y)
            )
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add row label
        if i == 0:
            row_label = f"Set 1 ({label1})" if not single_list_mode else label1
            ax.text(-0.15, 0.5, row_label, transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)

    # Add vertical colorbar for first row (aligned with images)
    cbar_ax1 = fig.add_subplot(gs[0, num_images])  # Use GridSpec for colorbar too
    cbar1 = ColorbarBase(cbar_ax1, cmap=cm.gray, norm=norm1, orientation='vertical')
    cbar1.set_label(label1, fontsize=10)

    # Second row
    for i in range(num_images):
        ax = fig.add_subplot(gs[1, i])  # Use GridSpec positioning
        
        # Process image according to mode
        img_data, img_normalized = process_image(image_paths2[i], mode_row2)
        
        # Display normalized image
        ax.imshow(img_normalized, cmap='gray', vmin=0, vmax=1, aspect='equal')
        
        # Load and display bounding box
        bbox = load_bounding_box(label_paths2[i], *img_data.shape)
        if bbox:
            x, y, width, height, class_id = bbox
            rect = Rectangle((x, y), width, height, 
                            linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            
            # Add class label positioned off NW corner of BBox (exact same as visualise_crop)
            label_str = "is_fishing" if class_id == 1 else "is_vessel"
            label_x = x - 2   # 2 pixels left
            label_y = y
            ax.text(
                label_x, label_y, label_str,
                color='lime',
                fontsize=8,  # Slightly smaller for crop_compare grid layout
                verticalalignment='top',     # Top of label aligns with (x, y)
                horizontalalignment='right'  # Right end of label aligns with (x, y)
            )
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add row label
        if i == 0:
            row_label = f"Set 2 ({label2})" if not single_list_mode else label2
            ax.text(-0.15, 0.5, row_label, transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)

    # Add vertical colorbar for second row (aligned with images)
    cbar_ax2 = fig.add_subplot(gs[1, num_images])  # Use GridSpec for colorbar too
    cbar2 = ColorbarBase(cbar_ax2, cmap=cm.gray, norm=norm2, orientation='vertical')
    cbar2.set_label(label2, fontsize=10)

    # No tight_layout needed with GridSpec - spacing is controlled by GridSpec parameters
    plt.show()
    
    print(f"Crop comparison complete!")
    if single_list_mode:
        print(f"Single list mode: Top row = {mode_row1}, Bottom row = {mode_row2}")
    else:
        print(f"Two list mode: Top row = Set 1 ({mode_row1}), Bottom row = Set 2 ({mode_row2})")
    print(f"Normalization parameters - Magnitude: [{amp_min}, {amp_max}], Log-Magnitude: [{log_min}, {log_max}] dB, Real: [{real_min}, {real_max}]")
    print(f"All images normalized to [0, 1] range using above parameters")

