#!/usr/bin/env python3

"""
create_crop_v2.py

Author: Peter Millitz
Created: 2025-07-12
Modified: 2025-07-15 - Added support for spatial indexing module

This script extracts image crops of a given size from raw 2D SAR image arrays based on vessel
detection annotations and outputs them as NumPy arrays (.npy) with the same shape, along with
YOLO-style (.txt) label files. Accommodates duplicate detections in swath overlap zones and
uses a padding strategy for edge cases where a detection centre is close to an image boundary.
Zero padding is used as the default value for complex64 SAR data. The crop size, label
confidence level and other parameters are specified via the config.yaml file. A summary CSV
file listing processing statisitcs for each input image, is also output.

NEW in v2: Supports spatial indexing for efficient multi-object crops when processing
large crop sizes (e.g., 1024x1024). Automatically detects and uses spatial_processor module
when available and appropriate.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys
from datetime import datetime

# NEW: Try to import spatial processing module
try:
    from spatial_processor import SpatialCropProcessor, process_scene_with_spatial_indexing
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False
    print("Warning: spatial_processor module not found. Spatial indexing will be disabled.")

class Logger:
    """Logger class to handle both file and console output based on quiet mode."""
    
    def __init__(self, log_file_path, quiet_mode=False):
        self.log_file_path = log_file_path
        self.quiet_mode = quiet_mode
        
        # Open log file for writing
        try:
            self.log_file = open(log_file_path, 'w', encoding='utf-8')
            # Write header to log file
            self.log_file.write(f"Crop Processing Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("="*80 + "\n\n")
            self.log_file.flush()
        except Exception as e:
            print(f"Warning: Could not create log file {log_file_path}: {e}")
            self.log_file = None
    
    def print(self, message, force_screen=False):
        """
        Print message according to quiet mode settings and always log to file.
        
        Args:
            message (str): Message to print/log
            force_screen (bool): If True, always print to screen regardless of quiet mode
        """
        # Always write to log file if available
        if self.log_file:
            try:
                self.log_file.write(message + "\n")
                self.log_file.flush()
            except Exception:
                pass  # Silently continue if logging fails
        
        # Print to screen based on quiet mode or force_screen
        if not self.quiet_mode or force_screen:
            print(message)
    
    def close(self):
        """Close the log file."""
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass

def extract_crop_coords_with_padding(centre_row, centre_col, crop_size, img_height, img_width):
    """
    Compute the bounding box for a square crop centred on (centre_row, centre_col),
    using padding when necessary to maintain centring and crop size.
    
    Returns:
        (top, bottom, left, right): int crop boundaries for the image
        (pad_top, pad_bottom, pad_left, pad_right): int padding amounts needed
    """
    half = crop_size // 2
    
    # Desired crop boundaries (may extend outside image)
    desired_top = centre_row - half
    desired_bottom = centre_row + half + (crop_size % 2)  # Handle odd crop sizes
    desired_left = centre_col - half
    desired_right = centre_col + half + (crop_size % 2)
    
    # Actual image boundaries we can extract from
    actual_top = max(0, desired_top)
    actual_bottom = min(img_height, desired_bottom)
    actual_left = max(0, desired_left)
    actual_right = min(img_width, desired_right)
    
    # Calculate padding needed
    pad_top = max(0, -desired_top)
    pad_bottom = max(0, desired_bottom - img_height)
    pad_left = max(0, -desired_left)
    pad_right = max(0, desired_right - img_width)
    
    return (actual_top, actual_bottom, actual_left, actual_right), (pad_top, pad_bottom, pad_left, pad_right)

def create_padded_crop(img_array, centre_row, centre_col, crop_size, pad_value=0):
    """
    Extract a crop centred on the detection, using padding when necessary.
    
    Args:
        img_array (np.ndarray): Input image array
        centre_row, centre_col (int): Detection centre coordinates
        crop_size (int): Desired crop size (square)
        pad_value (complex or float): Value to use for padding (default: 0)
    
    Returns:
        np.ndarray: Cropped and potentially padded image of exact size (crop_size, crop_size)
        dict: Metadata about the cropping operation
    """
    img_height, img_width = img_array.shape
    
    # Get crop coordinates and padding requirements
    (actual_top, actual_bottom, actual_left, actual_right), \
    (pad_top, pad_bottom, pad_left, pad_right) = extract_crop_coords_with_padding(
        centre_row, centre_col, crop_size, img_height, img_width
    )
    
    # Extract the available portion from the image
    image_crop = img_array[actual_top:actual_bottom, actual_left:actual_right]
    
    # Apply padding if needed
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        padded_crop = np.pad(image_crop, 
                           ((pad_top, pad_bottom), (pad_left, pad_right)), 
                           mode='constant', 
                           constant_values=pad_value)
    else:
        padded_crop = image_crop
    
    # Create metadata about the operation
    metadata = {
        'padding_applied': any([pad_top, pad_bottom, pad_left, pad_right]),
        'padding_amounts': (pad_top, pad_bottom, pad_left, pad_right),
        'actual_image_region': (actual_top, actual_bottom, actual_left, actual_right),
        'centre_offset_in_crop': (crop_size // 2, crop_size // 2),  # Always centred now
        'original_centre': (centre_row, centre_col)
    }
    
    return padded_crop, metadata

def process_crop(ann, img_array, crop_size, out_img_dir, out_lbl_dir, swath_idx, saved_filenames=None, logger=None):
    """
    Process a single vessel annotation: create a cropped image (of size {crop_size}) centred on
    supplied vessel detection coordinates and save it along with a YOLO-style bounding box label.
    Uses padding strategy to ensure detection centres remain centred in crops.

    Args:
        ann (pd.Series): annotation row
        img_array (np.ndarray): full image array with shape (H, W)
        crop_size (int): crop dimensions (assumes square)
        out_img_dir (Path): output folder for .npy crops
        out_lbl_dir (Path): output folder for YOLO label files
        swath_idx (int): swath index (1, 2, or 3)
        saved_filenames (set): set to track filenames for duplicate detection
        logger (Logger): logger instance for output
    
    Returns:
        dict: {'success': bool, 'padded': bool, 'skipped': bool} - processing results
    """
    # Initialize return values
    result = {'success': False, 'padded': False, 'skipped': False, 'shrunk': False}
    
    detect_id = ann['detect_id']
    
    # Create unique filename incorporating swath information
    filename_base = f"{detect_id}_swath{swath_idx}"
    
    # Check for duplicate filenames (shouldn't happen with swath info, but safety check)
    if saved_filenames is not None:
        if filename_base in saved_filenames:
            if logger:
                logger.print(f"Warning: Duplicate filename found: {filename_base} - this shouldn't happen!")
            return result
        saved_filenames.add(filename_base)

    try:
        # Validate image shape
        if len(img_array.shape) != 2:
            if logger:
                logger.print(f"Error: Expected image shape (H, W), but got {img_array.shape} for {detect_id}")
            return result
            
        H, W = img_array.shape

        # Use provided detection centre if available, else fallback to bounding box centre
        if "detect_scene_row" in ann and "detect_scene_column" in ann:
            centre_row = int(ann["detect_scene_row"])
            centre_col = int(ann["detect_scene_column"])
        else:
            centre_row = int((ann["top"] + ann["bottom"]) / 2)
            centre_col = int((ann["left"] + ann["right"]) / 2)

        # Validate detection coordinates are within image bounds
        if not (0 <= centre_row < H and 0 <= centre_col < W):
            if logger:
                logger.print(f"Warning: Detection centre ({centre_row}, {centre_col}) outside image bounds ({H}, {W}) for {detect_id}")
            return result

        # Create padded crop with zero padding for complex64 SAR data
        crop, metadata = create_padded_crop(img_array, centre_row, centre_col, crop_size, pad_value=0)
        
        # Log padding information and track statistics
        if metadata['padding_applied']:
            pad_top, pad_bottom, pad_left, pad_right = metadata['padding_amounts']
            if logger:
                logger.print(f"Warning: Padding applied to {detect_id}")
            result['padded'] = True
        
        # Validate crop size
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            if logger:
                logger.print(f"Error: Crop size mismatch. Expected {crop_size}x{crop_size}, got {crop.shape[0]}x{crop.shape[1]} for {detect_id}")
            return result

        # Determine class: 0 = vessel, 1 = fishing vessel
        class_id = 1 if pd.notna(ann.get("is_fishing")) and ann["is_fishing"] is True else 0

        # Convert bounding box to YOLO format, accounting for the coordinate transformation
        # We need to transform the original bounding box coordinates to the crop coordinate system
        
        # Get the actual image region that was extracted
        actual_top, actual_bottom, actual_left, actual_right = metadata['actual_image_region']
        pad_top, pad_bottom, pad_left, pad_right = metadata['padding_amounts']
        
        # Transform original bounding box coordinates to crop coordinates
        # Remember: annotations use inverted Y-axis (top > bottom in original coordinates)
        box_left_in_crop = float(ann["left"]) - actual_left + pad_left
        box_right_in_crop = float(ann["right"]) - actual_left + pad_left
        box_top_in_crop = float(ann["bottom"]) - actual_top + pad_top  # inverted Y
        box_bottom_in_crop = float(ann["top"]) - actual_top + pad_top  # inverted Y
        
        # Check if bounding box extends outside crop
        bbox_outside_crop = (box_left_in_crop < 0 or box_top_in_crop < 0 or 
                           box_right_in_crop > crop_size or box_bottom_in_crop > crop_size)
        
        if bbox_outside_crop:
            # Try to shrink the bounding box by up to 5 pixels on each problematic edge
            shrunk_left = max(0, box_left_in_crop)
            shrunk_top = max(0, box_top_in_crop)
            shrunk_right = min(crop_size, box_right_in_crop)
            shrunk_bottom = min(crop_size, box_bottom_in_crop)
            
            # Check if shrinking was within the 5-pixel limit on each edge
            left_shrink = box_left_in_crop - shrunk_left
            top_shrink = box_top_in_crop - shrunk_top
            right_shrink = shrunk_right - box_right_in_crop
            bottom_shrink = shrunk_bottom - box_bottom_in_crop
            
            max_shrink_exceeded = (abs(left_shrink) > 5 or abs(top_shrink) > 5 or 
                                 abs(right_shrink) > 5 or abs(bottom_shrink) > 5)
            
            if max_shrink_exceeded:
                if logger:
                    logger.print(f"Warning: Skipping {detect_id} - bounding box too large for crop")
                result['skipped'] = True
                return result
            
            # Bounding box was successfully shrunk within limits
            if logger:
                logger.print(f"Warning: Bounding box shrunk for {detect_id} (within 5 pixels/edge)")
            result['shrunk'] = True
            
            # Use the shrunk bounding box
            box_left_in_crop = shrunk_left
            box_top_in_crop = shrunk_top
            box_right_in_crop = shrunk_right
            box_bottom_in_crop = shrunk_bottom
            
        # Convert to YOLO format (normalized xc, yc, w, h)
        xc = (box_left_in_crop + box_right_in_crop) / 2 / crop_size
        yc = (box_top_in_crop + box_bottom_in_crop) / 2 / crop_size
        w = (box_right_in_crop - box_left_in_crop) / crop_size
        h = (box_bottom_in_crop - box_top_in_crop) / crop_size

        # Additional validation for YOLO format
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            if logger:
                logger.print(f"Warning: YOLO coordinates out of range for {detect_id}: xc={xc:.3f}, yc={yc:.3f}, w={w:.3f}, h={h:.3f}")

        # Save .npy crop and .txt label using unique filename
        image_path = out_img_dir / f"{filename_base}.npy"
        label_path = out_lbl_dir / f"{filename_base}.txt"
        
        try:
            np.save(image_path, crop)
            with open(label_path, "w") as f:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            
            # Verify files were actually created
            if not image_path.exists():
                if logger:
                    logger.print(f"Error: Image file was not created: {image_path}")
                return result
            if not label_path.exists():
                if logger:
                    logger.print(f"Error: Label file was not created: {label_path}")
                return result
                
            status_flags = []
            if result['padded']:
                status_flags.append("PADDED")
            if result['shrunk']:
                status_flags.append("SHRUNK")
            
            status_str = f" [{', '.join(status_flags)}]" if status_flags else ""
            if logger:
                logger.print(f"  Saved: {filename_base}{status_str}")
            result['success'] = True
            return result
            
        except Exception as save_error:
            if logger:
                logger.print(f"Error saving files for {filename_base}: {save_error}")
            return result

    except FileNotFoundError as e:
        if logger:
            logger.print(f"File not found for {detect_id}: {e}")
        return result
    except ValueError as e:
        if logger:
            logger.print(f"Data validation error for {detect_id}: {e}")
        return result
    except KeyError as e:
        if logger:
            logger.print(f"Missing required data field for {detect_id}: {e}")
        return result
    except Exception as e:
        if logger:
            logger.print(f"Unexpected error processing {detect_id}: {e}")
            import traceback
            logger.print(traceback.format_exc())
        return result

def find_matching_array(filename_stem, correspondence_row):
    """
    Check if filename stem matches any of the swath columns in correspondence row.
    
    Args:
        filename_stem (str): stem of the .npy filename
        correspondence_row (pd.Series): row from correspondence file
        
    Returns:
        int or None: swath index (1, 2, 3) if match found, None otherwise
    """
    for swath_idx in [1, 2, 3]:
        swath_col = f"SLC_swath_{swath_idx}_vh"
        if pd.notna(correspondence_row.get(swath_col)):
            # Extract stem from the correspondence entry (remove .tiff extension)
            corr_stem = Path(correspondence_row[swath_col]).stem
            if filename_stem == corr_stem:
                return swath_idx
    return None

def main(config_file="config.yaml", base_dir=".", data_split="train"):
    """
    Main driver function to extract crops and labels for vessel detection from SAR SLC scenes.
    
    NEW in v2: Supports spatial indexing for efficient multi-object crops when processing
    large crop sizes. Automatically detects and uses spatial_processor module when available.

    Args:
        config_file (str): Path to configuration YAML file
        base_dir (str): Base directory for input/output paths
        data_split (str): Data split to process ('train', 'val', or 'test')

    Steps:
    1. Load paths and settings from config.yaml
    2. Create output directories for image and label crops
    3. Load correspondence and annotation files, join them on scene_id and SLC_product_identifier
    4. Scan arrays_path directory for all .npy files
    5. For each .npy file:
       a. Find matching entry in correspondence file by filename stem
       b. Filter annotations for that scene by confidence and vessel criteria
       c. Process annotations using either:
          - Spatial indexing (for large crops) - creates fewer crops with multiple annotations
          - Original method - one crop per annotation
    6. Generate a summary CSV of the number of crops created per scene
    -----------------------------------------------------------------------
    Expected configuration keys in config.yaml:
    
    correspondences_path: str
        Path to correspondence CSV file mapping scene IDs to SLC images
    annotations_path: str
        Path to annotations CSV file mapping detection labels to SLC images
    data_paths:
        arrays_path: dict
            Dict with train/val/test keys pointing to directories containing .npy files
    crop_path: dict
        Dictionary with train/val/test keys pointing to output directories for crops
    crop_size: int
        Crop size (e.g., 64 for 64 x 64 crops)
    label_confidence: list[str]
        Confidence levels to include (e.g., ["HIGH", "MEDIUM"])
    quiet_mode: bool (optional)
        If True, suppress detailed processing output (default: False)
    use_spatial_indexing: bool (optional)
        If True, use spatial indexing for multi-object crops (default: auto for large crops)
    min_crop_distance: int (optional)
        Minimum distance between crop centers when using spatial indexing (default: crop_size/2)
    -----------------------------------------------------------------------
    """
    # Validate data_split parameter
    if data_split not in ['train', 'val', 'test']:
        print(f"Error: data_split must be 'train', 'val', or 'test', got '{data_split}'")
        return
    
    # Initialize logger with data_split in filename
    log_filename = f"crops_{data_split}_log.txt"
    logger = Logger(log_filename, quiet_mode=False)  # Will be updated with actual quiet_mode
    
    # Load configuration from YAML
    try:
        with open(config_file, "r") as f:
            uni_config = yaml.safe_load(f)
            config = uni_config['create_crop']

    except FileNotFoundError:
        logger.print("Error: config.yaml configuration file not found.", force_screen=True)
        logger.close()
        return
    except yaml.YAMLError as e:
        logger.print(f"Error: Invalid YAML configuration file: {e}", force_screen=True)
        logger.close()
        return
    except KeyError as e:
        logger.print(f"Error: Missing 'create_crop' section in config file: {e}", force_screen=True)
        logger.close()
        return

    # Resolve paths and parameters from config
    base_path = Path(base_dir)
    correspondence_path = Path(config["correspondences_path"])
    annotation_path = Path(config["annotations_path"])
    
    # Get data_split-specific paths
    try:
        arrays_path = base_path / uni_config["data_paths"]["arrays_path"][data_split]
        crop_path = base_path / config["crop_path"][data_split]
    except KeyError as e:
        logger.print(f"Error: Missing {data_split} configuration in arrays_path or crop_path: {e}", force_screen=True)
        logger.close()
        return
    
    crop_size = int(config["crop_size"])
    confidence_levels = config["label_confidence"]
    quiet_mode = config.get("quiet_mode", False) # Default to False if not specified
    
    # NEW: Spatial indexing parameters
    use_spatial_indexing = config.get("use_spatial_indexing", False)
    if use_spatial_indexing and not SPATIAL_AVAILABLE:
        logger.print("Warning: Spatial indexing requested but module not available. " 
                    "Falling back to original processing.", force_screen=True)
        use_spatial_indexing = False
    
    min_crop_distance = config.get("min_crop_distance", crop_size // 2)
    
    # Auto-enable spatial indexing for large crops if not explicitly set
    if "use_spatial_indexing" not in config and crop_size >= 256:
        use_spatial_indexing = SPATIAL_AVAILABLE
        if use_spatial_indexing:
            logger.print(f"Auto-enabling spatial indexing for crop_size={crop_size}", 
                        force_screen=True)
    
    # Update logger with correct quiet_mode setting
    logger.quiet_mode = quiet_mode
    
    # Ensure confidence_levels is a list
    if isinstance(confidence_levels, str):
        confidence_levels = [confidence_levels]

    # Log processing information
    logger.print(f"Processing {data_split.upper()} split", force_screen=True)
    logger.print(f"Input arrays path: {arrays_path}", force_screen=True)
    logger.print(f"Output crops path: {crop_path}", force_screen=True)
    
    # NEW: Log processing method
    if use_spatial_indexing:
        logger.print(f"Using SPATIAL INDEXING with min_crop_distance={min_crop_distance}", 
                    force_screen=True)
    else:
        logger.print("Using ORIGINAL single-annotation processing", force_screen=True)

    # Check if required files and directories exist
    if not correspondence_path.exists():
        logger.print(f"Correspondence file not found: {correspondence_path}", force_screen=True)
        logger.close()
        return
    
    if not annotation_path.exists():
        logger.print(f"Annotation file not found: {annotation_path}", force_screen=True)
        logger.close()
        return
        
    if not arrays_path.exists():
        logger.print(f"Arrays directory not found: {arrays_path}", force_screen=True)
        logger.close()
        return

    # Create output subfolders for images and labels
    out_img_dir = crop_path / "images"
    out_lbl_dir = crop_path / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Load correspondence and annotation files
    try:
        corr_df = pd.read_csv(correspondence_path)
        annotations = pd.read_csv(annotation_path)
    except Exception as e:
        logger.print(f"Error loading CSV files: {e}", force_screen=True)
        logger.close()
        return

    # Join correspondence and annotations on scene_id and SLC_product_identifier
    try:
        joined_df = pd.merge(
            annotations, corr_df, 
            on=['scene_id', 'SLC_product_identifier'], 
            how='inner'
        )
    except Exception as e:
        logger.print(f"Error joining correspondence and annotation files: {e}", force_screen=True)
        logger.close()
        return

    # Filter annotations based on criteria
    filtered_annotations = joined_df[
        (joined_df["is_vessel"] == True) &
        (joined_df["is_fishing"].isin([True, False])) &
        (joined_df["confidence"].isin(confidence_levels))
    ].dropna(subset=["top", "left", "bottom", "right"])

    if filtered_annotations.empty:
        logger.print("No annotations found matching the specified criteria.", force_screen=True)
        logger.close()
        return

    # NEW: Initialize spatial processor if enabled
    spatial_processor = None
    if use_spatial_indexing:
        spatial_processor = SpatialCropProcessor(crop_size, min_crop_distance)

    # Get all .npy files in arrays directory
    npy_files = list(arrays_path.glob("*.npy"))
    if not npy_files:
        logger.print(f"No .npy files found in {arrays_path}", force_screen=True)
        logger.close()
        return

    summary = []  # to collect counts of saved crops per scene
    total_processed = 0
    total_padded = 0
    total_shrunk = 0
    total_skipped = 0
    images_with_no_crops = 0
    saved_filenames = set()  # Track all filenames to catch duplicates

    logger.print(f"Processing {len(npy_files)} array files...", force_screen=True)
    logger.print(f"Using zero padding (pad_value=0) for complex64 SAR data")

    # Process each .npy file
    for npy_file in npy_files:
        filename_stem = npy_file.stem
        
        # Find matching correspondence entry
        matching_rows = []
        for _, corr_row in corr_df.iterrows():
            swath_idx = find_matching_array(filename_stem, corr_row)
            if swath_idx is not None:
                matching_rows.append((corr_row, swath_idx))
        
        if not matching_rows:
            logger.print(f"Warning: No correspondence entry found for array file: {npy_file.name}")
            continue
            
        if len(matching_rows) > 1:
            logger.print(f"Warning: Multiple correspondence entries found for array file: {npy_file.name}")
            continue
            
        corr_row, swath_idx = matching_rows[0]
        scene_id = corr_row['scene_id']
        slc_product_id = corr_row['SLC_product_identifier']
        
        # Load the array
        try:
            img_array = np.load(npy_file)
            logger.print(f"Processing {npy_file.name} (scene: {scene_id}, swath: {swath_idx})")
        except Exception as e:
            logger.print(f"Error loading array file {npy_file}: {e}")
            continue

        # Get annotations for this specific scene and swath
        scene_annotations = filtered_annotations[
            (filtered_annotations["scene_id"] == scene_id) &
            (filtered_annotations["SLC_product_identifier"] == slc_product_id) &
            (filtered_annotations["swath_index"] == swath_idx)
        ]

        if scene_annotations.empty:
            logger.print(f"  -> No annotations found")
            continue
            
        logger.print(f"  Found {len(scene_annotations)} annotations")
        
        # NEW: Choose processing method based on configuration
        if use_spatial_indexing and spatial_processor is not None:
            # Use spatial indexing for efficient multi-object crops
            scene_stats = process_scene_with_spatial_indexing(
                scene_annotations=scene_annotations,
                img_array=img_array,
                crop_size=crop_size,
                out_img_dir=out_img_dir,
                out_lbl_dir=out_lbl_dir,
                swath_idx=swath_idx,
                saved_filenames=saved_filenames,
                spatial_processor=spatial_processor,
                logger=logger,
                create_padded_crop=create_padded_crop  # Pass our function
            )
            
            # Update summary
            summary.append({
                "scene_id": scene_id,
                "array_file": npy_file.name,
                "num_annotations": len(scene_annotations),
                "num_crops": scene_stats['crops_created'],
                "num_padded": scene_stats.get('crops_padded', 0),
                "num_shrunk": 0,  # Not tracked in spatial mode
                "num_skipped": scene_stats.get('annotations_skipped', 0),
                "avg_annotations_per_crop": (scene_stats['annotations_processed'] / 
                                            max(1, scene_stats['crops_created']))
            })
            
            total_processed += scene_stats['annotations_processed']
            total_padded += scene_stats.get('crops_padded', 0)
            total_skipped += scene_stats.get('annotations_skipped', 0)
            
            if scene_stats['crops_created'] == 0 and len(scene_annotations) > 0:
                images_with_no_crops += 1
            
        else:
            # Original processing method (one crop per annotation)
            scene_crop_count = 0
            scene_padded_count = 0
            scene_shrunk_count = 0
            scene_skipped_count = 0
            
            for _, ann in scene_annotations.iterrows():
                result = process_crop(ann, img_array, crop_size, out_img_dir, out_lbl_dir, swath_idx, saved_filenames, logger)
                if result['success']:
                    scene_crop_count += 1
                    total_processed += 1
                    if result['padded']:
                        scene_padded_count += 1
                        total_padded += 1
                    if result['shrunk']:
                        scene_shrunk_count += 1
                        total_shrunk += 1
                elif result['skipped']:
                    scene_skipped_count += 1
                    total_skipped += 1

            if scene_crop_count > 0 or scene_skipped_count > 0:
                summary.append({
                    "scene_id": scene_id, 
                    "array_file": npy_file.name, 
                    "num_crops": scene_crop_count,
                    "num_padded": scene_padded_count,
                    "num_shrunk": scene_shrunk_count,
                    "num_skipped": scene_skipped_count,
                    "avg_annotations_per_crop": 1.0  # Always 1 for original method
                })
                status_msg = f"Successfully created {scene_crop_count} crops"
                if scene_skipped_count > 0:
                    status_msg += f", skipped {scene_skipped_count} crops"
                logger.print(f"  -> {status_msg}")
            else:
                images_with_no_crops += 1
                logger.print(f"  -> No crops created (all failed validation)")

    # Final verification
    actual_image_count = len(list(out_img_dir.glob("*.npy")))
    actual_label_count = len(list(out_lbl_dir.glob("*.txt")))
    
    logger.print(f"\n" + "="*60, force_screen=True)
    logger.print(f"PROCESSING SUMMARY - {data_split.upper()} SPLIT", force_screen=True)
    logger.print(f"="*60, force_screen=True)
    logger.print(f"Number of input images processed: {len(npy_files)}", force_screen=True)
    
    # NEW: Enhanced summary for spatial indexing
    if use_spatial_indexing:
        logger.print(f"Method: Spatial Indexing (min_crop_distance={min_crop_distance})", 
                    force_screen=True)
        total_crops = sum(s.get('num_crops', 0) for s in summary)
        avg_ann_per_crop = total_processed / max(1, total_crops) if summary else 0
        logger.print(f"Total crops created: {total_crops}", force_screen=True)
        logger.print(f"Total annotations processed: {total_processed}", force_screen=True)
        logger.print(f"Average annotations per crop: {avg_ann_per_crop:.2f}", force_screen=True)
        
        # Calculate reduction percentage
        total_annotations = len(filtered_annotations)
        reduction_pct = (1 - total_crops/total_annotations) * 100 if total_annotations > 0 else 0
        logger.print(f"Crop reduction: {reduction_pct:.1f}% compared to single-annotation method", 
                    force_screen=True)
    else:
        logger.print(f"Method: Original (one crop per annotation)", force_screen=True)
        logger.print(f"Total crops of size {crop_size} x {crop_size} created: {total_processed}", force_screen=True)
    
    logger.print(f"Images with no crops created: {images_with_no_crops}", force_screen=True)
    logger.print(f"Crops with padding applied: {total_padded}", force_screen=True)
    logger.print(f"Crops with bounding box shrunk: {total_shrunk}", force_screen=True)
    logger.print(f"Crops skipped (bounding box exceeds crop boundary): {total_skipped}", force_screen=True)
    logger.print(f"Actual image files written: {actual_image_count}", force_screen=True)
    logger.print(f"Actual label files written: {actual_label_count}", force_screen=True)
    logger.print(f"Padding strategy: Zero padding (pad_value=0) for complex64 SAR data", force_screen=True)
     
    if use_spatial_indexing:
        # For spatial indexing, check if total crops match actual files
        total_crops = sum(s.get('num_crops', 0) for s in summary)
        if total_crops != actual_image_count:
            logger.print(f"MISMATCH: Expected {total_crops} images but found {actual_image_count}", force_screen=True)
    else:
        # For original method, check if processed count matches actual files
        if total_processed != actual_image_count:
            logger.print(f"MISMATCH: Expected {total_processed} images but found {actual_image_count}", force_screen=True)
    
    if actual_image_count != actual_label_count:
        logger.print(f"MISMATCH: Image count ({actual_image_count}) != Label count ({actual_label_count})", force_screen=True)

    # Save summary CSV with data_split in filename
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_file = crop_path / f"crop_summary_{data_split}.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.print(f"Crop summary saved to: {summary_file}", force_screen=True)
        
        # NEW: Add average annotations per crop to summary file for spatial indexing
        if use_spatial_indexing and 'avg_annotations_per_crop' in summary_df.columns:
            overall_avg = summary_df['avg_annotations_per_crop'].mean()
            logger.print(f"Overall average annotations per crop: {overall_avg:.2f}", force_screen=True)
    else:
        logger.print("No crops were created.", force_screen=True)

    # Close the logger
    logger.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create crops from SAR data (v2 with spatial indexing support)")
    parser.add_argument("--config", default="config.yaml", help="Configuration file (default: config.yaml)")
    parser.add_argument("--base-dir", required=True, help="Base directory for input/output paths")
    parser.add_argument("--data-split", choices=['train', 'val', 'test'], required=True, 
                       help="Data split to process (train, val, or test)")
    
    args = parser.parse_args()
    main(args.config, args.base_dir, args.data_split)

