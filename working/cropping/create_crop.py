#!/usr/bin/env python3

"""
create_crop.py

Author: Peter Millitz
Created: 2025-06-24

This script extracts image crops of a given size from raw 2D SAR image arrays based on vessel
detection annotations and outputs them as NumPy arrays (.npy) with the same shape, along with
YOLO-style (.txt) label files. Accommodates duplicate detections in swath overlap zones and
uses a padding strategy for edge cases where the detection centre is located close to an image
boundary, ensuring detection centres are always centred in crops. Zero padding is used as the
default value for complex64 SAR data. The crop size and label confidence level is specified
via the cropping.yaml file. A summary CSV file listing the total number of crops created from
each scene is also output along with comprehensive logging of padding and boundary box issues.
"""
#!/usr/bin/env python3

"""
create_crop.py - Padding Version

Author: Peter Millitz
Created: 2025-06-17
Modified: 2025-06-24 - Added padding strategy for boundary handling

This script extracts image crops of a given size from raw 2D SAR image arrays
based on vessel detection annotations and outputs them as NumPy arrays (.npy)
with the same shape, along with YOLO-style (.txt) label files. The crop size
and label confidence level is specified via the cropping.yaml file. A summary
CSV file listing the total number of crops created from each scene, is also
output. Accommodates duplicate detections in swath overlap zones.

Padding Version Changes:
- Replaced shifting strategy with padding strategy for boundary handling
- Detection centers are now always centered in crops
- Added comprehensive logging for padding and boundary box issues
- Uses zero padding (pad_value=0) for complex64 SAR data

"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

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

def process_crop(ann, img_array, crop_size, out_img_dir, out_lbl_dir, swath_idx, saved_filenames=None, quiet_mode=False):
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
        quiet_mode (bool): if True, suppress detailed output messages
    
    Returns:
        dict: {'success': bool, 'padded': bool, 'bbox_clipped': bool} - processing results
    """
    # Initialize return values
    result = {'success': False, 'padded': False, 'bbox_clipped': False}
    
    detect_id = ann['detect_id']
    
    # Create unique filename incorporating swath information
    filename_base = f"{detect_id}_swath{swath_idx}"
    
    # Check for duplicate filenames (shouldn't happen with swath info, but safety check)
    if saved_filenames is not None:
        if filename_base in saved_filenames:
            if not quiet_mode:
                print(f"Warning: Duplicate filename found: {filename_base} - this shouldn't happen!")
            return result
        saved_filenames.add(filename_base)

    try:
        # Validate image shape
        if len(img_array.shape) != 2:
            if not quiet_mode:
                print(f"Error: Expected image shape (H, W), but got {img_array.shape} for {detect_id}")
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
            if not quiet_mode:
                print(f"Warning: Detection centre ({centre_row}, {centre_col}) outside image bounds ({H}, {W}) for {detect_id}")
            return result

        # Create padded crop with zero padding for complex64 SAR data
        crop, metadata = create_padded_crop(img_array, centre_row, centre_col, crop_size, pad_value=0)
        
        # Log padding information and track statistics
        if metadata['padding_applied']:
            pad_top, pad_bottom, pad_left, pad_right = metadata['padding_amounts']
            if not quiet_mode:
                print(f"  Warning: Padding applied to {detect_id}")
            result['padded'] = True
        
        # Validate crop size
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            if not quiet_mode:
                print(f"Error: Crop size mismatch. Expected {crop_size}x{crop_size}, got {crop.shape[0]}x{crop.shape[1]} for {detect_id}")
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
        
        # Check if bounding box extends outside crop (this should be rare)
        bbox_outside_crop = (box_left_in_crop < 0 or box_top_in_crop < 0 or 
                           box_right_in_crop > crop_size or box_bottom_in_crop > crop_size)
        
        if bbox_outside_crop:
            if not quiet_mode:
                print(f"  Warning: Bounding box extends outside crop for {detect_id}")
            result['bbox_clipped'] = True
            # Continue processing but flag this case
            
        # Clamp bounding box to crop boundaries for YOLO format calculation
        box_left_in_crop = max(0, min(crop_size, box_left_in_crop))
        box_right_in_crop = max(0, min(crop_size, box_right_in_crop))
        box_top_in_crop = max(0, min(crop_size, box_top_in_crop))
        box_bottom_in_crop = max(0, min(crop_size, box_bottom_in_crop))
            
        # Convert to YOLO format (normalized xc, yc, w, h)
        xc = (box_left_in_crop + box_right_in_crop) / 2 / crop_size
        yc = (box_top_in_crop + box_bottom_in_crop) / 2 / crop_size
        w = (box_right_in_crop - box_left_in_crop) / crop_size
        h = (box_bottom_in_crop - box_top_in_crop) / crop_size

        # Additional validation for YOLO format
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            if not quiet_mode:
                print(f"Warning: YOLO coordinates out of range for {detect_id}: xc={xc:.3f}, yc={yc:.3f}, w={w:.3f}, h={h:.3f}")

        # Save .npy crop and .txt label using unique filename
        image_path = out_img_dir / f"{filename_base}.npy"
        label_path = out_lbl_dir / f"{filename_base}.txt"
        
        try:
            np.save(image_path, crop)
            with open(label_path, "w") as f:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            
            # Verify files were actually created
            if not image_path.exists():
                if not quiet_mode:
                    print(f"Error: Image file was not created: {image_path}")
                return result
            if not label_path.exists():
                if not quiet_mode:
                    print(f"Error: Label file was not created: {label_path}")
                return result
                
            status_flags = []
            if result['padded']:
                status_flags.append("PADDED")
            if result['bbox_clipped']:
                status_flags.append("BBOX_CLIPPED")
            
            status_str = f" [{', '.join(status_flags)}]" if status_flags else ""
            if not quiet_mode:
                print(f"  Saved: {filename_base}{status_str}")
            result['success'] = True
            return result
            
        except Exception as save_error:
            if not quiet_mode:
                print(f"Error saving files for {filename_base}: {save_error}")
            return result

    except FileNotFoundError as e:
        if not quiet_mode:
            print(f"File not found for {detect_id}: {e}")
        return result
    except ValueError as e:
        if not quiet_mode:
            print(f"Data validation error for {detect_id}: {e}")
        return result
    except KeyError as e:
        if not quiet_mode:
            print(f"Missing required data field for {detect_id}: {e}")
        return result
    except Exception as e:
        if not quiet_mode:
            import traceback
            print(f"Unexpected error processing {detect_id}: {e}")
            traceback.print_exc()
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

def main():
    """
    Main driver function to extract crops and labels for vessel detection from SAR SLC scenes.

    Steps:
    1. Load paths and settings from cropping.yaml
    2. Create output directories for image and label crops
    3. Load correspondence and annotation files, join them on scene_id and SLC_product_identifier
    4. Scan arrays_path directory for all .npy files
    5. For each .npy file:
       a. Find matching entry in correspondence file by filename stem
       b. Filter annotations for that scene by confidence and vessel criteria
       c. Process each valid annotation to create crops named by detect_id
    6. Generate a summary CSV of the number of crops created per scene
    -----------------------------------------------------------------------
    Expected configuration keys in cropping.yaml:
    
    xView3_SLC_GRD_correspondences_path: str
        Path to correspondence CSV file mapping scene IDs to SLC images
    annotations_path: str
        Path to annotations CSV file mapping detection labels to SLC images
    arrays_path: str
        Path to directory containing .npy array files to process
    CREATE_CROP:
      CropPath: str
          Output directory where cropped images and labels are saved
      CropSize: int
          Crop size (e.g., 64 for 64 x 64 crops)
      LabelConfidence: list[str]
          Confidence levels to include (e.g., ["HIGH", "MEDIUM"])
      QuietMode: bool (optional)
          If true, suppress detailed processing output (default: false)
    -----------------------------------------------------------------------
    """
    # Load configuration from YAML
    try:
        with open("/home/peterm/UWA/CITS5014/SARFish/working/cropping/cropping.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: cropping.yaml configuration file not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML configuration file: {e}")
        return

    # Resolve paths and parameters from config
    correspondence_path = Path(config["xView3_SLC_GRD_correspondences_path"])
    annotation_path = Path(config["annotations_path"])
    arrays_path = Path(config["arrays_path"])
    crop_path = Path(config["CREATE_CROP"]["CropPath"])
    crop_size = int(config["CREATE_CROP"]["CropSize"])
    confidence_levels = config["CREATE_CROP"]["LabelConfidence"]
    quiet_mode = config["CREATE_CROP"].get("QuietMode", False)  # Default to False if not specified
    
    # Ensure confidence_levels is a list
    if isinstance(confidence_levels, str):
        confidence_levels = [confidence_levels]

    # Check if required files and directories exist
    if not correspondence_path.exists():
        if not quiet_mode:
            print(f"Correspondence file not found: {correspondence_path}")
        return
    
    if not annotation_path.exists():
        if not quiet_mode:
            print(f"Annotation file not found: {annotation_path}")
        return
        
    if not arrays_path.exists():
        if not quiet_mode:
            print(f"Arrays directory not found: {arrays_path}")
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
        if not quiet_mode:
            print(f"Error loading CSV files: {e}")
        return

    # Join correspondence and annotations on scene_id and SLC_product_identifier
    try:
        joined_df = pd.merge(
            annotations, corr_df, 
            on=['scene_id', 'SLC_product_identifier'], 
            how='inner'
        )
    except Exception as e:
        if not quiet_mode:
            print(f"Error joining correspondence and annotation files: {e}")
        return

    # Filter annotations based on criteria
    filtered_annotations = joined_df[
        (joined_df["is_vessel"] == True) &
        (joined_df["is_fishing"].isin([True, False])) &
        (joined_df["confidence"].isin(confidence_levels))
    ].dropna(subset=["top", "left", "bottom", "right"])

    if filtered_annotations.empty:
        if not quiet_mode:
            print("No annotations found matching the specified criteria.")
        return

    # Get all .npy files in arrays directory
    npy_files = list(arrays_path.glob("*.npy"))
    if not npy_files:
        if not quiet_mode:
            print(f"No .npy files found in {arrays_path}")
        return

    summary = []  # to collect counts of saved crops per scene
    total_processed = 0
    total_padded = 0
    total_bbox_clipped = 0
    images_with_no_crops = 0
    saved_filenames = set()  # Track all filenames to catch duplicates

    print(f"Processing {len(npy_files)} array files...")
    if not quiet_mode:
        print(f"Using zero padding (pad_value=0) for complex64 SAR data")

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
            if not quiet_mode:
                print(f"Warning: No correspondence entry found for array file: {npy_file.name}")
            continue
            
        if len(matching_rows) > 1:
            if not quiet_mode:
                print(f"Warning: Multiple correspondence entries found for array file: {npy_file.name}")
            continue
            
        corr_row, swath_idx = matching_rows[0]
        scene_id = corr_row['scene_id']
        slc_product_id = corr_row['SLC_product_identifier']
        
        # Load the array
        try:
            img_array = np.load(npy_file)
            if not quiet_mode:
                print(f"Processing {npy_file.name} (scene: {scene_id}, swath: {swath_idx})")
        except Exception as e:
            if not quiet_mode:
                print(f"Error loading array file {npy_file}: {e}")
            continue

        # Get annotations for this specific scene and swath
        scene_annotations = filtered_annotations[
            (filtered_annotations["scene_id"] == scene_id) &
            (filtered_annotations["SLC_product_identifier"] == slc_product_id) &
            (filtered_annotations["swath_index"] == swath_idx)
        ]

        scene_crop_count = 0
        scene_padded_count = 0
        scene_bbox_clipped_count = 0
        scene_padding_details = []
        scene_clipping_details = []
        
        for _, ann in scene_annotations.iterrows():
            result = process_crop(ann, img_array, crop_size, out_img_dir, out_lbl_dir, swath_idx, saved_filenames, quiet_mode)
            if result['success']:
                scene_crop_count += 1
                total_processed += 1
                
                if result['padded']:
                    scene_padded_count += 1
                    total_padded += 1
                    # Get padding details from the annotation processing
                    crop, metadata = create_padded_crop(img_array, 
                                                      int(ann.get("detect_scene_row", (ann["top"] + ann["bottom"]) / 2)),
                                                      int(ann.get("detect_scene_column", (ann["left"] + ann["right"]) / 2)),
                                                      crop_size, pad_value=0)
                    pad_top, pad_bottom, pad_left, pad_right = metadata['padding_amounts']
                    scene_padding_details.append(f"{ann['detect_id']}: top={pad_top},bottom={pad_bottom},left={pad_left},right={pad_right}")
                
                if result['bbox_clipped']:
                    scene_bbox_clipped_count += 1
                    total_bbox_clipped += 1
                    # Get clipping details
                    centre_row = int(ann.get("detect_scene_row", (ann["top"] + ann["bottom"]) / 2))
                    centre_col = int(ann.get("detect_scene_column", (ann["left"] + ann["right"]) / 2))
                    scene_clipping_details.append(f"{ann['detect_id']}: center=({centre_row},{centre_col}),bbox=({ann['left']},{ann['top']},{ann['right']},{ann['bottom']})")

        if scene_crop_count > 0:
            summary.append({
                "scene_id": scene_id, 
                "array_file": npy_file.name, 
                "num_crops": scene_crop_count,
                "num_padded": scene_padded_count,
                "num_bbox_clipped": scene_bbox_clipped_count,
                "padding_details": "; ".join(scene_padding_details) if scene_padding_details else "",
                "clipping_details": "; ".join(scene_clipping_details) if scene_clipping_details else ""
            })
            if not quiet_mode:
                print(f"  -> Successfully created {scene_crop_count} crops")
        else:
            images_with_no_crops += 1
            if not quiet_mode:
                print(f"  -> No crops created (all failed validation)")

    # Final verification
    actual_image_count = len(list(out_img_dir.glob("*.npy")))
    actual_label_count = len(list(out_lbl_dir.glob("*.txt")))
    
    print(f"\n" + "="*60)
    print(f"PROCESSING SUMMARY")
    print(f"="*60)
    print(f"Number of input images processed: {len(npy_files)}")
    print(f"Total crops of size {crop_size} x {crop_size} created: {total_processed}")
    print(f"Images with no crops created: {images_with_no_crops}")
    #print(f"Images with crops created: {len(npy_files) - images_with_no_crops}")
    print(f"Crops with padding applied: {total_padded}")
    print(f"Crops with bounding box clipping: {total_bbox_clipped}")
    print(f"Actual image files written: {actual_image_count}")
    print(f"Actual label files written: {actual_label_count}")
    #print(f"Unique filenames processed: {len(saved_filenames)}")
    print(f"Padding strategy: Zero padding (pad_value=0) for complex64 SAR data")
    
    if total_processed != actual_image_count:
        print(f"MISMATCH: Expected {total_processed} images but found {actual_image_count}")
    if actual_image_count != actual_label_count:
        print(f"MISMATCH: Image count ({actual_image_count}) != Label count ({actual_label_count})")

    # Save summary CSV
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(crop_path / "crop_summary.csv", index=False)
        print(f"Crop summary saved to: {crop_path / 'crop_summary.csv'}")
        #print(f"Total crops processed: {total_processed}")
    else:
        print("No crops were created.")

if __name__ == "__main__":
    main()

