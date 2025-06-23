#!/usr/bin/env python3

"""
create_crop.py

Author: Peter Millitz
Created: 2025-06-17

This script extracts image crops of a given size from raw 2D SAR image arrays
based on vessel detection annotations and outputs them as NumPy arrays (.npy)
with the same shape, along with YOLO-style (.txt) label files. The crop size
and label confidence level is specified via the cropping.yaml file. A summary
CSV file listing the total number of crops created from each scene, is also
output. Accommodates duplicate detections in swath overlap zones.

"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

def extract_crop_coords(centre_row, centre_col, crop_size, img_height, img_width):
    """
    Compute the bounding box for a square crop centred on (centre_row, centre_col),
    ensuring the crop stays within image bounds.
    
    Returns:
        (top, bottom, left, right): int crop boundaries
    """
    half = crop_size // 2
    top = max(0, centre_row - half)
    left = max(0, centre_col - half)
    bottom = min(img_height, top + crop_size)
    right = min(img_width, left + crop_size)

    # Ensure dimensions match requested crop_size
    if bottom - top < crop_size:
        top = max(0, bottom - crop_size)
    if right - left < crop_size:
        left = max(0, right - crop_size)

    return top, bottom, left, right

def process_crop(ann, img_array, crop_size, out_img_dir, out_lbl_dir, swath_idx, saved_filenames=None):
    """
    Process a single vessel annotation: create a cropped image (of size {crop_size}) centred on
    supplied vessel detection coordinates and save it along with a YOLO-style bounding box label.

    Args:
        ann (pd.Series): annotation row
        img_array (np.ndarray): full image array with shape (H, W)
        crop_size (int): crop dimensions (assumes square)
        out_img_dir (Path): output folder for .npy crops
        out_lbl_dir (Path): output folder for YOLO label files
        swath_idx (int): swath index (1, 2, or 3)
        saved_filenames (set): set to track filenames for duplicate detection
    
    Returns:
        bool: True if crop was successfully created and saved, False otherwise
    """
    detect_id = ann['detect_id']
    
    # Create unique filename incorporating swath information
    filename_base = f"{detect_id}_swath{swath_idx}"
    
    # Check for duplicate filenames (shouldn't happen with swath info, but safety check)
    if saved_filenames is not None:
        if filename_base in saved_filenames:
            print(f"Warning: Duplicate filename found: {filename_base} - this shouldn't happen!")
            return False
        saved_filenames.add(filename_base)

    try:
        # Validate image shape
        if len(img_array.shape) != 2:
            print(f"Error: Expected image shape (H, W), but got {img_array.shape} for {detect_id}")
            return False
            
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
            print(f"Warning: Detection center ({centre_row}, {centre_col}) outside image bounds ({H}, {W}) for {detect_id}")
            return False

        # Compute crop coordinates
        top, bottom, left, right = extract_crop_coords(centre_row, centre_col, crop_size, H, W)

        # Extract 2D crop
        crop = img_array[top:bottom, left:right]
        
        # Validate crop size
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            print(f"Warning: Crop size mismatch. Expected {crop_size}x{crop_size}, got {crop.shape[0]}x{crop.shape[1]} for {detect_id}")
            return False

        # Determine class: 0 = vessel, 1 = fishing vessel
        class_id = 1 if pd.notna(ann.get("is_fishing")) and ann["is_fishing"] is True else 0

        # Convert bounding box to YOLO format (xc, yc, w, h)
        box_left = float(ann["left"]) - left
        box_top = float(ann["top"]) - top
        box_right = float(ann["right"]) - left
        box_bottom = float(ann["bottom"]) - top
        
        # Validate bounding box coordinates
        if box_left < 0 or box_top < 0 or box_right > crop_size or box_bottom > crop_size:
            print(f"Warning: Bounding box extends outside crop boundaries for {detect_id}")
            return False
            
        xc = (box_left + box_right) / 2 / crop_size
        yc = (box_top + box_bottom) / 2 / crop_size
        w = (box_right - box_left) / crop_size
        h = (box_bottom - box_top) / crop_size

        # Save .npy crop and .txt label using unique filename
        image_path = out_img_dir / f"{filename_base}.npy"
        label_path = out_lbl_dir / f"{filename_base}.txt"
        
        try:
            np.save(image_path, crop)
            with open(label_path, "w") as f:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            
            # Verify files were actually created
            if not image_path.exists():
                print(f"Error: Image file was not created: {image_path}")
                return False
            if not label_path.exists():
                print(f"Error: Label file was not created: {label_path}")
                return False
                
            print(f"  Saved: {filename_base}")
            return True
            
        except Exception as save_error:
            print(f"Error saving files for {filename_base}: {save_error}")
            return False

    except FileNotFoundError as e:
        print(f"File not found for {detect_id}: {e}")
        return False
    except ValueError as e:
        print(f"Data validation error for {detect_id}: {e}")
        return False
    except KeyError as e:
        print(f"Missing required data field for {detect_id}: {e}")
        return False
    except Exception as e:
        import traceback
        print(f"Unexpected error processing {detect_id}: {e}")
        traceback.print_exc()
        return False

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
    -----------------------------------------------------------------------
    """
    # Load configuration from YAML
    try:
        with open("cropping.yaml", "r") as f:
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
    
    # Ensure confidence_levels is a list
    if isinstance(confidence_levels, str):
        confidence_levels = [confidence_levels]

    # Check if required files and directories exist
    if not correspondence_path.exists():
        print(f"Correspondence file not found: {correspondence_path}")
        return
    
    if not annotation_path.exists():
        print(f"Annotation file not found: {annotation_path}")
        return
        
    if not arrays_path.exists():
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
        print(f"Error joining correspondence and annotation files: {e}")
        return

    # Filter annotations based on criteria
    filtered_annotations = joined_df[
        (joined_df["is_vessel"] == True) &
        (joined_df["is_fishing"].isin([True, False])) &
        (joined_df["confidence"].isin(confidence_levels))
    ].dropna(subset=["top", "left", "bottom", "right"])

    if filtered_annotations.empty:
        print("No annotations found matching the specified criteria.")
        return

    # Get all .npy files in arrays directory
    npy_files = list(arrays_path.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {arrays_path}")
        return

    summary = []  # to collect counts of saved crops per scene
    total_processed = 0
    saved_filenames = set()  # Track all filenames to catch duplicates

    print(f"Processing {len(npy_files)} array files...")

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
            print(f"Warning: No correspondence entry found for array file: {npy_file.name}")
            continue
            
        if len(matching_rows) > 1:
            print(f"Warning: Multiple correspondence entries found for array file: {npy_file.name}")
            continue
            
        corr_row, swath_idx = matching_rows[0]
        scene_id = corr_row['scene_id']
        slc_product_id = corr_row['SLC_product_identifier']
        
        # Load the array
        try:
            img_array = np.load(npy_file)
            print(f"Processing {npy_file.name} (scene: {scene_id}, swath: {swath_idx})")
        except Exception as e:
            print(f"Error loading array file {npy_file}: {e}")
            continue

        # Get annotations for this specific scene and swath
        scene_annotations = filtered_annotations[
            (filtered_annotations["scene_id"] == scene_id) &
            (filtered_annotations["SLC_product_identifier"] == slc_product_id) &
            (filtered_annotations["swath_index"] == swath_idx)
        ]

        scene_crop_count = 0
        for _, ann in scene_annotations.iterrows():
            success = process_crop(ann, img_array, crop_size, out_img_dir, out_lbl_dir, swath_idx, saved_filenames)
            if success:
                scene_crop_count += 1
                total_processed += 1

        if scene_crop_count > 0:
            summary.append({"scene_id": scene_id, "array_file": npy_file.name, "num_crops": scene_crop_count})
            print(f"  -> Successfully created {scene_crop_count} crops")
        else:
            print(f"  -> No crops created (all failed validation)")

    # Final verification
    actual_image_count = len(list(out_img_dir.glob("*.npy")))
    actual_label_count = len(list(out_lbl_dir.glob("*.txt")))
    
    print(f"\nFinal verification:")
    print(f"Script reported: {total_processed} crops")
    print(f"Actual image files: {actual_image_count}")
    print(f"Actual label files: {actual_label_count}")
    print(f"Unique filenames processed: {len(saved_filenames)}")
    
    if total_processed != actual_image_count:
        print(f"MISMATCH: Expected {total_processed} images but found {actual_image_count}")
    if actual_image_count != actual_label_count:
        print(f"MISMATCH: Image count ({actual_image_count}) != Label count ({actual_label_count})")

    # Save summary CSV
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(crop_path / "crop_summary.csv", index=False)
        print(f"Crop summary saved to: {crop_path / 'crop_summary.csv'}")
        print(f"Total crops processed: {total_processed}")
    else:
        print("No crops were created.")

if __name__ == "__main__":
    main()
