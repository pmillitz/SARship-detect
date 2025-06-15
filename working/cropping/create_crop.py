# create_crop.py
# 
# Author: Peter Millitz
# Date: 26-05-2025
# ----------------------------------------------------------------------
# This script extracts 4-channel crops (vh_mag, vh_phase, vv_mag, vv_phase)
# from SAR SLC images using vessel annotations, and outputs them as NumPy
# arrays (.npy) and YOLO-style (.txt) label files. It supports selecting specific
# scenes for processing and saves a summary CSV of the number of crops created.

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from GeoTiff import load_GeoTiff

def extract_crop_coords(center_row, center_col, crop_size, img_height, img_width):
    """
    Compute the bounding box for a square crop centered on (center_row, center_col),
    ensuring the crop stays within image bounds.
    
    Returns:
        (top, bottom, left, right): int crop boundaries
    """
    half = crop_size // 2
    top = max(0, center_row - half)
    left = max(0, center_col - half)
    bottom = min(img_height, top + crop_size)
    right = min(img_width, left + crop_size)

    # Ensure dimensions match requested crop_size
    if bottom - top < crop_size:
        top = max(0, bottom - crop_size)
    if right - left < crop_size:
        left = max(0, right - crop_size)

    return top, bottom, left, right

def process_crop(ann, row, crop_size, measurement_dir, swath_index, out_img_dir, out_lbl_dir):
    """
    Process a single vessel annotation: extract 4-channel crop and save it
    along with a YOLO-style bounding box label.

    Args:
        ann (pd.Series): annotation row
        row (pd.Series): correspondence row
        crop_size (int): crop dimensions (assumes square)
        measurement_dir (Path): path to measurement directory
        swath_index (int): 1, 2, or 3
        out_img_dir (Path): output folder for .npy crops
        out_lbl_dir (Path): output folder for YOLO label files
    """
    swath_key_vh = f"SLC_swath_{swath_index}_vh"
    swath_key_vv = f"SLC_swath_{swath_index}_vv"
    base_name = f"{ann['scene_id']}_{ann['detect_id']}"

    try:
        # Load VH and VV complex SAR images
        vh_path = measurement_dir / row[swath_key_vh]
        vv_path = measurement_dir / row[swath_key_vv]
        vh_img, *_ = load_GeoTiff(str(vh_path))
        vv_img, *_ = load_GeoTiff(str(vv_path))
        assert vh_img.shape == vv_img.shape
        H, W = vh_img.shape

       # Use provided detection center if available, else fallback to bounding box center
        if "detect_scene_row" in ann and "detect_scene_column" in ann:
            center_row = int(ann["detect_scene_row"])
            center_col = int(ann["detect_scene_column"])
        else:
            center_row = int((ann["top"] + ann["bottom"]) / 2)
            center_col = int((ann["left"] + ann["right"]) / 2)

        # Compute crop coordinates
        top, bottom, left, right = extract_crop_coords(center_row, center_col, crop_size, H, W)

        # Assemble 4-channel crop: [VH_mag, VH_phase, VV_mag, VV_phase]
        vh_crop = vh_img[top:bottom, left:right]
        vv_crop = vv_img[top:bottom, left:right]
        crop_4ch = np.stack([
            np.abs(vh_crop), np.angle(vh_crop),
            np.abs(vv_crop), np.angle(vv_crop)
        ], axis=0)

        # Determine class: 0 = vessel, 1 = fishing vessel
        class_id = 1 if pd.notna(ann.get("is_fishing")) and ann["is_fishing"] is True else 0

        # Convert bounding box to YOLO format (xc, yc, w, h)
        box_left = float(ann["left"]) - left
        box_top = float(ann["top"]) - top
        box_right = float(ann["right"]) - left
        box_bottom = float(ann["bottom"]) - top
        xc = (box_left + box_right) / 2 / crop_size
        yc = (box_top + box_bottom) / 2 / crop_size
        w = (box_right - box_left) / crop_size
        h = (box_bottom - box_top) / crop_size

        # Save .npy crop and .txt label
        image_path = out_img_dir / f"{base_name}.npy"
        label_path = out_lbl_dir / f"{base_name}.txt"
        np.save(image_path, crop_4ch)
        with open(label_path, "w") as f:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    except Exception as e:
        import traceback
        print(f"Failed to crop {base_name}: {e}")
        traceback.print_exc()

def main():
    """
    Main driver function to extract crops and labels for vessel detection from SAR SLC scenes.

    Steps:
    1. Load paths and settings from cropping.yaml
    2. Create output directories for image and label crops
    3. Optionally restrict to a subset of scenes if 'ScenesToProcess' is specified
    4. Iterate through correspondence file for each selected scene
       a. Find the associated annotation file
       b. Filter for valid vessel detections with matching scene_id, swath, and confidence level
    5. For each swath, locate the correct VH and VV image files from the correspondence
    6. Extract and save a 4-channel crop + YOLO-format label for each valid detection
    7. Finally, generate a summary CSV of number of crops created per scene

    -----------------------------------------------------------------------
    Expected configuration keys in cropping.yaml:
    
    SARFish_root_directory: str
        Root directory for all SARFISH data (e.g., /data/SARFishSample/SLC)

    product_type: str
        The product type (e.g., "SLC") used to construct file paths

    xView3_SLC_GRD_correspondences_path: str
        Path to the correspondence CSV file mapping scene IDs to SLC products

    CREATE_CROP:
      CropPath: str
          Output directory where cropped images and labels are saved
      CropSize: int
          Crop size (e.g., 256 for 256x256 crops)
      LabelConfidence: str
          Confidence level to filter labels (e.g., "HIGH")
      ScenesToProcess: list[str] (optional)
          If present, only these scene IDs will be processed. If absent or
          empty, all scenes in the correspondence file will be used.
    -----------------------------------------------------------------------
    """
    # Load configuration from YAML
    with open("cropping.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Resolve paths and parameters from config
    sar_root = Path(config["SARFish_root_directory"])
    product_type = config["product_type"]
    correspondence_path = Path(config["xView3_SLC_GRD_correspondences_path"])
    crop_path = Path(config["CREATE_CROP"]["CropPath"])
    crop_size = int(config["CREATE_CROP"]["CropSize"])
    confidence = config["CREATE_CROP"]["LabelConfidence"]
    scene_filter = config["CREATE_CROP"].get("ScenesToProcess", None)

    # Create output subfolders for images and labels
    out_img_dir = crop_path / "images"
    out_lbl_dir = crop_path / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata correspondence table
    corr_df = pd.read_csv(correspondence_path)

    # If scene filter is specified, reduce the correspondence table
    if scene_filter:
        corr_df = corr_df[corr_df["scene_id"].isin(scene_filter)]

    summary = []  # to collect counts of saved crops per scene

    print(f"Processing...")

    # Iterate over each scene listed in correspondence file
    for _, row in corr_df.iterrows():
        scene_id = row["scene_id"]
        partition = row["DATA_PARTITION"]
        annotation_csv = sar_root / product_type / partition / f"{product_type}_{partition}.csv"

        if not annotation_csv.exists():
            print(f"Annotation file missing: {annotation_csv}")
            continue

        annotations = pd.read_csv(annotation_csv)
        annotations = annotations[
            (annotations["scene_id"] == scene_id) &
            (annotations["is_vessel"] == True) &
            (annotations["confidence"] == confidence)
        ].dropna(subset=["top", "left", "bottom", "right", "swath_index"])

        scene_crop_count = 0

        for swath_index in [1, 2, 3]:
            vh_key = f"SLC_swath_{swath_index}_vh"
            vv_key = f"SLC_swath_{swath_index}_vv"

            if pd.isna(row.get(vh_key)) or pd.isna(row.get(vv_key)):
                continue

            measurement_dir = sar_root / product_type / partition / f"{row[f'{product_type}_product_identifier']}.SAFE" / "measurement"

            swath_annotations = annotations[annotations["swath_index"] == swath_index]
            if swath_annotations.empty:
                continue

            for _, ann in swath_annotations.iterrows():
                process_crop(ann, row, crop_size, measurement_dir, swath_index, out_img_dir, out_lbl_dir)
                scene_crop_count += 1

        summary.append({"scene_id": scene_id, "num_crops": scene_crop_count})

    # Save summary CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(crop_path / "crop_summary.csv", index=False)
    print(f"Crop summary saved to: {crop_path / 'crop_summary.csv'}")

if __name__ == "__main__":
    main()

