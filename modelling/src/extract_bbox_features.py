# extract_bbox_features.py
'''
Extracts features from predicted bounding boxes produced by vessel detector model
'''

import argparse
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

def extract_bbox_features(model_path, metadata_path, image_dir, file_ext, output_csv, logfile="extract_bbox_features.log"):
    with open(logfile, 'w') as log:
        def log_print(message, print_to_screen=False):
            if print_to_screen:
                print(message)
            log.write(message + '\n')
            log.flush()
        
        model = YOLO(model_path)
        image_dir = Path(image_dir)
        metadata = pd.read_csv(metadata_path)
        
        log_print(f"Loaded metadata with {len(metadata)} records", print_to_screen=True)

        # Normalize image names
        metadata['filename'] = metadata['filename'].apply(lambda x: Path(x).name)
        length_lookup = dict(zip(metadata['filename'], metadata['vessel_length_m']))
        log_print(f"Created lookup dictionary with {len(length_lookup)} entries", print_to_screen=True)

        records = []
        images_processed = 0
        skipped_no_metadata = 0
        skipped_invalid_length = 0
        skipped_no_detections = 0
        total_detections = 0
        
        image_files = list(image_dir.glob(f'*{file_ext}'))
        log_print(f"Found {len(image_files)} image files with extension {file_ext}", print_to_screen=True)
        
        for image_path in tqdm(image_files, desc="Processing images"):
            image_name = image_path.name
            images_processed += 1
            
            if image_name not in length_lookup:
                skipped_no_metadata += 1
                log_print(f"SKIP: No metadata for {image_name}")
                continue

            vessel_length_m = length_lookup[image_name]
            
            # Skip images with invalid vessel lengths
            if pd.isna(vessel_length_m) or vessel_length_m is None:
                skipped_invalid_length += 1
                log_print(f"SKIP: Invalid vessel length for {image_name} (length={vessel_length_m})")
                continue

            result = model(image_path, conf=0.01, iou=0.65, verbose=False)[0]  # conf default=0.25; iou default=0.7
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                skipped_no_detections += 1
                log_print(f"SKIP: No detections for {image_name}")
                continue

            # Select only the highest confidence detection
            best_idx = boxes.conf.argmax()
            best_box = boxes[best_idx]

            log_print(f"PROCESS: {image_name} - {len(boxes)} detections, using highest conf, vessel_length={vessel_length_m}")

            xywh = best_box.xywhn.cpu().numpy().flatten()
            conf = best_box.conf.item()
            cls = int(best_box.cls.item())
            width, height = xywh[2], xywh[3]
            total_detections += 1

            records.append({
                'image': image_name,
                'width': width,     # predicted
                'height': height,   # predicted
                'conf': conf,       # predicted
                'class': cls,       # predicted
                'vessel_length_m': vessel_length_m  # ground truth
            })

        log_print(f"\n=== PROCESSING SUMMARY ===", print_to_screen=True)
        log_print(f"Total images found: {len(image_files)}", print_to_screen=True)
        log_print(f"Images processed: {images_processed}", print_to_screen=True)
        log_print(f"Skipped - no metadata: {skipped_no_metadata}", print_to_screen=True)
        log_print(f"Skipped - invalid vessel length: {skipped_invalid_length}", print_to_screen=True)
        log_print(f"Skipped - no detections: {skipped_no_detections}", print_to_screen=True)
        log_print(f"Total detections found: {total_detections}", print_to_screen=True)
        log_print(f"Records created: {len(records)}", print_to_screen=True)

        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        log_print(f"Saved {len(df)} records to {output_csv}", print_to_screen=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract YOLO detection features for vessel length regression")
    parser.add_argument('--model', required=True, help='Path to YOLOv8 model (e.g. best.pt)')
    parser.add_argument('--metadata', required=True, help='Path to CSV with image names and vessel lengths')
    parser.add_argument('--images', required=True, help='Directory containing input images')
    parser.add_argument('--ext', default='.png', help='Image file extension (default: .png)')
    parser.add_argument('--output', required=True, help='Path to output CSV file')
    parser.add_argument('--logfile', default='extract_bbox_features.log', help='Log file path (default: extract_bbox_features.log)')

    args = parser.parse_args()
    extract_bbox_features(args.model, args.metadata, args.images, args.ext, args.output, args.logfile)
