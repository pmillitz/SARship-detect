# extract_features.py

import argparse
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

def extract_features(model_path, metadata_path, image_dir, file_ext, output_csv):
    model = YOLO(model_path)
    image_dir = Path(image_dir)
    metadata = pd.read_csv(metadata_path)

    # Normalize image names
    metadata['filename'] = metadata['filename'].apply(lambda x: Path(x).name)
    length_lookup = dict(zip(metadata['filename'], metadata['vessel_length_m']))

    records = []
    for image_path in tqdm(list(image_dir.glob(f'*{file_ext}'))):
        image_name = image_path.name
        if image_name not in length_lookup:
            continue

        result = model(image_path, verbose=False)[0]
        boxes = result.boxes
        vessel_length = length_lookup[image_name]
        
        # Skip images with invalid vessel lengths
        if pd.isna(vessel_length) or vessel_length is None:
            continue

        for box in boxes:
            xywh = box.xywhn.cpu().numpy().flatten()
            conf = box.conf.item()
            cls = int(box.cls.item())
            width, height = xywh[2], xywh[3]

            records.append({
                'image': image_name,
                'width': width,
                'height': height,
                'conf': conf,
                'class': cls,
                'length': vessel_length
            })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} records to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract YOLO detection features for vessel length regression")
    parser.add_argument('--model', required=True, help='Path to YOLOv8 model (e.g. best.pt)')
    parser.add_argument('--metadata', required=True, help='Path to CSV with image names and vessel lengths')
    parser.add_argument('--images', required=True, help='Directory containing input images')
    parser.add_argument('--ext', default='.png', help='Image file extension (default: .png)')
    parser.add_argument('--output', required=True, help='Path to output CSV file')

    args = parser.parse_args()
    extract_features(args.model, args.metadata, args.images, args.ext, args.output)
