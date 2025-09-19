import pandas as pd
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_label_features(input_csv_path, label_dir, output_csv_path, logfile="feature_extraction.log"):
    """
    Extract width, height, and class from label files and aggregate with vessel lengths.
    
    Args:
        input_csv_path: Path to CSV file created by match_label_vessel_lengths.py
        label_dir: Directory containing .txt label files
        output_csv_path: Path for output CSV file with aggregated features
        logfile: Path to log file for detailed output
    """
    
    with open(logfile, 'w') as log:
        def log_print(message):
            log.write(message + '\n')
            log.flush()
        
        # Read the input CSV (filename -> vessel_length_m mapping)
        try:
            input_df = pd.read_csv(input_csv_path)
            log_print(f"Loaded {len(input_df)} records from input CSV")
        except Exception as e:
            log_print(f"Error loading input CSV: {e}")
            return None
        
        if not os.path.exists(label_dir):
            log_print(f"Error: Label directory {label_dir} does not exist")
            return None
        
        log_print(f"Processing labels from directory: {label_dir}")
        
        results = []
        processed_count = 0
        skipped_no_file = 0
        skipped_empty_file = 0
        skipped_invalid_format = 0
        
        # Process each row in the input CSV
        for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing label files"):
            filename = row['filename']
            vessel_length = row['vessel_length_m']
            
            label_path = Path(label_dir) / filename
            
            # Check if label file exists
            if not label_path.exists():
                log_print(f"Warning: Label file not found: {filename}")
                skipped_no_file += 1
                continue
            
            # Read and parse the label file
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    log_print(f"Warning: Empty label file: {filename}")
                    skipped_empty_file += 1
                    continue
                
                # Process each detection in the label file
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) < 5:
                            log_print(f"Warning: Invalid label format in {filename}, line {line_idx + 1}")
                            skipped_invalid_format += 1
                            continue
                        
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate YOLO format values (should be between 0 and 1)
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                               0 <= width <= 1 and 0 <= height <= 1):
                            log_print(f"Warning: Invalid YOLO coordinates in {filename}, line {line_idx + 1}")
                            skipped_invalid_format += 1
                            continue
                        
                        results.append({
                            'label_filename': filename,
                            'width': width,
                            'height': height,
                            'class': class_id,
                            'vessel_length_m': vessel_length
                        })
                        
                        log_print(f"Extracted: {filename} -> class:{class_id}, w:{width:.4f}, h:{height:.4f}, length:{vessel_length}")
                        
                    except (ValueError, IndexError) as e:
                        log_print(f"Warning: Error parsing {filename}, line {line_idx + 1}: {e}")
                        skipped_invalid_format += 1
                        continue
                
                processed_count += 1
                
            except Exception as e:
                log_print(f"Warning: Error reading label file {filename}: {e}")
                skipped_no_file += 1
                continue
        
        # Create output DataFrame
        output_df = pd.DataFrame(results)
        
        # Report processing statistics
        log_print(f"\nProcessing Summary:")
        log_print(f"  Input CSV records: {len(input_df)}")
        log_print(f"  Label files processed: {processed_count}")
        log_print(f"  Total feature records extracted: {len(output_df)}")
        log_print(f"  Skipped - no label file: {skipped_no_file}")
        log_print(f"  Skipped - empty label file: {skipped_empty_file}")
        log_print(f"  Skipped - invalid format: {skipped_invalid_format}")
        
        if len(output_df) > 0:
            # Additional statistics
            log_print(f"\nFeature Statistics:")
            log_print(f"  Class distribution: {dict(output_df['class'].value_counts())}")
            log_print(f"  Width range: {output_df['width'].min():.4f} - {output_df['width'].max():.4f}")
            log_print(f"  Height range: {output_df['height'].min():.4f} - {output_df['height'].max():.4f}")
            log_print(f"  Vessel length range: {output_df['vessel_length_m'].min():.2f} - {output_df['vessel_length_m'].max():.2f}")
            
            # Save output CSV
            output_df.to_csv(output_csv_path, index=False)
            log_print(f"\nOutput saved to: {output_csv_path}")
            
            return output_df
        else:
            log_print("No valid feature records generated!")
            return None

def main():
    parser = argparse.ArgumentParser(
        description="Extract width, height, and class from label files and aggregate with vessel lengths"
    )
    parser.add_argument('--input_csv', required=True, 
                       help='Path to CSV file from match_label_vessel_lengths.py')
    parser.add_argument('--label_dir', required=True,
                       help='Directory containing .txt label files')
    parser.add_argument('--output_csv', required=True,
                       help='Path to output CSV file with aggregated features')
    parser.add_argument('--logfile', default='feature_extraction.log',
                       help='Log file path (default: feature_extraction.log)')
    
    args = parser.parse_args()
    
    # Run feature extraction
    result = extract_label_features(
        args.input_csv, 
        args.label_dir, 
        args.output_csv, 
        args.logfile
    )
    
    if result is not None:
        print(f"Successfully extracted {len(result)} feature records")
        
        # Show first 3 lines of log
        with open(args.logfile, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:3]):
                print(f"{line.strip()}")
        
        print()
        
        # Show last 5 lines of log
        print("Last 5 lines of log:")
        for i, line in enumerate(lines[-5:]):
            print(f"{len(lines)-5+i+1}: {line.strip()}")
    else:
        print("Feature extraction failed - see log for details")

if __name__ == '__main__':
    main()