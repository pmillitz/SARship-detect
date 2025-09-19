import pandas as pd
import os
import re
import argparse
from pathlib import Path
from tqdm import tqdm

def match_label_with_vessel_lengths(input_csv_path, label_dir, output_csv_path, logfile="matching.log"):
    """
    Match .txt label files with vessel lengths from CSV lookup table.
    
    Args:
        input_csv_path: Path to df_labels_train_filt.csv or similar
        label_dir: Directory containing .txt files with format [detect_id]_swath[swath_index]*.txt
        output_csv_path: Path for output CSV file
        logfile: Path to log file for detailed output
    
    Returns:
        Tuple of (output_df, matched_count, unparsed_count, skipped_count) or None if failed
    """
    
    with open(logfile, 'w') as log:
        def log_print(message):
            log.write(message + '\n')
            log.flush()
        
        # Read the lookup CSV
        df_labels = pd.read_csv(input_csv_path)
        log_print(f"Loaded {len(df_labels)} records from CSV")
        
        # Create lookup dictionary: (detect_id, swath_index) -> vessel_length_m
        lookup_dict = {}
        for _, row in df_labels.iterrows():
            key = (row['detect_id'], row['swath_index'])
            lookup_dict[key] = row['vessel_length_m']
        
        log_print(f"Created lookup dictionary with {len(lookup_dict)} entries")
        
        # Get all .txt files
        label_files = []
        if os.path.exists(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith('.txt'):
                    label_files.append(file)
        else:
            log_print(f"Warning: Directory {label_dir} does not exist")
            return None
        
        log_print(f"Found {len(label_files)} .txt files")
        
        # Match files with vessel lengths
        results = []
        pattern = r'^(.+)_swath(\d+).*\.txt$'
        mosaic_pattern = r'^mosaic_(minority|majority)_\d+_proc\.txt$'
        unparsed_files = []
        skipped_mosaic_files = []
        
        for filename in label_files:
            # Skip mosaic files
            if re.match(mosaic_pattern, filename):
                log_print(f"Skipping mosaic file: {filename}")
                skipped_mosaic_files.append(filename)
                continue
                
            match = re.match(pattern, filename)
            if match:
                detect_id = match.group(1)
                swath_index = int(match.group(2))
                
                # Look up vessel length
                key = (detect_id, swath_index)
                vessel_length = lookup_dict.get(key, None)
                
                if vessel_length is None:
                    log_print(f"Warning: No vessel length found for {filename} (detect_id: {detect_id}, swath_index: {swath_index}) - SKIPPING")
                elif pd.isna(vessel_length):
                    log_print(f"Warning: Vessel length is NaN for {filename} (detect_id: {detect_id}, swath_index: {swath_index}) - SKIPPING")
                else:
                    log_print(f"Matched: {filename} -> vessel_length_m: {vessel_length}")
                    results.append({
                        'filename': filename,
                        'vessel_length_m': vessel_length
                    })
            else:
                log_print(f"Warning: Could not parse filename {filename} - SKIPPING")
                unparsed_files.append(filename)
        
        # Create output DataFrame
        output_df = pd.DataFrame(results)
        
        # Report matching statistics
        matched_count = output_df['vessel_length_m'].notna().sum()
        nan_count = output_df['vessel_length_m'].isna().sum()
        log_print(f"Successfully matched {matched_count}/{len(results)} files")
        log_print(f"Files with NaN vessel lengths: {nan_count}")
        log_print(f"Unparsed files: {len(unparsed_files)}")
        log_print(f"Skipped mosaic files: {len(skipped_mosaic_files)}")
        
        # Save output CSV
        output_df.to_csv(output_csv_path, index=False)
        log_print(f"Output saved to {output_csv_path}")
        
        return output_df, matched_count, len(unparsed_files), len(skipped_mosaic_files)

def build_label_vessel_length_table(input_csv_path, label_dir, output_csv_path, logfile="matching.log", return_data=False):
    """
    Run the matching and display summary output (useful for notebooks)
    
    Args:
        return_data: If True, returns the result data. If False, only prints summary.
    """
    result = match_label_with_vessel_lengths(input_csv_path, label_dir, output_csv_path, logfile)
    
    if result is not None:
        result_df, matched_count, unparsed_count, skipped_count = result
        
        # Show first 3 lines of log
        with open(logfile, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:3]):
                print(f"{line.strip()}")
        
        # Show last 5 lines of log")
        for i, line in enumerate(lines[-5:]):
            print(f"{line.strip()}")
            
        if return_data:
            return result
    else:
        print("Processing failed - see log for details")

def extract_label_features(input_csv_path, label_dir, output_csv_path, logfile="feature_extraction.log"):
    """
    Extract width, height, and class from label files and aggregate with vessel lengths.
    
    Args:
        input_csv_path: Path to CSV file created by match_label_vessel_lengths.py
        label_dir: Directory containing .txt label files
        output_csv_path: Path for output CSV file with aggregated features
        logfile: Path to log file for detailed output
    
    Returns:
        DataFrame with extracted features or None if failed
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

def extract_features_from_labels(metadata_csv_path, label_dir, output_csv_path, 
                                matching_logfile="matching.log", 
                                extraction_logfile="feature_extraction.log",
                                temp_csv_path=None):
    """
    Complete pipeline: match labels with vessel lengths, then extract features.
    Convenience function that combines both steps.
    
    Args:
        metadata_csv_path: Path to CSV with detect_id, swath_index, vessel_length_m
        label_dir: Directory containing .txt label files
        output_csv_path: Final output CSV with extracted features
        matching_logfile: Log file for matching step
        extraction_logfile: Log file for feature extraction step
        temp_csv_path: Temporary CSV for matching results (auto-generated if None)
    
    Returns:
        DataFrame with extracted features or None if failed
    """
    
    # Generate temp file path if not provided
    if temp_csv_path is None:
        temp_csv_path = str(Path(output_csv_path).parent / "temp_vessel_length_mapping.csv")
    
    print("Step 1: Matching label files with vessel lengths...")
    
    # Step 1: Match labels with vessel lengths
    match_result = match_label_with_vessel_lengths(
        metadata_csv_path, label_dir, temp_csv_path, matching_logfile
    )
    
    if match_result is None:
        print("Step 1 failed - see matching log for details")
        return None
    
    result_df, matched_count, unparsed_count, skipped_count = match_result
    print(f"Step 1 completed: {matched_count} files matched")
    
    # Step 2: Extract features from matched labels
    print("\nStep 2: Extracting features from label files...")
    
    features_df = extract_label_features(
        temp_csv_path, label_dir, output_csv_path, extraction_logfile
    )
    
    if features_df is not None:
        print(f"Step 2 completed: {len(features_df)} feature records extracted")
        
        # Clean up temp file
        try:
            os.remove(temp_csv_path)
        except:
            pass  # Ignore cleanup errors
    else:
        print("Step 2 failed - see extraction log for details")
    
    return features_df

def main():
    parser = argparse.ArgumentParser(
        description="Extract features from ground truth YOLO labels with vessel length matching"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Match command
    match_parser = subparsers.add_parser('match', help='Match label files with vessel lengths')
    match_parser.add_argument('--input_csv', required=True, help='Path to df_labels_train_filt.csv')
    match_parser.add_argument('--label_dir', required=True, help='Directory containing .txt label files')
    match_parser.add_argument('--output_csv', required=True, help='Output CSV file path')
    match_parser.add_argument('--logfile', default='matching.log', help='Log file path')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract features from label files')
    extract_parser.add_argument('--input_csv', required=True, help='CSV from match step')
    extract_parser.add_argument('--label_dir', required=True, help='Directory containing .txt label files')
    extract_parser.add_argument('--output_csv', required=True, help='Output CSV with features')
    extract_parser.add_argument('--logfile', default='feature_extraction.log', help='Log file path')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run complete pipeline (match + extract)')
    full_parser.add_argument('--metadata_csv', required=True, help='Path to metadata CSV')
    full_parser.add_argument('--label_dir', required=True, help='Directory containing .txt label files')
    full_parser.add_argument('--output_csv', required=True, help='Final output CSV with features')
    full_parser.add_argument('--match_log', default='matching.log', help='Matching log file')
    full_parser.add_argument('--extract_log', default='feature_extraction.log', help='Extraction log file')
    
    args = parser.parse_args()
    
    if args.command == 'match':
        result = match_label_with_vessel_lengths(args.input_csv, args.label_dir, args.output_csv, args.logfile)
        if result:
            print(f"Matching completed successfully")
        else:
            print("Matching failed")
    
    elif args.command == 'extract':
        result = extract_label_features(args.input_csv, args.label_dir, args.output_csv, args.logfile)
        if result is not None:
            print(f"Feature extraction completed: {len(result)} records")
        else:
            print("Feature extraction failed")
    
    elif args.command == 'full':
        result = extract_features_from_labels(
            args.metadata_csv, args.label_dir, args.output_csv, 
            args.match_log, args.extract_log
        )
        if result is not None:
            print(f"Complete pipeline finished: {len(result)} feature records")
        else:
            print("Pipeline failed")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()