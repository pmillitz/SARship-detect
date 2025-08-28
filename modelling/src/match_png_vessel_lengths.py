import pandas as pd
import os
import re
import argparse
from pathlib import Path

def match_png_with_vessel_lengths(input_csv_path, png_dir, output_csv_path, logfile="matching.log"):
    """
    Match .png files with vessel lengths from CSV lookup table.
    
    Args:
        input_csv_path: Path to df_labels_train_filt.csv or similar
        png_dir: Directory containing .png files with format [detect_id]_swath[swath_index]*.png
        output_csv_path: Path for output CSV file
        logfile: Path to log file for detailed output
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
        
        # Get all .png files
        png_files = []
        if os.path.exists(png_dir):
            for file in os.listdir(png_dir):
                if file.endswith('.png'):
                    png_files.append(file)
        else:
            log_print(f"Warning: Directory {png_dir} does not exist")
            return None
        
        log_print(f"Found {len(png_files)} .png files")
        
        # Match files with vessel lengths
        results = []
        pattern = r'^(.+)_swath(\d+).*\.png$'
        mosaic_pattern = r'^mosaic_(minority|majority)_\d+_proc\.png$'
        unparsed_files = []
        skipped_mosaic_files = []
        
        for filename in png_files:
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

def build_image_vessel_length_table(input_csv_path, png_dir, output_csv_path, logfile="matching.log", return_data=False):
    """
    Run the matching and display summary output (useful for notebooks)
    
    Args:
        return_data: If True, returns the result data. If False, only prints summary.
    """
    result = match_png_with_vessel_lengths(input_csv_path, png_dir, output_csv_path, logfile)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Match .png files with vessel lengths from CSV lookup')
    parser.add_argument('--input_csv_path', required=True, help='Path to df_labels_train_filt.csv')
    parser.add_argument('--png_dir', required=True, help='Directory containing .png files')
    parser.add_argument('--output_csv_path', required=True, help='Output CSV file path')
    parser.add_argument('--logfile', default='matching.log', help='Log file path (default: matching.log)')
    
    args = parser.parse_args()
    
    # Run the matching
    result = match_png_with_vessel_lengths(args.input_csv_path, args.png_dir, args.output_csv_path, args.logfile)
    
    if result is not None:
        result_df, matched_count, unparsed_count, skipped_count = result
        
        # Show first 3 lines of log
        with open(args.logfile, 'r') as f:
            lines = f.readlines()
            #print("First 3 lines of log:")
            for i, line in enumerate(lines[:3]):
                print(f"{line.strip()}")
        
        print()
        
        # Show last 5 lines of log
        print("Last 5 lines of log:")
        for i, line in enumerate(lines[-5:]):
            print(f"{len(lines)-5+i+1}: {line.strip()}")
            
    else:
        print("Processing failed - see log for details")
