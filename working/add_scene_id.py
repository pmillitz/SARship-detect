#!/usr/bin/env python3

"""
add_scene_id.py

Maps each SLC or GRD image file to its scene_id in the correspondences file
and inserts the scene_id as an entry into the corresponding compute stats file. 

Author: Peter Millitz
Date: 07-06-2025
"""

import pandas as pd
from pathlib import Path

def add_scene_id_to_files(correspondence_file, slc_file, grd_file, output_slc_file, output_grd_file):
    """
    Add scene_id column to SLC and GRD files using correspondence mapping.
    If one file is missing/empty, process the other. If both are missing/empty, terminate with error.
    
    Args:
        correspondence_file: Path to the correspondence CSV file
        slc_file: Path to the SLC CSV file
        grd_file: Path to the GRD CSV file
        output_slc_file: Path for the output SLC file with scene_id
        output_grd_file: Path for the output GRD file with scene_id
    """
    
    # Read the correspondence file
    print("Reading correspondence file...")
    correspondence_df = pd.read_csv(correspondence_file)
    
    # Check if correspondence file is empty
    if correspondence_df.empty:
        raise ValueError("Correspondence file is empty")
    
    # Pre-create mapping dictionaries from correspondence file
    print("Creating scene_id mappings...")
    slc_to_scene = {}
    grd_to_scene = {}
    
    if 'SLC_product_identifier' in correspondence_df.columns:
        slc_to_scene = correspondence_df.set_index('SLC_product_identifier')['scene_id'].to_dict()
        #print(f"SLC mapping created: {len(slc_to_scene)} entries")
    
    if 'GRD_product_identifier' in correspondence_df.columns:
        grd_to_scene = correspondence_df.set_index('GRD_product_identifier')['scene_id'].to_dict()
        #print(f"GRD mapping created: {len(grd_to_scene)} entries")
    
    # Function to process a single file type
    def process_file(file_path, file_type, mapping_dict, output_path):
        """Process a single file and return success status and missing count."""
        if not mapping_dict:
            print(f"No {file_type} mapping available, skipping...")
            return False, 0
            
        # Check if file exists and is readable
        if not Path(file_path).exists():
            print(f"Warning: {file_type} file not found, skipping...")
            return False, 0
            
        try:
            print(f"Processing {file_type} file: {file_path}")
            
            # Read given file type
            df = pd.read_csv(file_path, dtype={'safe_directory': 'string'})
            
            if df.empty:
                print(f"Warning: {file_type} file is empty, skipping...")
                return False, 0
                
            print(f"{file_type} file loaded: {len(df)} rows")
            
            # Validate required column exists
            if 'safe_directory' not in df.columns:
                print(f"Error: 'safe_directory' column not found in {file_type} file")
                print(f"Available columns: {list(df.columns)}")
                return False, 0
            
            # Remove .SAFE suffix and map to scene_id
            safe_dirs_clean = df['safe_directory'].str.replace('.SAFE', '', regex=False)
            scene_ids = safe_dirs_clean.map(mapping_dict)
            
            # Insert scene_id as first column
            df.insert(0, 'scene_id', scene_ids)
            
            # Count and report missing mappings
            missing_count = scene_ids.isnull().sum()
            if missing_count > 0:
                print(f"Warning: {missing_count} rows in {file_type} file could not be mapped to scene_id")
            
            # Write output with optimized settings
            df.to_csv(output_path, index=False, lineterminator='\n')
            print(f"{file_type} file with scene_id saved to: {output_path}")
            
            return True, missing_count
            
        except Exception as e:
            print(f"Warning: Error processing {file_type} file ({e}), skipping...")
            return False, 0
    
    # Process both files
    slc_success, slc_missing = process_file(slc_file, "SLC", slc_to_scene, output_slc_file)
    grd_success, grd_missing = process_file(grd_file, "GRD", grd_to_scene, output_grd_file)
    
    # Check if at least one file was processed successfully
    if not slc_success and not grd_success:
        raise ValueError("Both SLC and GRD files could not be processed. Cannot proceed.")
    
    return slc_success, grd_success, slc_missing, grd_missing

if __name__ == "__main__":
    # Input files name
    correspondence_file = "correspondences.csv"
    slc_file = "slc-vh_stats.csv"  
    grd_file = "grd-vh_stats.csv"
    
    # Output file names
    output_slc_file = "slc-vh_stats_mod.csv"
    output_grd_file = "grd-vh_stats_mod.csv"
    
    try:
        slc_success, grd_success, slc_missing, grd_missing = add_scene_id_to_files(
            correspondence_file=correspondence_file,
            slc_file=slc_file, 
            grd_file=grd_file,
            output_slc_file=output_slc_file,
            output_grd_file=output_grd_file
        )
        
#        print("\n" + "="*50)
#        print("PROCESS SUMMARY:")
#        print(f"SLC file processed: {'Yes' if slc_success else 'No'}")
#        if slc_success:
#            print(f"  - Missing mappings: {slc_missing}")
#        print(f"GRD file processed: {'Yes' if grd_success else 'No'}")
#        if grd_success:
#            print(f"  - Missing mappings: {grd_missing}")
#        print("Process completed successfully!")
        
    except ValueError as e:
        print(f"Error: {e}")
    except KeyError as e:
        print(f"Error: Missing expected column - {e}")
    except Exception as e:
        print(f"Error: {e}")

