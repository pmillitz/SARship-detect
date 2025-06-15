#!/usr/bin/env python3
"""This code combines VH,VV complex SLC images into tiles images for the purpose of
training/testing the detection/classification algorithm.
After running this script, it should create the following file structure:
[TilePath]/[scene_id]/swath1/0_0.npy | This is the chip data (complex numpy array),
| the filename is tx_ty.npy, where tx is the
tile index in x axis,
| and ty is the tile index in y axis
| Each .npy has 4 channels, vh_mag, vh_phase,
vv_mag, vv_phase
[TilePath]/img_file_info.csv | This file saved the file info of the scenes
used
"""
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import GeoTiff
import os
import sys
import gc  # Garbage collection
import time
import traceback

def process_tile_directly(
    vhFN: Path, vvFN: Path, outDir: Path, 
    tileSize: int, tileOverlap: int,
    tile_x: int, tile_y: int,
    memory_efficient: bool = False
):
    """Process a single tile directly from the source files to save memory.
    
    Args:
        vhFN: Path to VH GeoTiff file
        vvFN: Path to VV GeoTiff file
        outDir: Output directory for the tile
        tileSize: Size of the tile
        tileOverlap: Overlap between tiles
        tile_x: X index of the tile
        tile_y: Y index of the tile
        memory_efficient: If True, use memory-efficient processing
    """
    try:
        # Calculate the tile boundaries
        stride = tileSize - tileOverlap
        x1 = tile_x * stride
        y1 = tile_y * stride
        x2 = x1 + tileSize
        y2 = y1 + tileSize
        
        # Create output filename
        tile_fn = Path(outDir, f"{tile_x}_{tile_y}.npy")
        
        # Skip if the tile already exists
        if tile_fn.exists():
            print(f"Tile {tile_x}_{tile_y}.npy already exists, skipping.")
            return True
        
        if memory_efficient:
            # Load only the region of interest from VH
            vhImg = GeoTiff.load_GeoTiff_region(str(vhFN), x1, y1, tileSize, tileSize)
            if vhImg is None or vhImg.size == 0:
                print(f"Warning: Empty VH region for tile {tile_x}_{tile_y}. Skipping.")
                return False
            
            # Calculate magnitude and phase for VH
            vh_abs = np.abs(vhImg)
            vh_phase = np.angle(vhImg)
            
            # Free memory
            del vhImg
            gc.collect()
            
            # Load only the region of interest from VV
            vvImg = GeoTiff.load_GeoTiff_region(str(vvFN), x1, y1, tileSize, tileSize)
            if vvImg is None or vvImg.size == 0:
                print(f"Warning: Empty VV region for tile {tile_x}_{tile_y}. Skipping.")
                return False
                
            # Calculate magnitude and phase for VV
            vv_abs = np.abs(vvImg)
            vv_phase = np.angle(vvImg)
            
            # Free memory
            del vvImg
            gc.collect()
            
            # Stack the channels
            combined = np.stack([vh_abs, vh_phase, vv_abs, vv_phase], axis=0)
            
            # Free memory
            del vh_abs, vh_phase, vv_abs, vv_phase
            gc.collect()
        else:
            # Note: This is not a memory-efficient approach, but included as fallback
            # if GeoTiff doesn't support region loading
            
            # Load full VH image
            (vhImg, _, _, _) = GeoTiff.load_GeoTiff(str(vhFN))
            padImg_vh = np.pad(vhImg, ((0, y2-vhImg.shape[0]) if y2 > vhImg.shape[0] else (0,0), 
                                       (0, x2-vhImg.shape[1]) if x2 > vhImg.shape[1] else (0,0)), 
                             mode="constant", constant_values=0)
            vh_abs = np.abs(padImg_vh[y1:y2, x1:x2])
            vh_phase = np.angle(padImg_vh[y1:y2, x1:x2])
            
            # Free memory
            del vhImg, padImg_vh
            gc.collect()
            
            # Load full VV image
            (vvImg, _, _, _) = GeoTiff.load_GeoTiff(str(vvFN))
            padImg_vv = np.pad(vvImg, ((0, y2-vvImg.shape[0]) if y2 > vvImg.shape[0] else (0,0), 
                                       (0, x2-vvImg.shape[1]) if x2 > vvImg.shape[1] else (0,0)), 
                             mode="constant", constant_values=0)
            vv_abs = np.abs(padImg_vv[y1:y2, x1:x2])
            vv_phase = np.angle(padImg_vv[y1:y2, x1:x2])
            
            # Free memory
            del vvImg, padImg_vv
            gc.collect()
            
            # Stack the channels
            combined = np.stack([vh_abs, vh_phase, vv_abs, vv_phase], axis=0)
        
        # Save the tile
        with open(str(tile_fn), "wb") as f:
            np.save(f, combined)
        
        # Free memory
        del combined
        gc.collect()
        
        return True
    
    except Exception as e:
        print(f"Error processing tile {tile_x}_{tile_y}: {e}")
        traceback.print_exc()
        return False

def get_image_dimensions(file_path):
    """Get the dimensions of an image without loading the full data."""
    try:
        # This is a placeholder - you may need to adapt this based on your GeoTiff library
        # Many libraries have ways to get metadata without loading the full image
        (img, _, _, _) = GeoTiff.load_GeoTiff(str(file_path))
        return img.shape
    except Exception as e:
        print(f"Error getting image dimensions: {e}")
        return None

def process_image_tiles(
    vhFN: Path, vvFN: Path, outDir: Path, 
    tileSize: int, tileOverlap: int,
    memory_efficient: bool = True
):
    """Process all tiles for an image pair using a memory-efficient approach."""
    print(f"Processing tiles with memory-efficient mode: {memory_efficient}")
    
    try:
        # Get image dimensions without loading the full data
        dims = get_image_dimensions(vhFN)
        if dims is None:
            print("Could not determine image dimensions. Aborting.")
            return False
        
        imgH, imgW = dims
        print(f"Image dimensions: {imgH}x{imgW}")
        
        # Calculate the effective stride and number of tiles
        stride = tileSize - tileOverlap
        numTileX = int(np.ceil(imgW / stride))
        numTileY = int(np.ceil(imgH / stride))
        
        print(f"Will create approximately {numTileX}x{numTileY} = {numTileX*numTileY} tiles")
        
        # Create output directory if needed
        if not outDir.exists():
            print(f"Creating output directory: {outDir}")
            outDir.mkdir(parents=True, exist_ok=True)
        
        # Track progress
        start_time = time.time()
        total_tiles = numTileX * numTileY
        tiles_processed = 0
        tiles_successful = 0
        
        # Process each tile individually
        for tx in range(numTileX):
            for ty in range(numTileY):
                print(f"Processing tile {tx}_{ty} ({tiles_processed+1}/{total_tiles})...")
                
                success = process_tile_directly(
                    vhFN, vvFN, outDir, tileSize, tileOverlap, tx, ty, memory_efficient
                )
                
                tiles_processed += 1
                if success:
                    tiles_successful += 1
                
                # Report progress every 10 tiles or at the end
                if tiles_processed % 10 == 0 or tiles_processed == total_tiles:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / tiles_processed
                    remaining = avg_time * (total_tiles - tiles_processed)
                    print(f"Progress: {tiles_processed}/{total_tiles} tiles processed "
                          f"({tiles_successful} successful, {elapsed:.1f}s elapsed, "
                          f"~{remaining:.1f}s remaining)")
        
        print(f"Tile processing complete: {tiles_successful}/{total_tiles} tiles created successfully")
        return tiles_successful > 0
    
    except Exception as e:
        print(f"Error processing image tiles: {e}")
        traceback.print_exc()
        return False

def main():
    #=========================
    # Config
    #=========================
    print("Starting tile creation process...")
    environment_path = Path("environment.yaml")
    print(f"Loading configuration from: {environment_path}")
    try:
        with open(str(environment_path), "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Print important configuration values
    foldCSV = Path(config["FOLD"]["FoldCSV"])
    print(f"Using fold CSV: {foldCSV}")
    
    try:
        foldDf = pd.read_csv(foldCSV)
        sceneList = foldDf['scene_id'].values
        print(f"Found {len(sceneList)} scenes to process")
    except Exception as e:
        print(f"Error reading fold CSV: {e}")
        sys.exit(1)
    
    # the xView3_SLC_GRD_correspondences DataFrame is the mapping that
    # allows you to iterate over the dataset and build paths
    xView3_SLC_GRD_correspondences_path = Path(
        config["xView3_SLC_GRD_correspondences_path"]
    )
    print(f"Using SLC/GRD correspondences file: {xView3_SLC_GRD_correspondences_path}")
    
    try:
        xView3_SLC_GRD_correspondences = pd.read_csv(
            str(xView3_SLC_GRD_correspondences_path)
        )
        xView3_SLC_GRD_correspondences = xView3_SLC_GRD_correspondences[
            xView3_SLC_GRD_correspondences["scene_id"].isin(sceneList)
        ]
        print(f"Found {len(xView3_SLC_GRD_correspondences)} matching scenes in correspondences file")
    except Exception as e:
        print(f"Error reading SLC/GRD correspondences: {e}")
        sys.exit(1)
    
    SARFish_root_directory = Path(config["SARFish_root_directory"])
    product_type = config["product_type"]
    tileSize = int(config["CREATE_TILE"]["TileSize"])
    tileOverlap = int(config["CREATE_TILE"]["TileOverlap"])
    tilePath = Path(config["CREATE_TILE"]["TilePath"])
    
    # Use memory efficient processing by default (can be overridden in config)
    memory_efficient = config.get("CREATE_TILE", {}).get("MemoryEfficient", True)
    
    print(f"SARFish root directory: {SARFish_root_directory}")
    print(f"Product type: {product_type}")
    print(f"Tile size: {tileSize}")
    print(f"Tile overlap: {tileOverlap}")
    print(f"Output tile path: {tilePath}")
    print(f"Memory efficient processing: {memory_efficient}")
    
    # Check if SARFish_root_directory exists
    if not SARFish_root_directory.exists():
        print(f"ERROR: SARFish root directory does not exist: {SARFish_root_directory}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not tilePath.exists():
        print(f"Creating tile output directory: {tilePath}")
        tilePath.mkdir(parents=True, exist_ok=True)
    
    # Flag to indicate if any tiles were created
    any_tiles_created = False
    
    for index, row in xView3_SLC_GRD_correspondences.iterrows():
        for swath_index in [1, 2, 3]:
            print(f"\nProcessing {row[f'{product_type}_product_identifier']}, swath {swath_index}")
            outDir = Path(tilePath, row[f"{product_type}_product_identifier"], f"swath{swath_index}")
            
            # Check if output directory already exists
            if outDir.exists() and len(list(outDir.glob("*.npy"))) > 0:
                print(f"Output directory already exists with tiles: {outDir}")
                print("Skipping because directory already has tiles.")
                continue
            
            measurement_directory = Path(
                SARFish_root_directory, product_type, row["DATA_PARTITION"],
                f"{row[f'{product_type}_product_identifier']}.SAFE", "measurement",
            )
            
            # Check if measurement directory exists
            if not measurement_directory.exists():
                print(f"ERROR: Measurement directory does not exist: {measurement_directory}")
                continue
            
            vh_FN = Path(
                measurement_directory, row[f"SLC_swath_{swath_index}_vh"]
            )
            vv_FN = Path(
                measurement_directory, row[f"SLC_swath_{swath_index}_vv"]
            )
            
            # Check if input files exist
            if not vh_FN.exists():
                print(f"ERROR: VH file does not exist: {vh_FN}")
                continue
            if not vv_FN.exists():
                print(f"ERROR: VV file does not exist: {vv_FN}")
                continue
            
            print(f"VH file: {vh_FN}")
            print(f"VV file: {vv_FN}")
            
            try:
                print("\nProcessing tiles using memory-efficient approach...")
                success = process_image_tiles(
                    vh_FN, vv_FN, outDir, tileSize, tileOverlap, memory_efficient
                )
                
                if success:
                    any_tiles_created = True
                    print(f"Successfully processed {row[f'{product_type}_product_identifier']}, swath {swath_index}")
                else:
                    print(f"Failed to process {row[f'{product_type}_product_identifier']}, swath {swath_index}")
            except Exception as e:
                print(f"ERROR processing {row[f'{product_type}_product_identifier']}, swath {swath_index}: {e}")
                traceback.print_exc()
    
    if not any_tiles_created:
        print("\nWARNING: No tiles were created! Check the error messages above.")
    else:
        print("\nTile creation process completed successfully.")

if __name__ == "__main__":
    main()