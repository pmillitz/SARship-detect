#!/usr/bin/env python3

"""This code combines VH,VV complex SLC images into tiles images for the purpose of
   training/testing the detection/classification algorithm.

After running this script, it should create the following file structure:

[TilePath]/[scene_id]/swath1/0_0.npy    | This is the chip data (complex numpy array),
                                        | the filename is tx_ty.npy, where tx is the tile index in x axis,
                                        | and ty is the tile index in y axis
                                        | Each .npy has 4 channels, vh_mag, vh_phase, vv_mag, vv_phase

[TilePath]/img_file_info.csv            | This file saved the file info of the scenes used
"""
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import GeoTiff
import os
import sys

def combineVHVV(vhFN: Path, vvFN: Path, tileSize: int, tileOverlap: int) -> np.ndarray:
    """INPUT: the file name of VH and VV
    
    1 - Pad an image to make it divisible by some block_size with along with overlap
        Pad on the right and bottom edges so annotations are still usable.
    2 - Calculate the absolute and the phase, of the VH and VV, and place them into
        one 4 channel numpy array
        
    OUTPUT: numpy array of 4 x H x W
    """
    print(f"Loading VH image from: {vhFN}")
    try:
        # Read VH image, NOTE: img is a masked array
        (vhImg, _, _, _) = GeoTiff.load_GeoTiff(str(vhFN))
        imgH, imgW = vhImg.shape
        print(f"VH image loaded successfully. Shape: {imgH}x{imgW}")
    except Exception as e:
        print(f"Error loading VH image: {e}")
        raise

    s = tileSize - tileOverlap
    newH = int(np.ceil((imgH-tileOverlap) / s) * s) + tileOverlap
    newW = int(np.ceil((imgW-tileOverlap) / s) * s) + tileOverlap
    padH = newH - imgH
    padW = newW - imgW
    print(f"Padding image to: {newH}x{newW} (added {padH}x{padW} pixels)")

    #
    # This np.pad function also converts the masked array into a complex array. A
    # masked value becomes 0.
    padImg = np.pad(
        vhImg, pad_width=((0, padH), (0, padW)), mode="constant",
        constant_values=0
    )
    vhAbs = np.abs(padImg)
    vhPhase = np.angle(padImg)
    
    print(f"Loading VV image from: {vvFN}")
    try:
        # do the same for VV image
        (vvImg, _, _, _) = GeoTiff.load_GeoTiff(str(vvFN))
        print(f"VV image loaded successfully. Shape: {vvImg.shape}")
    except Exception as e:
        print(f"Error loading VV image: {e}")
        raise

    padImg = np.pad(
        vvImg, pad_width=((0, padH), (0, padW)), mode="constant",
        constant_values=0
    )
    vvAbs = np.abs(padImg)
    vvPhase = np.angle(padImg)
    
    # Create a 4 channel image of vhAbs, vhPhase, vvAbs, vvPhase
    out = np.stack([vhAbs, vhPhase, vvAbs, vvPhase], axis=0)
    print(f"Created combined 4-channel array with shape: {out.shape}")
    return out

def chopAndSaveTiles(combined: np.ndarray, outDir: Path, tileSize: int, tileOverlap: int):
    """Chop the combined image (4 x H x W) into tiles
    """
    s = tileSize - tileOverlap
    imgH = combined.shape[1]
    imgW = combined.shape[2]
    numTileX = int(imgW / s)
    numTileY = int(imgH / s)
    
    print(f"Creating {numTileX}x{numTileY} = {numTileX*numTileY} tiles with size {tileSize} and overlap {tileOverlap}")
    if not outDir.exists():
        print(f"Creating output directory: {outDir}")
        outDir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Output directory already exists: {outDir}")
    
    tiles_created = 0
    for tx in range(numTileX):
        for ty in range(numTileY):
            x1 = tx * s
            y1 = ty * s
            x2 = x1 + tileSize
            y2 = y1 + tileSize
            FN = Path(outDir, f"{tx}_{ty}.npy")
            try:
                with open(str(FN), "wb") as f:
                    np.save(f, combined[:, y1:y2, x1:x2])
                tiles_created += 1
            except Exception as e:
                print(f"Error saving tile {tx}_{ty}.npy: {e}")
    
    print(f"Successfully created {tiles_created} tiles in {outDir}")

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
    
    swathList = [1, 2, 3]
    
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
    
    print(f"SARFish root directory: {SARFish_root_directory}")
    print(f"Product type: {product_type}")
    print(f"Tile size: {tileSize}")
    print(f"Tile overlap: {tileOverlap}")
    print(f"Output tile path: {tilePath}")
    
    # Check if SARFish_root_directory exists
    if not SARFish_root_directory.exists():
        print(f"ERROR: SARFish root directory does not exist: {SARFish_root_directory}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not tilePath.exists():
        print(f"Creating tile output directory: {tilePath}")
        tilePath.mkdir(parents=True, exist_ok=True)
    
    # Flag to indicate if any tiles were created
    tiles_created = False
    
    for index, row in xView3_SLC_GRD_correspondences.iterrows():
        for swath_index in [1, 2, 3]:
            print(f"\nProcessing {row[f'{product_type}_product_identifier']}, swath {swath_index}")
            outDir = Path(tilePath, row[f"{product_type}_product_identifier"], f"swath{swath_index}")
            
            # Check if output directory already exists
            if outDir.exists():
                print(f"Output directory already exists: {outDir}")
                # Uncomment the following line to skip existing directories
                # print("Skipping because directory exists.")
                # continue
                print("Continuing anyway...")
            
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
                print("Combining VH and VV data...")
                combined = combineVHVV(vh_FN, vv_FN, tileSize, tileOverlap)
                print("Chopping and saving tiles...")
                chopAndSaveTiles(combined, outDir, tileSize, tileOverlap)
                tiles_created = True
                print(f"Successfully processed {row[f'{product_type}_product_identifier']}, swath {swath_index}")
            except Exception as e:
                print(f"ERROR processing {row[f'{product_type}_product_identifier']}, swath {swath_index}: {e}")
    
    if not tiles_created:
        print("\nWARNING: No tiles were created! Check the error messages above.")
    else:
        print("\nTile creation process completed successfully.")

if __name__ == "__main__":
    main()