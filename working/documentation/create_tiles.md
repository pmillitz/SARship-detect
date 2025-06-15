```python
#!/usr/bin/env python3

"""
This code combines VH,VV complex SLC images into tiled images for training/testing a 
detection/classification algorithm.

After running this script, it should create the following file structure:

[TilePath]/[scene_id]/swath1/0_0.npy    | This is the chip data (complex numpy array), 
                                        | the filename is tx_ty.npy, where tx is the tile index in x axis,
                                        | and ty is the tile index in y axis
                                        | Each .npy has 4 channels, vh_mag, vh_phase, vv_mag, vv_phase

[TilePath]/img_file_info.csv            | This file saved the file info of the scenes used
"""

# ---------------------------------------
# Importing Required Modules
# ---------------------------------------
from pathlib import Path  # For object-oriented filesystem path manipulations.
import yaml               # For reading YAML configuration files.
import numpy as np        # For numerical operations on arrays.
import pandas as pd       # For working with CSV files and dataframes.
import GeoTiff            # Custom module to load GeoTiff images

# ---------------------------------------
# Function: combineVHVV
# ---------------------------------------
def combineVHVV(vhFN: Path, vvFN: Path, tileSize: int, tileOverlap: int) -> np.ndarray:
    """
    Combines VH and VV complex SLC images into one 4-channel NumPy array.

    Steps:
      1. Reads the VH image and pads it on the right and bottom so that its dimensions 
         become divisible into tiles considering the specified overlap.
      2. Computes the magnitude and phase (using absolute value and angle) for the complex VH image.
      3. Repeats the same process for the VV image.
      4. Stacks these four results (vh magnitude, vh phase, vv magnitude, vv phase) into a single
         4-channel array.

    Parameters:
      vhFN (Path): File path to the VH complex image.
      vvFN (Path): File path to the VV complex image.
      tileSize (int): The desired size for each tile (in pixels).
      tileOverlap (int): The pixel overlap between adjacent tiles.

    Returns:
      np.ndarray: A 4 x H x W NumPy array, where the first axis represents the four channels.
    """
    # Load the VH image; the function returns a masked array along with other metadata (ignored here).
    (vhImg, _, _, _) = GeoTiff.load_GeoTiff(str(vhFN))

    # Retrieve the height and width of the VH image.
    imgH, imgW = vhImg.shape

    # Calculate the effective stride (tile size minus the overlap).
    s = tileSize - tileOverlap

    # Determine new dimensions (height and width) that allow the image to be evenly divided after accounting for the tile overlap.
    newH = int(np.ceil((imgH - tileOverlap) / s) * s) + tileOverlap
    newW = int(np.ceil((imgW - tileOverlap) / s) * s) + tileOverlap

    # Calculate how many pixels to add (pad) along height and width.
    padH = newH - imgH
    padW = newW - imgW

    # Pad the VH image using constant padding. This also converts the masked array into a regular complex array,
    # with padded (masked) values set to 0.
    padImg = np.pad(vhImg, pad_width=((0, padH), (0, padW)), mode="constant",
                    constant_values=0)

    # Compute the magnitude (absolute value) of the padded VH image.
    vhAbs = np.abs(padImg)
    # Compute the phase (angle) of the padded VH image.
    vhPhase = np.angle(padImg)

    # ----------------------------
    # Process VV Image in the Same Way
    # ----------------------------
    (vvImg, _, _, _) = GeoTiff.load_GeoTiff(str(vvFN))
    padImg = np.pad(vvImg, pad_width=((0, padH), (0, padW)), mode="constant",
                    constant_values=0)
    vvAbs = np.abs(padImg)
    vvPhase = np.angle(padImg)

    # ----------------------------
    # Create 4-Channel Combined Image
    # ----------------------------
    # Channels order: [VH magnitude, VH phase, VV magnitude, VV phase]
    out = np.stack([vhAbs, vhPhase, vvAbs, vvPhase], axis=0)
    return out

# ---------------------------------------
# Function: chopAndSaveTiles
# ---------------------------------------
def chopAndSaveTiles(combined: np.ndarray, outDir: Path, tileSize: int, tileOverlap: int):
    """
    Chops the combined 4-channel image (shape: 4 x H x W) into smaller tiles and saves each as a separate .npy file.

    The tiles are created by moving a sliding window across the image using a stride that accounts for the tile overlap.

    Parameters:
      combined (np.ndarray): The full combined image array from combineVHVV.
      outDir (Path): Output directory where the tile .npy files will be saved.
      tileSize (int): Size of each tile (in pixels).
      tileOverlap (int): Overlapping pixels between consecutive tiles.

    Each tile is saved with a filename in the form "tx_ty.npy", where:
      - tx: Tile's x-axis index.
      - ty: Tile's y-axis index.
    """
    # Calculate the effective stride (distance to jump between the top-left corners of adjacent tiles).
    s = tileSize - tileOverlap

    # Get the full height and width of the combined image.
    imgH = combined.shape[1]
    imgW = combined.shape[2]

    # Determine how many tiles fit horizontally and vertically.
    numTileX = int(imgW / s)
    numTileY = int(imgH / s)

    # Create the output directory if it doesn't exist already.
    if not outDir.exists():
        outDir.mkdir(parents=True, exist_ok=True)

    # Loop over each tile coordinate.
    for tx in range(numTileX):
        for ty in range(numTileY):
            # Calculate the starting (upper-left) x and y coordinates for the tile.
            x1 = tx * s
            y1 = ty * s

            # Calculate the ending (bottom-right) coordinates by adding the tile size.
            x2 = x1 + tileSize
            y2 = y1 + tileSize

            # Form the filename for this tile based on its position, e.g., "0_0.npy".
            FN = Path(outDir, f"{tx}_{ty}.npy")

            # Open the file in binary write mode and save the sliced tile using NumPy.
            with open(str(FN), "wb") as f:
                # Slice the combined array to extract the current tile (all channels, and spatial region by y and x).
                np.save(f, combined[:, y1:y2, x1:x2])

# ---------------------------------------
# Main Function: Workflow Entry Point
# ---------------------------------------
def main():
    # ==========================
    # Configuration Loading
    # ==========================

    # Define the path to the YAML configuration file.
    environment_path = Path("environment.yaml")
    # Read the YAML file to load configuration parameters into a dictionary.
    with open(str(environment_path), "r") as f:
        config = yaml.safe_load(f)

    # ==========================
    # Load Scene Fold Information
    # ==========================

    # Get the CSV file path for fold information from the configuration.
    foldCSV = Path(config["FOLD"]["FoldCSV"])
    # Read the CSV into a Pandas DataFrame.
    foldDf = pd.read_csv(foldCSV)
    # Extract an array of scene IDs that need to be processed.
    sceneList = foldDf['scene_id'].values
    # Define a list of swaths (different viewing angles or segments) to process.
    swathList = [1, 2, 3]

    # ==========================
    # Load and Filter the Data Correspondence Mapping
    # ==========================

    # Read the CSV that maps SLC to GRD images, allowing iteration through the dataset.
    xView3_SLC_GRD_correspondences_path = Path(config["xView3_SLC_GRD_correspondences_path"])
    xView3_SLC_GRD_correspondences = pd.read_csv(str(xView3_SLC_GRD_correspondences_path))
    # Filter the mapping to only include scenes from our fold list.
    xView3_SLC_GRD_correspondences = xView3_SLC_GRD_correspondences[
        xView3_SLC_GRD_correspondences["scene_id"].isin(sceneList)
    ]

    # ==========================
    # Set Up Directories and Tile Parameters
    # ==========================

    # Define the root directory where the SARFish data is stored.
    SARFish_root_directory = Path(config["SARFish_root_directory"])
    # Retrieve the product type (often used as a folder name or identifier).
    product_type = Path(config["product_type"])
    # Extract tile size and tile overlap from the configuration (convert to integers).
    tileSize = int(config["CREATE_TILE"]["TileSize"])
    tileOverlap = int(config["CREATE_TILE"]["TileOverlap"])
    # Define the output path where the tiled images will be saved.
    tilePath = Path(config["CREATE_TILE"]["TilePath"])
    # Ensure the output directory exists.
    if not tilePath.exists():
        tilePath.mkdir(parents=True, exist_ok=True)

    # ==========================
    # Process Each Scene and Swath
    # ==========================

    # Iterate over each scene (row in the mapping DataFrame).
    for index, row in xView3_SLC_GRD_correspondences.iterrows():
        # Process each swath (1, 2, and 3).
        for swath_index in [1, 2, 3]:
            # Log the current processing task.
            print(f"tiling {row[f'{product_type}_product_identifier']}, swath {swath_index}")

            # Build the output directory path for the current scene and swath.
            outDir = Path(tilePath, row[f"{product_type}_product_identifier"], f"swath{swath_index}")
            # If this output directory already exists, skip to avoid duplicate processing.
            if outDir.exists():
                continue

            # ----------------------------
            # Construct the Path to the Measurement Data
            # ----------------------------
            # Combine various configuration entries and DataFrame values to build the measurement directory path.
            measurement_directory = Path(
                SARFish_root_directory,
                product_type,
                row["DATA_PARTITION"],
                f"{row[f'{product_type}_product_identifier']}.SAFE",
                "measurement",
            )
            # Construct file paths for the VH and VV images for the current swath.
            vh_FN = Path(measurement_directory, row[f"SLC_swath_{swath_index}_vh"])
            vv_FN = Path(measurement_directory, row[f"SLC_swath_{swath_index}_vv"])

            # ----------------------------
            # Process the Complex Images
            # ----------------------------
            # Combine VH and VV images into a 4-channel NumPy array via magnitude and phase calculations.
            combined = combineVHVV(vh_FN, vv_FN, tileSize, tileOverlap)
            # Chop the combined image into smaller tiles and save each tile as a separate .npy file.
            chopAndSaveTiles(combined, outDir, tileSize, tileOverlap)

# ---------------------------------------
# Standard Boilerplate to Run the Script
# ---------------------------------------
if __name__ == "__main__":
    main()
```

# SAR Tile Creation Tool Documentation

## Purpose

This tool processes Synthetic Aperture Radar (SAR) data by combining VH (Vertical-Horizontal) and VV (Vertical-Vertical) complex Single Look Complex (SLC) images into tiles for machine learning applications in detection and classification algorithms.

## Output Structure

After running this script, it creates the following file structure:

```
[TilePath]/[scene_id]/swath[1-3]/[tx]_[ty].npy  # Tile data files
[TilePath]/img_file_info.csv                    # Scene file information
```

Where:

- `[tx]_[ty].npy`: Numpy array files containing the tile data
  - `tx`: tile index on x-axis
  - `ty`: tile index on y-axis
  - Each file contains 4 channels: vh_magnitude, vh_phase, vv_magnitude, vv_phase
- `swath[1-3]`: SAR swath index (1, 2, or 3)

## Core Functions

### `combineVHVV(vhFN, vvFN, tileSize, tileOverlap)`

Processes VH and VV image files to create a 4-channel array for tiling.

**Parameters:**

- `vhFN` (Path): Path to the VH polarization GeoTiff file
- `vvFN` (Path): Path to the VV polarization GeoTiff file
- `tileSize` (int): Size of each square tile in pixels
- `tileOverlap` (int): Overlap between adjacent tiles in pixels

**Process:**

1. Loads VH/VV images using GeoTiff loader
2. Pads images to ensure they're divisible by the effective tile stride (tileSize - tileOverlap)
3. Extracts magnitude (absolute value) and phase (angle) information from complex data
4. Stacks these 4 components (vh_magnitude, vh_phase, vv_magnitude, vv_phase) into a single array

**Returns:**

- 4-channel numpy array with shape (4, padded_height, padded_width)

### `chopAndSaveTiles(combined, outDir, tileSize, tileOverlap)`

Divides the combined 4-channel image into tiles and saves them as numpy files.

**Parameters:**

- `combined` (numpy.ndarray): 4-channel array from `combineVHVV`
- `outDir` (Path): Directory to save the tiles
- `tileSize` (int): Size of each square tile in pixels
- `tileOverlap` (int): Overlap between adjacent tiles in pixels

**Process:**

1. Calculates the effective stride between tiles (tileSize - tileOverlap)
2. Determines the number of tiles in x and y directions
3. Creates the output directory if it doesn't exist
4. Iterates through tile positions, extracting and saving each tile as a separate .npy file

### `main()`

Orchestrates the entire tiling process based on configuration.

**Process:**

1. Loads configuration from `environment.yaml`
2. Reads the fold CSV to determine which scenes to process
3. Loads the SLC/GRD correspondence data to find file paths
4. For each scene and swath:
   - Constructs input and output paths
   - Skips processing if output directory already exists
   - Calls `combineVHVV` to prepare the 4-channel data
   - Calls `chopAndSaveTiles` to divide and save the data

## Configuration Parameters

The script uses several configuration parameters from `environment.yaml`:

- `FOLD.FoldCSV`: CSV file containing scene IDs to process
- `xView3_SLC_GRD_correspondences_path`: Path to file mapping between SLC and GRD products
- `SARFish_root_directory`: Root directory containing SAR data
- `product_type`: Type of product to process (typically "SLC")
- `CREATE_TILE.TileSize`: Size of each tile in pixels
- `CREATE_TILE.TileOverlap`: Overlap between adjacent tiles in pixels   
- `CREATE_TILE.TilePath`: Output directory for tiles

## Technical Details

### Tiling Strategy

The script uses an overlapping tiling strategy to ensure features near tile boundaries are captured properly. The effective stride between tiles is `tileSize - tileOverlap`.

### Data Representation

- SAR data is complex-valued (having real and imaginary components)
- The script converts this to magnitude (intensity) and phase components
- Final representation: 4 channels [vh_magnitude, vh_phase, vv_magnitude, vv_phase]

### Padding Approach

Images are padded on the right and bottom edges to ensure they're divisible by the effective tile stride, preserving original image coordinates for any existing annotations.

## Usage Example

```python
# Configuration in environment.yaml
CREATE_TILE:
  TileSize: 512
  TileOverlap: 32
  TilePath: /path/to/output/tiles

# Run the script
python3 create_tiles.py
```
