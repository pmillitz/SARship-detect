#!/usr/bin/env python3
"""
Download products from the [SARFish dataset](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish) iteratively.
1. create python virtual environement:
```
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements.txt`
```
2. create hf access token
<https://huggingface.co/docs/huggingface_hub/quick-start#authentication>
3. login
```
huggingface-cli login --token <HF_TOKEN>
```
3. download products
```
./download_specific_files_from_the_SARFish_dataset.py --cache-dir <CACHE_DIR>
```
"""
import argparse
from pathlib import Path, PurePosixPath
from typing import List, Tuple
import huggingface_hub
from huggingface_hub import hf_hub_download
import numpy
import pandas


def pick_desired_subset_of_DataFrame(d: pandas.DataFrame) -> pandas.DataFrame:
    desired_scene_ids = [
        "00a035722196ee86t",
    ]
    return d[d["scene_id"].isin(desired_scene_ids)]


def get_product_path(row: pandas.Series, product_type: str) -> str:
    p = PurePosixPath(
        product_type,
        row["DATA_PARTITION"],
        f'{row[f"{product_type}_product_identifier"]}.SAFE.zip',
    )
    return str(p)


def get_paths_of_SARFish_products() -> Tuple[List[str], List[str]]:
    # Download correspondences file from GitHub
    correspondence_url = "https://raw.githubusercontent.com/DIUx-xView/SARFish/bfef9694946d192ce7f66c5c5f97d5be364de02c/reference/labels/xView3_SLC_GRD_correspondences.csv"
    print("Downloading correspondence file from GitHub...")
    SARFish_mapping: pandas.DataFrame = pandas.read_csv(correspondence_url)
    
    SARFish_mapping = pick_desired_subset_of_DataFrame(SARFish_mapping)
    GRD_product_paths = SARFish_mapping.apply(
        get_product_path, product_type="GRD", axis=1
    )
    SLC_product_paths = SARFish_mapping.apply(
        get_product_path, product_type="SLC", axis=1
    )
    return GRD_product_paths.to_list(), SLC_product_paths.to_list()


def download_specific_files_from_the_SARFish_dataset(cache_dir: Path) -> None:
    # [hf_hub_download API](https://huggingface.co/docs/huggingface_hub/v0.31.4/en/package_reference/file_download#huggingface_hub.hf_hub_download)
    import shutil
    
    dataset_repo = "ConnorLuckettDSTG/SARFish"
    GRD_product_paths, SLC_product_paths = get_paths_of_SARFish_products()
    
    for i, product_path in enumerate(GRD_product_paths + SLC_product_paths):
        proper_filename = Path(product_path).name
        target_path = cache_dir / proper_filename
        
        print(f"{i}: {proper_filename}")
        
        # Check if file already exists with proper name
        if target_path.exists():
            print(f"   File already exists: {proper_filename}")
            continue
            
        try:
            # Download to HF cache location
            downloaded_path = hf_hub_download(
                dataset_repo, 
                product_path, 
                repo_type="dataset", 
                cache_dir=cache_dir
            )
            
            downloaded_file = Path(downloaded_path)
            
            if downloaded_file.is_symlink():
                # Get the actual blob file
                actual_blob = downloaded_file.resolve()
                print(f"   Source blob: {actual_blob}")
                print(f"   Blob size: {actual_blob.stat().st_size / (1024**3):.2f} GB")
                
                # Copy the blob to target location
                shutil.copy2(actual_blob, target_path)
                print(f"   Copied to: {proper_filename}")
                
                # Delete the original blob file to free up space
                actual_blob.unlink()
                print(f"   Deleted source blob")
                
                # Remove the now-broken symlink
                downloaded_file.unlink()
                print(f"   Cleaned up symlink")
                
            else:
                # If it's already a regular file, just rename it
                downloaded_file.rename(target_path)
                print(f"   Renamed to: {proper_filename}")
                
        except (OSError, FileNotFoundError) as e:
            print(f"   Error: Could not process file - {e}")
            if 'downloaded_path' in locals():
                print(f"   File may still be accessible at: {downloaded_path}")
                
        # Print current target file size to verify
        if target_path.exists():
            print(f"   Final file size: {target_path.stat().st_size / (1024**3):.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="./")
    arguments = parser.parse_args()
    download_specific_files_from_the_SARFish_dataset(Path(arguments.cache_dir))


if __name__ == "__main__":
    main()
