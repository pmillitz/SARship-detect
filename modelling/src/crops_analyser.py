import os
import cv2
import numpy as np
from pathlib import Path

def analyse_crops(crops_dir):
    """Simple crop analysis with debug output"""
    
    print(f"Analysing crops in: {crops_dir}")
    
    # Convert to Path object
    crops_path = Path(crops_dir)
    
    # Check if directory exists
    if not crops_path.exists():
        print(f"ERROR: Directory doesn't exist: {crops_path}")
        return
        
    if not crops_path.is_dir():
        print(f"ERROR: Not a directory: {crops_path}")
        return
    
    print(f"Directory exists: ✓")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for ext in image_extensions:
        files = list(crops_path.glob(f"*{ext}")) + list(crops_path.glob(f"*{ext.upper()}"))
        image_files.extend(files)
    
    print(f"Found {len(image_files)} image files")
    
    if len(image_files) == 0:
        print("No image files found!")
        print("Files in directory:")
        for f in crops_path.iterdir():
            print(f"  - {f.name}")
        return
    
    # Show first few filenames
    print("First few files:")
    for i, f in enumerate(image_files[:5]):
        print(f"  - {f.name}")
    
    # analyse dimensions
    widths = []
    heights = []
    
    print("\nAnalysing dimensions...")
    
    for i, img_file in enumerate(image_files):
        try:
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                if i < 5:  # Show first few
                    print(f"  {img_file.name}: {w} x {h}")
            else:
                print(f"  Could not read: {img_file.name}")
        except Exception as e:
            print(f"  Error reading {img_file.name}: {e}")
    
    if not widths:
        print("ERROR: Could not read any images!")
        return
    
    # Calculate stats
    print(f"\n{'='*50}")
    print(f"ANALYSIS RESULTS ({len(widths)} images)")
    print(f"{'='*50}")
    
    print(f"Width  - Min: {min(widths):3d}, Max: {max(widths):3d}, Mean: {np.mean(widths):6.1f}")
    print(f"Height - Min: {min(heights):3d}, Max: {max(heights):3d}, Mean: {np.mean(heights):6.1f}")
    
    # Show some specific examples
    print(f"\nSize distribution:")
    size_counts = {}
    for w, h in zip(widths, heights):
        size_key = f"{w}x{h}"
        size_counts[size_key] = size_counts.get(size_key, 0) + 1
    
    # Show most common sizes
    sorted_sizes = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)
    print("Most common sizes:")
    for size, count in sorted_sizes[:10]:
        print(f"  {size}: {count} images")
    
    # Recommendations for SAR vessel length prediction
    print(f"\n{'='*50}")
    print("RECOMMENDATIONS FOR SAR VESSEL LENGTH PREDICTION")
    print(f"{'='*50}")
    
    w_95 = int(np.percentile(widths, 95))
    h_95 = int(np.percentile(heights, 95))
    max_dim = max(max(widths), max(heights))
    
    print(f"95th percentile: {w_95} x {h_95}")
    print(f"Maximum dimensions: {max(widths)} x {max(heights)}")
    
    # Conservative target sizes
    conservative_targets = [
        (64, 64, "Minimal upscaling"),
        (96, 96, "Moderate upscaling, matches training resolution"),
        (128, 128, "Conservative CNN size")
    ]
    
    print(f"\nSuggested target sizes:")
    for w, h, desc in conservative_targets:
        max_upscale = max(w / max(widths), h / max(heights))
        print(f"  {w}x{h} - {desc} (max upscale: {max_upscale:.1f}x)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python simple_analyser.py /path/to/crops")
        sys.exit(1)
    
    crops_directory = sys.argv[1]
    analyse_crops(crops_directory)