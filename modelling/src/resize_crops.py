import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def resize_crops(crops_dir, output_dir, target_size, method='bilinear'):
    """
    Resize all crop images to consistent size (may distort aspect ratio)
    
    Args:
        crops_dir: Directory containing crop images
        output_dir: Directory to save resized images
        target_size: Tuple (width, height) for target size
        method: Resize method ('bilinear', 'cubic', 'nearest', 'area', 'lanczos')
    """
    crops_dir = Path(crops_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_w, target_h = target_size
    
    # OpenCV interpolation methods
    interp_methods = {
        'bilinear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    interp = interp_methods.get(method, cv2.INTER_LINEAR)
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(crops_dir.glob(f"*{ext}")) + list(crops_dir.glob(f"*{ext.upper()}")))
    
    print(f"Resizing {len(image_files)} images to {target_w}x{target_h} using {method} interpolation...")
    
    for img_path in tqdm(image_files, desc="Resizing images", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Resize image
        resized = cv2.resize(img, (target_w, target_h), interpolation=interp)
        
        # Save resized image
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), resized)
    
    print(f"Resizing complete! Images saved to: {output_dir}")

def resize_with_aspect_ratio_preserved(crops_dir, output_dir, target_size, method='bilinear', padding_color=(0,0,0)):
    """
    Resize maintaining aspect ratio with padding
    Perfect for SAR vessel data where preserving proportions is critical
    Uses black padding (0,0,0) to mimic ocean background in SAR imagery
    
    Args:
        crops_dir: Directory containing crop images
        output_dir: Directory to save resized images
        target_size: Tuple (width, height) for target size
        method: Resize method ('bilinear', 'cubic', 'nearest', 'area', 'lanczos')
        padding_color: Color for padding (default black for SAR ocean background)
    """
    crops_dir = Path(crops_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_w, target_h = target_size
    
    # OpenCV interpolation methods
    interp_methods = {
        'bilinear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    interp = interp_methods.get(method, cv2.INTER_LINEAR)
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(crops_dir.glob(f"*{ext}")) + list(crops_dir.glob(f"*{ext.upper()}")))
    
    print(f"Resizing {len(image_files)} images to {target_w}x{target_h} with aspect ratio preservation...")
    print(f"Using {method} interpolation with padding color: {padding_color}")
    
    for img_path in tqdm(image_files, desc="Resizing with aspect preservation", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Calculate scaling factor to fit within target dimensions
        scale = min(target_w/w, target_h/h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
        # Create canvas with padding
        canvas = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
        
        # Calculate position to center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image on canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Save result
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), canvas)
    
    print(f"Resizing complete! Images saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Resize crop images for CNN input")
    parser.add_argument("--crops", required=True, help="Directory containing crop images")
    parser.add_argument("--output", required=True, help="Output directory for resized images")
    parser.add_argument("--size", nargs=2, type=int, metavar=('W', 'H'), required=True, help="Target size (width height)")
    parser.add_argument("--method", default="bilinear", choices=['bilinear', 'cubic', 'nearest', 'area', 'lanczos'], help="Interpolation method (default: bilinear)")
    parser.add_argument("--preserve-aspect", action="store_true", help="Preserve aspect ratio with black padding (recommended for SAR vessels)")
    parser.add_argument("--padding-color", nargs=3, type=int, default=[0, 0, 0], metavar=('R', 'G', 'B'), help="Padding color RGB values (default: 0 0 0 for black)")
    
    args = parser.parse_args()
    
    target_size = tuple(args.size)
    padding_color = tuple(args.padding_color)
    
    print(f"Resizing crops from: {args.crops}")
    print(f"Output directory: {args.output}")
    print(f"Target size: {target_size[0]} x {target_size[1]}")
    print(f"Interpolation: {args.method}")
    
    if args.preserve_aspect:
        print(f"Mode: Aspect ratio preserving with padding color {padding_color}")
        resize_with_aspect_ratio_preserved(
            args.crops, 
            args.output, 
            target_size, 
            args.method, 
            padding_color
        )
    else:
        print("Mode: Direct resize (may distort aspect ratio)")
        resize_crops(args.crops, args.output, target_size, args.method)

if __name__ == "__main__":
    main()

# Example usage for SAR vessel length prediction:
# 
# Recommended approach (preserves vessel proportions):
# python resize_crops.py --crops path/to/crops --output path/to/resized --size 96 96 --preserve-aspect --method bilinear
#
# Alternative sizes for different upscaling factors:
# python resize_crops.py --crops path/to/crops --output path/to/resized --size 64 64 --preserve-aspect  # Minimal upscaling
# python resize_crops.py --crops path/to/crops --output path/to/resized --size 128 128 --preserve-aspect  # More CNN capacity
#
# For SAR data, bilinear interpolation with black padding is recommended to:
# - Preserve SAR amplitude characteristics 
# - Maintain vessel aspect ratios critical for length prediction
# - Simulate realistic ocean background with black padding