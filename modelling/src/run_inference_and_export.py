import os
import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
from tqdm import tqdm

def run_inference_and_save_results(model_path, test_images_dir, output_dir, conf_threshold=0.25, iou_threshold=0.7, verbose=False):
    """
    Run YOLOv8 inference on test images and save results + cropped detections
    
    Args:
        model_path: Path to your trained YOLOv8 model (.pt file)
        test_images_dir: Directory containing test images
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for Non-Maximum Suppression
        verbose: If True, print detailed progress information
    """
    
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "detection_results.txt"
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Get list of test images
    test_images_dir = Path(test_images_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in test_images_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    if verbose:
        print(f"Using confidence threshold: {conf_threshold}")
        print(f"Using IoU threshold: {iou_threshold}")
        print(f"Output directory: {output_dir}")
        print(f"Crops directory: {crops_dir}")
    
    # Create progress bar for non-verbose mode
    if not verbose:
        pbar = tqdm(total=len(image_files), desc="Processing images", unit="img")
    
    # Open results file for writing
    with open(results_file, 'w') as f:
        # Write header
        f.write("image_name,class_id,class_name,confidence,x1,y1,x2,y2,width,height\n")
        
        # Process each image
        for img_idx, image_path in enumerate(image_files):
            if verbose:
                print(f"Processing {img_idx + 1}/{len(image_files)}: {image_path.name}")
            
            # Run inference
            results = model(str(image_path), verbose=False, conf=conf_threshold, iou=iou_threshold)
            
            # Load original image for cropping
            original_img = cv2.imread(str(image_path))
            img_height, img_width = original_img.shape[:2]
            
            # Process detections for this image
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = model.names[class_id]
                        
                        # Calculate width and height
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Write detection to results file
                        f.write(f"{image_path.name},{class_id},{class_name},{confidence:.4f},"
                               f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{width:.2f},{height:.2f}\n")
                        
                        if verbose:
                            print(f"  Detection {i+1}: {class_name} conf={confidence:.3f} size={width:.0f}x{height:.0f}")
                        
                        # Crop and save detection
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_width, x2)
                        y2 = min(img_height, y2)
                        
                        # Crop the detection
                        cropped_img = original_img[y1:y2, x1:x2]
                        
                        # Create crop filename
                        crop_filename = f"{image_path.stem}_{class_name}_{i}_conf{confidence:.3f}.png"
                        crop_path = crops_dir / crop_filename
                        
                        # Save cropped image
                        cv2.imwrite(str(crop_path), cropped_img)
                
                else:
                    # No detections found for this image
                    if verbose:
                        print(f"  No detections found")
                    f.write(f"{image_path.name},,,,,,,,,\n")
            
            # Update progress bar for non-verbose mode
            if not verbose:
                pbar.update(1)
    
    # Close progress bar
    if not verbose:
        pbar.close()
    
    print(f"\nInference completed!")
    print(f"Results saved to: {results_file}")
    print(f"Cropped images saved to: {crops_dir}")
    print(f"Total images processed: {len(image_files)}")

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference and save results")
    parser.add_argument("--model", required=True, help="Path to trained YOLOv8 model (.pt file)")
    parser.add_argument("--images", required=True, help="Directory containing test images")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS (default: 0.7)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    
    args = parser.parse_args()
    
    run_inference_and_save_results(
        model_path=args.model,
        test_images_dir=args.images,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()

# Example usage (if running directly):
# run_inference_and_save_results(
#     model_path="path/to/your/best.pt",
#     test_images_dir="path/to/test/images",
#     output_dir="path/to/output",
#     conf_threshold=0.25,
#     iou_threshold=0.7,
#     verbose=False
# )