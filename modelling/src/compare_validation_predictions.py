# compare_validation_predictions.py
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
from pathlib import Path
from ultralytics import YOLO

def format_filename(filename, max_chars_per_line=25):
    """
    Format filename by wrapping at logical breakpoints
    """
    # Remove .png extension
    name = filename
    
    # Split at underscores
    parts = name.split('_')
    
    # Group parts to fit within reasonable line lengths
    lines = []
    current_line = ""
    
    for i, part in enumerate(parts):
        # Add underscore back except for first part
        part_with_underscore = part if i == 0 else '_' + part
        
        # Check if adding this part would make line too long
        if len(current_line + part_with_underscore) > max_chars_per_line and current_line:
            lines.append(current_line)
            current_line = part_with_underscore
        else:
            current_line += part_with_underscore
    
    # Add remaining part
    if current_line:
        lines.append(current_line)
    
    return '\n'.join(lines)

def create_sar_prediction_comparison(model, val_images_path, num_images=6):
    """
    Create comparison plots of ground truth vs predictions on SAR amplitude
    """
    # Define colors for each class
    class_colors = {
        0: 'tomato',
        1: 'lime'
    }
    
    # Get validation images
    image_files = glob.glob(f"{val_images_path}/*.png")
    if not image_files:
        print(f"No PNG images found in {val_images_path}")
        image_files = glob.glob(f"{val_images_path}/*.jpg")
        if not image_files:
            print(f"No JPG images found either in {val_images_path}")
            return
    
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    print(f"Found {len(image_files)} images. Displaying {num_images} randomly selected images:\n")
    
    fig, axes = plt.subplots(2, len(selected_images), figsize=(4*len(selected_images), 8))
    if len(selected_images) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, img_path in enumerate(selected_images):
        filename = Path(img_path).name
        formatted_filename = format_filename(filename)
        
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to load: {img_path}")
            continue
            
        # Extract amplitude (R channel)
        amplitude = img[:, :, 2]  # BGR format, so R is index 2
        
        # Look for ground truth labels
        label_path = str(Path(img_path).with_suffix('.txt'))
        possible_label_paths = [
            label_path.replace('/images/', '/labels/'),
            label_path.replace('\\images\\', '\\labels\\'),
            str(Path(img_path).parent.parent / 'labels' / Path(img_path).with_suffix('.txt').name)
        ]
        
        gt_boxes = []
        gt_classes = []
        
        for label_path in possible_label_paths:
            if Path(label_path).exists():
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            gt_boxes.append([x_center, y_center, width, height])
                            gt_classes.append(class_id)
                break
        
        # Plot ground truth with filename
        axes[0, i].imshow(amplitude, cmap='gray')
        axes[0, i].set_title(f'{formatted_filename}\n\nGround Truth', 
                           fontsize=8, pad=10, ha='center')
        axes[0, i].axis('off')
        
        # Draw ground truth boxes
        h, w = amplitude.shape
        for bbox, cls_id in zip(gt_boxes, gt_classes):
            x_center, y_center, width_norm, height_norm = bbox
            x1 = int((x_center - width_norm/2) * w)
            y1 = int((y_center - height_norm/2) * h)
            x2 = int((x_center + width_norm/2) * w)
            y2 = int((y_center + height_norm/2) * h)
            
            # Use class-specific color
            color = class_colors.get(cls_id, 'yellow')  # Default to yellow if class not defined
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color=color, linewidth=2)
            axes[0, i].add_patch(rect)
            
            class_name = model.names.get(cls_id, f'Class_{cls_id}')
            axes[0, i].text(x1, y1-5, f'{class_name}', 
                          color=color, fontsize=8, weight='bold',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
        
        # Plot predictions with just "Prediction" title
        axes[1, i].imshow(amplitude, cmap='gray')
        axes[1, i].set_title('Prediction', fontsize=8, pad=10, ha='center')
        axes[1, i].axis('off')
        
        # Get predictions
        try:
            results = model(img_path, verbose=False)
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = model.names.get(int(cls), f'Class_{int(cls)}')
                    
                    # Use class-specific color
                    color = class_colors.get(int(cls), 'yellow')
                    
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, color=color, linewidth=2)
                    axes[1, i].add_patch(rect)
                    
                    axes[1, i].text(x1, y1-5, f'{class_name} ({conf:.2f})', 
                                  color=color, fontsize=8, weight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
            else:
                axes[1, i].text(w//2, h//2, 'No detections', 
                              ha='center', va='center', color='white', fontsize=12,
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='tomato', alpha=0.5))
                
        except Exception as e:
            print(f"Error with predictions: {e}")
            axes[1, i].text(w//2, h//2, 'Prediction Error', 
                          ha='center', va='center', color='white', fontsize=12,
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='tomato', alpha=0.5))
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.75, bottom=0.08, hspace=0.25)  # Adjusted spacing for super title and caption
    
    # Add super title
    fig.suptitle('Sample Model Predictions', fontsize=14, fontweight='bold', y=0.98)
    
    # Add figure caption
    fig.text(0.5, 0.01, 'Figure: Random sample of model predictions. Image with ground truth label and bounding box (top row); image with predicted label and bounding box with prediction confidence (bottom row).', 
             ha='center', va='bottom', fontsize=10, wrap=True)
    
    plt.savefig('sar_amplitude_predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
