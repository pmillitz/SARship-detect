# compare_test_inference_results.py
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

def create_sar_prediction_comparison(model, val_images_path, num_images=5, labels_path=None, save_path=None, **kwargs):
    """
    Create comparison plots of ground truth vs predictions on SAR amplitude

    Args:
        model: YOLO model for inference
        val_images_path: Path to validation/test images
        num_images: Number of images to display (max 5 with labels, max 10 without)
        labels_path: Path to ground truth label files (None for inference-only mode)
        save_path: File path to save the plot (None to display only)
        **kwargs: Additional arguments passed to YOLO model.predict() (e.g., conf=0.5, iou=0.4, save=True)
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

    # Adjust max images based on whether labels are available
    if labels_path is not None:
        max_images = min(num_images, 5)
        rows, cols = 2, max_images
        figsize = (4*max_images, 8)
    else:
        max_images = min(num_images, 10)
        rows, cols = 2, 5
        figsize = (20, 8)

    selected_images = random.sample(image_files, min(max_images, len(image_files)))
    print(f"Found {len(image_files)} images. Displaying {len(selected_images)} randomly selected images:\n")

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle single image case
    if len(selected_images) == 1 and labels_path is not None:
        axes = axes.reshape(-1, 1)

    for i, img_path in enumerate(selected_images):
        # Calculate subplot position
        if labels_path is not None:
            row_gt, col = 0, i
            row_pred, col_pred = 1, i
        else:
            # For inference mode, use 2x5 grid
            row_gt = i // 5
            col = i % 5
            row_pred, col_pred = row_gt, col

        filename = Path(img_path).name
        formatted_filename = format_filename(filename)

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to load: {img_path}")
            continue

        # Extract VH amplitude (R channel) - BGR format, so R is index 2
        vh_amplitude = img[:, :, 2]  # BGR format, so R is index 2

        gt_boxes = []
        gt_classes = []

        # Look for ground truth labels if labels_path is provided
        if labels_path is not None:
            # Construct label file path
            img_filename = Path(img_path).stem
            label_file = Path(labels_path) / f"{img_filename}.txt"

            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            gt_boxes.append([x_center, y_center, width, height])
                            gt_classes.append(class_id)

        h, w = vh_amplitude.shape

        # Plot ground truth only if labels are available
        if labels_path is not None:
            axes[row_gt, col].imshow(vh_amplitude, cmap='gray')
            axes[row_gt, col].set_title(f'{formatted_filename}\n\nGround Truth',
                               fontsize=8, pad=10, ha='center')
            axes[row_gt, col].axis('off')

            # Draw ground truth boxes
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
                axes[row_gt, col].add_patch(rect)

                class_name = model.names.get(cls_id, f'Class_{cls_id}')
                axes[row_gt, col].text(x1, y1-5, f'{class_name}',
                              color=color, fontsize=8, weight='bold',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

        # Plot predictions
        axes[row_pred, col_pred].imshow(vh_amplitude, cmap='gray')
        if labels_path is not None:
            axes[row_pred, col_pred].set_title('Prediction', fontsize=8, pad=10, ha='center')
        else:
            axes[row_pred, col_pred].set_title(f'{formatted_filename}\n\nPrediction', fontsize=8, pad=10, ha='center')
        axes[row_pred, col_pred].axis('off')

        # Get predictions
        try:
            # Always set verbose=False, but allow other kwargs to be passed through
            inference_kwargs = {'verbose': False, **kwargs}
            results = model(img_path, **inference_kwargs)

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
                    axes[row_pred, col_pred].add_patch(rect)

                    axes[row_pred, col_pred].text(x1, y1-5, f'{class_name} ({conf:.2f})',
                                  color=color, fontsize=8, weight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
            else:
                axes[row_pred, col_pred].text(w//2, h//2, 'No detections',
                              ha='center', va='center', color='white', fontsize=12,
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='tomato', alpha=0.5))

        except Exception as e:
            print(f"Error with predictions: {e}")
            axes[row_pred, col_pred].text(w//2, h//2, 'Prediction Error',
                          ha='center', va='center', color='white', fontsize=12,
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='tomato', alpha=0.5))

    # Hide unused subplots
    if labels_path is None:
        # Hide unused subplots in 2x5 grid
        for idx in range(len(selected_images), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)

    plt.tight_layout()

    if labels_path is not None:
        plt.subplots_adjust(top=0.75, bottom=0.08, hspace=0.25)
        fig.suptitle('Sample Model Predictions', fontsize=14, fontweight='bold', y=0.98)
        fig.text(0.5, 0.01, 'Figure: Random sample of model predictions. Image with ground truth label and bounding box (top row); image with predicted label and bounding box with prediction confidence (bottom row).',
                 ha='center', va='bottom', fontsize=10, wrap=True)
    else:
        plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.25)
        fig.suptitle('Model Inference Results', fontsize=14, fontweight='bold', y=0.95)
        fig.text(0.5, 0.01, 'Figure: Random sample of model inference results. Images with predicted labels and bounding boxes with prediction confidence.',
                 ha='center', va='bottom', fontsize=10, wrap=True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()