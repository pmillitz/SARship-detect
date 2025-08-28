import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random

class CompleteVesselEvaluator:
    """
    Complete evaluator that properly handles:
    - True Positives (TP): Correct detections with correct class
    - False Positives (FP): Detections with no matching ground truth
    - False Negatives (FN): Ground truth boxes with no matching detection
    - Class Confusion: Correct detection but wrong class
    """
    
    def __init__(self, model_path, class_names=['is_vessel', 'is_fishing']):
        """Initialize evaluator with trained model."""
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def extract_complete_predictions(self, image_dir, label_dir,
                                    batch_size=32,
                                    conf_threshold=0.25,
                                    iou_threshold=0.5,
                                    sample_size=None,
                                    show_progress=True):
        """
        Extract ALL predictions including FP and FN.
        
        Returns:
            dict with complete detection statistics
        """
        # Initialize counters for the complete confusion matrix
        # This will be (num_classes + 1) x (num_classes + 1) where +1 is for "background"
        # Rows = Ground Truth, Columns = Predictions
        detection_matrix = np.zeros((self.num_classes + 1, self.num_classes + 1))
        
        # Track individual predictions for detailed analysis
        all_detections = []
        confidence_scores = {'TP': [], 'FP': []}
        
        # Get all image paths
        image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
        
        if sample_size and len(image_paths) > sample_size:
            image_paths = random.sample(image_paths, sample_size)
            print(f"Sampling {sample_size} images from total dataset")
        
        print(f"Processing {len(image_paths)} images in batches of {batch_size}")
        
        if show_progress:
            pbar = tqdm(total=len(image_paths), desc="Evaluating")
        
        # Process in batches
        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            
            # Run inference on batch
            batch_results = self.model(batch_paths, conf=conf_threshold, verbose=False)
            
            # Process each image result
            for img_path, results in zip(batch_paths, batch_results):
                # Get ground truth boxes
                label_path = Path(label_dir) / f"{img_path.stem}.txt"
                gt_boxes = []
                
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if class_id < self.num_classes:  # Valid class
                                    bbox = [float(x) for x in parts[1:5]]
                                    gt_boxes.append({
                                        'class': class_id,
                                        'bbox': bbox,
                                        'matched': False
                                    })
                
                # Get prediction boxes
                pred_boxes = []
                if results.boxes is not None and len(results.boxes) > 0:
                    for box in results.boxes:
                        class_id = int(box.cls.item())
                        if class_id < self.num_classes:  # Valid class
                            conf = float(box.conf.item())
                            
                            # Convert to normalized xywh
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            img_h, img_w = results.orig_shape
                            
                            x_center = ((x1 + x2) / 2) / img_w
                            y_center = ((y1 + y2) / 2) / img_h
                            width = (x2 - x1) / img_w
                            height = (y2 - y1) / img_h
                            
                            pred_boxes.append({
                                'class': class_id,
                                'conf': conf,
                                'bbox': [x_center, y_center, width, height],
                                'matched': False
                            })
                
                # Match predictions to ground truth
                for pred in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for i, gt in enumerate(gt_boxes):
                        if not gt['matched']:  # Only match unmatched GT boxes
                            iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                            if iou > best_iou and iou > iou_threshold:
                                best_iou = iou
                                best_gt_idx = i
                    
                    if best_gt_idx >= 0:
                        # TRUE POSITIVE - Detection matched a ground truth
                        gt = gt_boxes[best_gt_idx]
                        gt['matched'] = True
                        pred['matched'] = True
                        
                        # Add to detection matrix (GT class vs Predicted class)
                        detection_matrix[gt['class'], pred['class']] += 1
                        
                        # Track as TP with confidence
                        confidence_scores['TP'].append(pred['conf'])
                        
                        # Store detection details
                        all_detections.append({
                            'type': 'TP',
                            'gt_class': gt['class'],
                            'pred_class': pred['class'],
                            'confidence': pred['conf'],
                            'iou': best_iou,
                            'correct_class': gt['class'] == pred['class']
                        })
                    else:
                        # FALSE POSITIVE - No matching ground truth
                        detection_matrix[self.num_classes, pred['class']] += 1  # Background -> Predicted
                        
                        confidence_scores['FP'].append(pred['conf'])
                        
                        all_detections.append({
                            'type': 'FP',
                            'gt_class': None,
                            'pred_class': pred['class'],
                            'confidence': pred['conf'],
                            'iou': 0,
                            'correct_class': False
                        })
                
                # Check for FALSE NEGATIVES (unmatched ground truth)
                for gt in gt_boxes:
                    if not gt['matched']:
                        # Ground truth with no matching prediction
                        detection_matrix[gt['class'], self.num_classes] += 1  # GT -> Background (not detected)
                        
                        all_detections.append({
                            'type': 'FN',
                            'gt_class': gt['class'],
                            'pred_class': None,
                            'confidence': 0,
                            'iou': 0,
                            'correct_class': False
                        })
                
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        return {
            'detection_matrix': detection_matrix,
            'all_detections': all_detections,
            'confidence_scores': confidence_scores
        }
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes in YOLO format."""
        x1_1 = box1[0] - box1[2]/2
        y1_1 = box1[1] - box1[3]/2
        x2_1 = box1[0] + box1[2]/2
        y2_1 = box1[1] + box1[3]/2
        
        x1_2 = box2[0] - box2[2]/2
        y1_2 = box2[1] - box2[3]/2
        x2_2 = box2[0] + box2[2]/2
        y2_2 = box2[1] + box2[3]/2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_complete_metrics(self, detection_results):
        """
        Calculate comprehensive metrics including detection and classification performance.
        """
        matrix = detection_results['detection_matrix']
        detections = detection_results['all_detections']
        
        # Separate detection metrics for each class
        metrics = []
        
        for i, class_name in enumerate(self.class_names):
            # True Positives: Correctly detected (regardless of classification)
            tp_detected = matrix[i, :self.num_classes].sum()  # All detections of this GT class
            # True Positives with correct class
            tp_correct_class = matrix[i, i]
            # False Negatives: Missed detections
            fn = matrix[i, self.num_classes]
            # False Positives: Background detected as this class
            fp = matrix[self.num_classes, i]
            
            # Ground truth count
            gt_count = tp_detected + fn
            
            # Detection metrics (did we find the object?)
            detection_recall = tp_detected / gt_count if gt_count > 0 else 0
            detection_precision = tp_detected / (tp_detected + fp) if (tp_detected + fp) > 0 else 0
            
            # Classification metrics (did we classify correctly?)
            classification_accuracy = tp_correct_class / tp_detected if tp_detected > 0 else 0
            
            # Combined metrics (detection + correct classification)
            precision = tp_correct_class / (tp_correct_class + fp + (tp_detected - tp_correct_class)) if (tp_correct_class + fp + (tp_detected - tp_correct_class)) > 0 else 0
            recall = tp_correct_class / gt_count if gt_count > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.append({
                'Class': class_name,
                'GT_Count': int(gt_count),
                'Detections': int(tp_detected),
                'Correct_Class': int(tp_correct_class),
                'Missed (FN)': int(fn),
                'False_Pos': int(fp),
                'Detection_Recall': detection_recall,
                'Classification_Acc': classification_accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
        
        # Overall metrics
        total_tp = sum([m['Correct_Class'] for m in metrics])
        total_fp = sum([m['False_Pos'] for m in metrics])
        total_fn = sum([m['Missed (FN)'] for m in metrics])
        total_gt = sum([m['GT_Count'] for m in metrics])
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / total_gt if total_gt > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        metrics.append({
            'Class': 'OVERALL',
            'GT_Count': total_gt,
            'Detections': total_tp + sum([m['Detections'] - m['Correct_Class'] for m in metrics[:self.num_classes]]),
            'Correct_Class': total_tp,
            'Missed (FN)': total_fn,
            'False_Pos': total_fp,
            'Detection_Recall': (total_gt - total_fn) / total_gt if total_gt > 0 else 0,
            'Classification_Acc': total_tp / (total_tp + sum([m['Detections'] - m['Correct_Class'] for m in metrics[:self.num_classes]])) if total_tp > 0 else 0,
            'Precision': overall_precision,
            'Recall': overall_recall,
            'F1-Score': overall_f1
        })
        
        return pd.DataFrame(metrics)
    
    def plot_complete_confusion_matrix(self, detection_matrix, title="Complete Detection Matrix"):
        """
        Plot the complete confusion matrix including background (FP/FN).
        """
        # Create extended labels
        labels = self.class_names + ['Not Detected/Background']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Raw counts
        sns.heatmap(detection_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=labels, yticklabels=labels, ax=ax1,
                   cbar_kws={'label': 'Count'})
        ax1.set_title(f'{title} - Raw Counts', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        
        # Add lines to separate background
        ax1.axhline(y=self.num_classes, color='black', linewidth=2)
        ax1.axvline(x=self.num_classes, color='black', linewidth=2)
        
        # Normalized by row (per ground truth class)
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix_norm = detection_matrix / detection_matrix.sum(axis=1, keepdims=True)
            matrix_norm = np.nan_to_num(matrix_norm)
        
        sns.heatmap(matrix_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=labels, yticklabels=labels, ax=ax2,
                   cbar_kws={'label': 'Proportion'})
        ax2.set_title(f'{title} - Normalized by GT', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        
        # Add lines to separate background
        ax2.axhline(y=self.num_classes, color='black', linewidth=2)
        ax2.axvline(x=self.num_classes, color='black', linewidth=2)
        
        plt.tight_layout()
        return fig
    
    def analyze_errors(self, detection_results):
        """
        Detailed error analysis.
        """
        detections = pd.DataFrame(detection_results['all_detections'])
        
        print("\n" + "="*60)
        print("DETECTION ERROR ANALYSIS")
        print("="*60)
        
        # Detection type distribution
        print("\n📊 Detection Types:")
        print("-" * 40)
        type_counts = detections['type'].value_counts()
        for det_type, count in type_counts.items():
            pct = count / len(detections) * 100
            print(f"{det_type}: {count} ({pct:.1f}%)")
        
        # True Positive class accuracy
        tp_detections = detections[detections['type'] == 'TP']
        if len(tp_detections) > 0:
            print("\n✅ True Positive Classification:")
            print("-" * 40)
            correct_class = tp_detections['correct_class'].sum()
            total_tp = len(tp_detections)
            print(f"Correct classification: {correct_class}/{total_tp} ({correct_class/total_tp*100:.1f}%)")
            
            # Confusion within TPs
            for i, class_name in enumerate(self.class_names):
                class_tps = tp_detections[tp_detections['gt_class'] == i]
                if len(class_tps) > 0:
                    correct = class_tps['correct_class'].sum()
                    print(f"  {class_name}: {correct}/{len(class_tps)} correct ({correct/len(class_tps)*100:.1f}%)")
        
        # False Positive analysis
        fp_detections = detections[detections['type'] == 'FP']
        if len(fp_detections) > 0:
            print("\n❌ False Positives by Predicted Class:")
            print("-" * 40)
            for i, class_name in enumerate(self.class_names):
                class_fps = fp_detections[fp_detections['pred_class'] == i]
                if len(class_fps) > 0:
                    avg_conf = class_fps['confidence'].mean()
                    print(f"  {class_name}: {len(class_fps)} FPs (avg conf: {avg_conf:.3f})")
        
        # False Negative analysis
        fn_detections = detections[detections['type'] == 'FN']
        if len(fn_detections) > 0:
            print("\n⚠️ Missed Detections by Class:")
            print("-" * 40)
            for i, class_name in enumerate(self.class_names):
                class_fns = fn_detections[fn_detections['gt_class'] == i]
                if len(class_fns) > 0:
                    print(f"  {class_name}: {len(class_fns)} missed")
        
        # Confidence analysis
        if detection_results['confidence_scores']['TP']:
            print("\n📈 Confidence Score Analysis:")
            print("-" * 40)
            tp_confs = detection_results['confidence_scores']['TP']
            fp_confs = detection_results['confidence_scores']['FP']
            
            print(f"True Positives:  mean={np.mean(tp_confs):.3f}, std={np.std(tp_confs):.3f}")
            if fp_confs:
                print(f"False Positives: mean={np.mean(fp_confs):.3f}, std={np.std(fp_confs):.3f}")
    
    def evaluate_complete(self, train_imgs, train_labels, val_imgs, val_labels,
                         batch_size=32, conf_threshold=0.25, iou_threshold=0.5,
                         train_sample_size=1000, val_sample_size=None):
        """
        Complete evaluation including all detection outcomes.
        """
        print("="*60)
        print("COMPLETE DETECTION EVALUATION")
        print("="*60)
        
        # Training evaluation
        print("\n📊 Evaluating Training Set...")
        train_results = self.extract_complete_predictions(
            train_imgs, train_labels,
            batch_size=batch_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            sample_size=train_sample_size,
            show_progress=True
        )
        
        # Validation evaluation  
        print("\n📊 Evaluating Validation Set...")
        val_results = self.extract_complete_predictions(
            val_imgs, val_labels,
            batch_size=batch_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            sample_size=val_sample_size,
            show_progress=True
        )
        
        # Calculate metrics
        train_metrics = self.calculate_complete_metrics(train_results)
        val_metrics = self.calculate_complete_metrics(val_results)
        
        # Display metrics
        print("\n" + "="*50)
        print("TRAINING METRICS (Complete)")
        print("="*50)
        print(train_metrics.to_string(index=False))
        
        print("\n" + "="*50)
        print("VALIDATION METRICS (Complete)")  
        print("="*50)
        print(val_metrics.to_string(index=False))
        
        # Error analysis
        self.analyze_errors(train_results)
        self.analyze_errors(val_results)
        
        # Plot confusion matrices
        train_fig = self.plot_complete_confusion_matrix(
            train_results['detection_matrix'], 
            "Training Set"
        )
        val_fig = self.plot_complete_confusion_matrix(
            val_results['detection_matrix'],
            "Validation Set"
        )
        
        plt.show()
        
        return {
            'train_results': train_results,
            'val_results': val_results,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }


# Example usage
if __name__ == "__main__":
    evaluator = CompleteVesselEvaluator(
        model_path='runs/detect/train/weights/best.pt',
        class_names=['is_vessel', 'is_fishing']
    )
    
    results = evaluator.evaluate_complete(
        train_imgs='datasets/vessels/train/images',
        train_labels='datasets/vessels/train/labels',
        val_imgs='datasets/vessels/val/images',
        val_labels='datasets/vessels/val/labels',
        batch_size=32,
        conf_threshold=0.25,
        iou_threshold=0.5,
        train_sample_size=1000,  # Sample for speed
        val_sample_size=None     # Full validation
    )