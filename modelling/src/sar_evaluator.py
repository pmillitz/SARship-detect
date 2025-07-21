# sar_evaluator.py
# Comprehensive evaluation for float32 SAR models

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics.utils.ops import non_max_suppression

class SARFloat32Evaluator:
    """Comprehensive evaluator for YOLO models on float32 SAR data"""
    
    def __init__(self, model_path, data_yaml_path, device='cuda'):
        """
        Args:
            model_path: Path to saved model checkpoint (best.pt)
            data_yaml_path: Path to data configuration yaml
            device: Device to run evaluation on
        """
        import yaml
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load data configuration
        with open(data_yaml_path, 'r') as f:
            self.data_dict = yaml.safe_load(f)
        
        # Load model from checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Recreate model architecture (you'll need to match your trainer's setup)
        from ultralytics import YOLO
        base_model = YOLO('yolov8n.pt')  # Or whichever base you used
        self.model = base_model.model
        
        # Load trained weights
        if 'model_state_dict' in self.checkpoint:
            # Handle wrapped model state dict
            state_dict = self.checkpoint['model_state_dict']
            # Remove 'model.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v  # Remove 'model.' prefix
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict, strict=False)
        else:
            self.model.load_state_dict(self.checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = self.data_dict.get('names', ['vessel'])
        self.num_classes = len(self.class_names)
        
    def evaluate_dataset(self, dataset, batch_size=32, conf_thres=0.25, 
                        iou_thres=0.45, save_dir=None):
        """
        Comprehensive evaluation on a dataset
        
        Args:
            dataset: SARPreprocessedDataset instance
            batch_size: Batch size for evaluation
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS and matching
            save_dir: Directory to save results
        """
        from sar_dataset import SARPreprocessedDataset
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize metrics storage
        all_predictions = []
        all_ground_truths = []
        all_confidences = []
        inference_times = []
        
        print(f"Evaluating on {len(dataset)} images...")
        
        with torch.no_grad():
            for batch_idx, (images, batch_dict) in enumerate(tqdm(dataloader, desc='Evaluating')):
                images = images.to(self.device)
                
                # Time inference
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                outputs = self.model(images)
                end_time.record()
                
                # Wait for GPU sync
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                inference_times.append(inference_time / images.shape[0])  # Per image
                
                # Apply NMS
                predictions = non_max_suppression(
                    outputs,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    max_det=300
                )
                
                # Process each image in batch
                for i in range(images.shape[0]):
                    # Get predictions for this image
                    pred = predictions[i]
                    if pred is not None and len(pred) > 0:
                        # Convert to numpy [x1, y1, x2, y2, conf, class]
                        pred_np = pred.cpu().numpy()
                        all_predictions.append(pred_np)
                        all_confidences.extend(pred_np[:, 4].tolist())
                    else:
                        all_predictions.append(np.zeros((0, 6)))
                    
                    # Get ground truth for this image
                    mask = batch_dict['batch_idx'] == i
                    if mask.any():
                        gt_classes = batch_dict['cls'][mask].cpu().numpy()
                        gt_bboxes = batch_dict['bboxes'][mask].cpu().numpy()
                        
                        # Convert normalized xywh to xyxy
                        gt_xyxy = self._xywh2xyxy(gt_bboxes, images.shape[3], images.shape[2])
                        gt = np.column_stack([gt_xyxy, gt_classes])
                        all_ground_truths.append(gt)
                    else:
                        all_ground_truths.append(np.zeros((0, 5)))
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(
            all_predictions, 
            all_ground_truths, 
            iou_thres
        )
        
        # Add timing statistics
        metrics['inference_stats'] = {
            'mean_time_ms': np.mean(inference_times),
            'std_time_ms': np.std(inference_times),
            'min_time_ms': np.min(inference_times),
            'max_time_ms': np.max(inference_times)
        }
        
        # Add confidence distribution
        metrics['confidence_distribution'] = {
            'mean': np.mean(all_confidences) if all_confidences else 0,
            'std': np.std(all_confidences) if all_confidences else 0,
            'min': np.min(all_confidences) if all_confidences else 0,
            'max': np.max(all_confidences) if all_confidences else 0
        }
        
        # Generate plots and save results
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(metrics, all_confidences, save_dir)
        
        return metrics
    
    def _calculate_metrics(self, predictions, ground_truths, iou_threshold):
        """Calculate comprehensive detection metrics"""
        
        # Initialize per-class metrics
        tp = np.zeros(self.num_classes)
        fp = np.zeros(self.num_classes)
        fn = np.zeros(self.num_classes)
        
        # For AP calculation
        all_scores = [[] for _ in range(self.num_classes)]
        all_matched = [[] for _ in range(self.num_classes)]
        
        # Process each image
        for pred, gt in zip(predictions, ground_truths):
            if len(pred) == 0 and len(gt) == 0:
                continue
            
            # Match predictions to ground truth
            matches, unmatched_preds, unmatched_gts = self._match_detections(
                pred, gt, iou_threshold
            )
            
            # Process matches
            for match in matches:
                pred_idx, gt_idx, iou = match
                pred_class = int(pred[pred_idx, 5])
                gt_class = int(gt[gt_idx, 4])
                
                if pred_class == gt_class:
                    tp[pred_class] += 1
                    all_scores[pred_class].append(pred[pred_idx, 4])
                    all_matched[pred_class].append(True)
                else:
                    # Class mismatch
                    fp[pred_class] += 1
                    fn[gt_class] += 1
                    all_scores[pred_class].append(pred[pred_idx, 4])
                    all_matched[pred_class].append(False)
            
            # Process unmatched predictions (false positives)
            for pred_idx in unmatched_preds:
                pred_class = int(pred[pred_idx, 5])
                if pred_class < self.num_classes:
                    fp[pred_class] += 1
                    all_scores[pred_class].append(pred[pred_idx, 4])
                    all_matched[pred_class].append(False)
            
            # Process unmatched ground truths (false negatives)
            for gt_idx in unmatched_gts:
                gt_class = int(gt[gt_idx, 4])
                if gt_class < self.num_classes:
                    fn[gt_class] += 1
        
        # Calculate metrics
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        # Calculate AP for each class
        ap = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            if len(all_scores[c]) > 0:
                ap[c] = self._calculate_ap(
                    np.array(all_scores[c]), 
                    np.array(all_matched[c]),
                    tp[c] + fn[c]
                )
        
        # Overall metrics
        mAP = np.mean(ap)
        
        # Compile results
        metrics = {
            'mAP': mAP,
            'mAP50': mAP,  # Since we used single threshold
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'ap': ap.tolist(),
            'tp': tp.tolist(),
            'fp': fp.tolist(),
            'fn': fn.tolist(),
            'class_names': self.class_names,
            'total_detections': int(np.sum(tp + fp)),
            'total_ground_truths': int(np.sum(tp + fn)),
            'per_class_metrics': {}
        }
        
        # Add per-class breakdown
        for i, class_name in enumerate(self.class_names):
            metrics['per_class_metrics'][class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'ap': ap[i],
                'support': int(tp[i] + fn[i])
            }
        
        return metrics
    
    def _match_detections(self, predictions, ground_truths, iou_threshold):
        """Match predictions to ground truths using Hungarian algorithm"""
        matches = []
        unmatched_preds = set(range(len(predictions)))
        unmatched_gts = set(range(len(ground_truths)))
        
        if len(predictions) == 0 or len(ground_truths) == 0:
            return matches, unmatched_preds, unmatched_gts
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(predictions), len(ground_truths)))
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truths):
                iou_matrix[i, j] = self._calculate_iou(pred[:4], gt[:4])
        
        # Find matches using greedy approach (can upgrade to Hungarian if needed)
        while True:
            # Find best remaining match
            if len(unmatched_preds) == 0 or len(unmatched_gts) == 0:
                break
                
            max_iou = 0
            best_pred = None
            best_gt = None
            
            for i in unmatched_preds:
                for j in unmatched_gts:
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_pred = i
                        best_gt = j
            
            if max_iou >= iou_threshold:
                matches.append((best_pred, best_gt, max_iou))
                unmatched_preds.remove(best_pred)
                unmatched_gts.remove(best_gt)
            else:
                break
        
        return matches, unmatched_preds, unmatched_gts
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes in xyxy format"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _calculate_ap(self, scores, matched, num_gt):
        """Calculate Average Precision"""
        if num_gt == 0:
            return 0.0
        
        # Sort by confidence
        indices = np.argsort(-scores)
        scores = scores[indices]
        matched = matched[indices]
        
        # Calculate precision-recall curve
        tp_cumsum = np.cumsum(matched)
        fp_cumsum = np.cumsum(~matched)
        
        recall = tp_cumsum / num_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Calculate AP using all points
        ap = 0
        for i in range(len(recall)):
            if i == 0:
                ap += precision[i] * recall[i]
            else:
                ap += precision[i] * (recall[i] - recall[i-1])
        
        return ap
    
    def _xywh2xyxy(self, boxes, w, h):
        """Convert normalized xywh to xyxy pixel coordinates"""
        boxes_xyxy = boxes.copy()
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2]/2) * w  # x1
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3]/2) * h  # y1
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2]/2) * w  # x2
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3]/2) * h  # y2
        return boxes_xyxy
    
    def _collate_fn(self, batch):
        """Collate function matching trainer's implementation"""
        images, labels = zip(*batch)
        images = torch.stack(images, 0)
        
        batch_idx = []
        cls = []
        bboxes = []
        
        for i, label in enumerate(labels):
            if label.shape[0] > 0:
                batch_idx.append(torch.full((label.shape[0],), i, dtype=torch.long))
                cls.append(label[:, 0].long())
                bboxes.append(label[:, 1:5])
        
        if batch_idx:
            batch_dict = {
                'batch_idx': torch.cat(batch_idx, 0),
                'cls': torch.cat(cls, 0),
                'bboxes': torch.cat(bboxes, 0),
            }
        else:
            batch_dict = {
                'batch_idx': torch.zeros(0, dtype=torch.long),
                'cls': torch.zeros(0, dtype=torch.long),
                'bboxes': torch.zeros(0, 4),
            }
        
        return images, batch_dict
    
    def _save_results(self, metrics, confidences, save_dir):
        """Save evaluation results and plots"""
        
        # Save metrics as JSON
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Precision-Recall bar chart
        ax = axes[0, 0]
        x = np.arange(len(self.class_names))
        width = 0.35
        ax.bar(x - width/2, metrics['precision'], width, label='Precision', alpha=0.8)
        ax.bar(x + width/2, metrics['recall'], width, label='Recall', alpha=0.8)
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # 2. F1 Score
        ax = axes[0, 1]
        ax.bar(self.class_names, metrics['f1_score'], alpha=0.8, color='green')
        ax.set_xlabel('Class')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score by Class')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # 3. Confidence distribution
        ax = axes[1, 0]
        if confidences:
            ax.hist(confidences, bins=50, alpha=0.8, edgecolor='black')
            ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(confidences):.3f}')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Count')
            ax.set_title('Detection Confidence Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Detection statistics
        ax = axes[1, 1]
        stats_text = f"""Overall Statistics:
        
mAP@0.5: {metrics['mAP']:.4f}
Total Detections: {metrics['total_detections']}
Total Ground Truths: {metrics['total_ground_truths']}

Inference Speed:
Mean: {metrics['inference_stats']['mean_time_ms']:.2f} ms
Std: {metrics['inference_stats']['std_time_ms']:.2f} ms

Per-class AP:
"""
        for i, class_name in enumerate(self.class_names):
            stats_text += f"{class_name}: {metrics['ap'][i]:.4f}\n"
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='center',
                fontfamily='monospace')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save confusion matrix if multi-class
        if self.num_classes > 1:
            self._plot_confusion_matrix(metrics, save_dir)
        
        print(f"\nResults saved to: {save_dir}")
        print(f"- metrics.json: Detailed metrics")
        print(f"- evaluation_results.png: Visualization")
    
    def _plot_confusion_matrix(self, metrics, save_dir):
        """Plot confusion matrix for multi-class problems"""
        # This is simplified - you'd need to track class confusions during matching
        pass
    
    def evaluate_single_image(self, image_path, conf_thres=0.25, visualize=True):
        """Evaluate and visualize single image"""
        import cv2
        
        # Load image
        if image_path.endswith('.npy'):
            im = np.load(image_path)  # (3, H, W)
            im_input = torch.from_numpy(im).unsqueeze(0).float()
            # For visualization, convert to uint8
            im_vis = np.transpose(im, (1, 2, 0))
            im_vis = (im_vis * 255).clip(0, 255).astype(np.uint8)
        else:
            # Regular image
            im_vis = cv2.imread(str(image_path))
            im_input = torch.from_numpy(im_vis).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(im_input.to(self.device))
            predictions = non_max_suppression(outputs, conf_thres, iou_thres=0.45)
        
        pred = predictions[0]
        
        if visualize and pred is not None:
            for det in pred:
                x1, y1, x2, y2, conf, cls = det
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw box
                cv2.rectangle(im_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{self.class_names[int(cls)]}: {conf:.2f}"
                cv2.putText(im_vis, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return pred, im_vis
        
        return pred, None


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = SARFloat32Evaluator(
        model_path='runs/train/best.pt',
        data_yaml_path='configs/sar_data.yaml',
        device='cuda'
    )
    
    # Evaluate on test dataset
    from sar_dataset import SARPreprocessedDataset
    
    test_dataset = SARPreprocessedDataset(
        image_dir='data/test/images',
        label_dir='data/test/labels',
        imgsz=640
    )
    
    # Run comprehensive evaluation
    metrics = evaluator.evaluate_dataset(
        test_dataset,
        batch_size=32,
        conf_thres=0.25,
        iou_thres=0.45,
        save_dir='runs/evaluation/test_results'
    )
    
    print(f"\nFinal mAP@0.5: {metrics['mAP']:.4f}")