#!/usr/bin/env python3
"""
Label Comparison Tool for YOLO Format Object Detection

This tool compares ground truth YOLO format labels with predicted labels and provides:
1. File count comparison (missed detections)
2. Detection count analysis (multiple predictions per ground truth)
3. Classification accuracy assessment

YOLO format: class x_center y_center width height (normalized 0-1)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


def load_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Load YOLO format labels from a text file.

    Args:
        label_path: Path to the label file

    Returns:
        List of tuples (class_id, x_center, y_center, width, height)
    """
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append((class_id, x_center, y_center, width, height))
    return labels


def get_label_files(labels_dir: str, is_prediction_dir: bool = False) -> Dict[str, str]:
    """
    Get all label files from a directory.

    Args:
        labels_dir: Directory containing label files
        is_prediction_dir: If True, removes "_pred" suffix to match with ground truth stems

    Returns:
        Dictionary mapping base filename (without extension) to full path
    """
    label_files = {}
    labels_path = Path(labels_dir)

    if not labels_path.exists():
        print(f"Warning: Directory {labels_dir} does not exist")
        return label_files

    for label_file in labels_path.glob("*.txt"):
        base_name = label_file.stem

        # For prediction files, remove "_pred" suffix to match ground truth naming
        if is_prediction_dir and base_name.endswith("_pred"):
            base_name = base_name[:-5]  # Remove last 5 characters ("_pred")

        label_files[base_name] = str(label_file)

    return label_files


def calculate_iou(box1: Tuple[float, float, float, float],
                  box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two YOLO format bounding boxes.

    Args:
        box1: (x_center, y_center, width, height) in normalized coordinates
        box2: (x_center, y_center, width, height) in normalized coordinates

    Returns:
        IoU value between 0 and 1
    """
    x1_center, y1_center, w1, h1 = box1
    x2_center, y2_center, w2, h2 = box2

    # Convert to corner coordinates
    x1_min = x1_center - w1/2
    y1_min = y1_center - h1/2
    x1_max = x1_center + w1/2
    y1_max = y1_center + h1/2

    x2_min = x2_center - w2/2
    y2_min = y2_center - h2/2
    x2_max = x2_center + w2/2
    y2_max = y2_center + h2/2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compare_labels(gt_dir: str, pred_dir: str, iou_threshold: float = 0.5) -> Dict:
    """
    Compare ground truth and predicted labels.

    Args:
        gt_dir: Directory containing ground truth label files
        pred_dir: Directory containing predicted label files
        iou_threshold: IoU threshold for considering a detection as correct (must be strictly greater than)

    Returns:
        Dictionary containing comparison results
    """
    # Load all label files
    gt_files = get_label_files(gt_dir, is_prediction_dir=False)
    pred_files = get_label_files(pred_dir, is_prediction_dir=True)

    # Initialize counters
    results = {
        'gt_file_count': len(gt_files),
        'pred_file_count': len(pred_files),
        'missed_detections': 0,
        'total_gt_objects': 0,
        'total_pred_objects': 0,
        'correct_detections': 0,  # IoU > threshold AND correct class
        'incorrect_classifications': 0,  # IoU > threshold AND wrong class
        'false_positives': 0,
        'file_analysis': [],
        'class_breakdown': {}
    }

    # Files with ground truth but no predictions (missed detections)
    missed_files = set(gt_files.keys()) - set(pred_files.keys())
    results['missed_detections'] = len(missed_files)

    # Analyze each file that has both ground truth and predictions
    common_files = set(gt_files.keys()) & set(pred_files.keys())

    for file_name in common_files:
        gt_labels = load_yolo_labels(gt_files[file_name])
        pred_labels = load_yolo_labels(pred_files[file_name])

        results['total_gt_objects'] += len(gt_labels)
        results['total_pred_objects'] += len(pred_labels)

        file_result = {
            'filename': file_name,
            'gt_count': len(gt_labels),
            'pred_count': len(pred_labels),
            'correct_detections': 0,  # IoU > threshold AND correct class
            'incorrect_class': 0,     # IoU > threshold AND wrong class
            'matched_predictions': 0, # Any IoU > threshold
            'false_positives': 0
        }

        # Track which predictions have been matched to avoid double counting
        matched_pred_indices = set()

        # For each ground truth object, find best matching prediction
        for gt_class, gt_x, gt_y, gt_w, gt_h in gt_labels:
            if gt_class not in results['class_breakdown']:
                results['class_breakdown'][gt_class] = {
                    'gt_count': 0, 'correct': 0, 'incorrect': 0, 'missed': 0
                }
            results['class_breakdown'][gt_class]['gt_count'] += 1

            best_correct_iou = 0
            best_correct_idx = None
            best_incorrect_iou = 0
            best_incorrect_idx = None

            # Find best matching predictions - separate for correct and incorrect class
            for i, (pred_class, pred_x, pred_y, pred_w, pred_h) in enumerate(pred_labels):
                iou = calculate_iou((gt_x, gt_y, gt_w, gt_h),
                                  (pred_x, pred_y, pred_w, pred_h))

                if pred_class == gt_class and iou > best_correct_iou:
                    best_correct_iou = iou
                    best_correct_idx = i
                elif pred_class != gt_class and iou > best_incorrect_iou:
                    best_incorrect_iou = iou
                    best_incorrect_idx = i

            # Check for correct detection (IoU > threshold AND correct class)
            if best_correct_iou > iou_threshold:
                file_result['matched_predictions'] += 1
                file_result['correct_detections'] += 1
                results['correct_detections'] += 1
                results['class_breakdown'][gt_class]['correct'] += 1
                matched_pred_indices.add(best_correct_idx)
            # Check for incorrect classification with sufficient IoU (only if no correct match)
            elif best_incorrect_iou > iou_threshold:
                file_result['matched_predictions'] += 1
                file_result['incorrect_class'] += 1
                results['incorrect_classifications'] += 1
                results['class_breakdown'][gt_class]['incorrect'] += 1
                matched_pred_indices.add(best_incorrect_idx)
            else:
                # No valid match found
                results['class_breakdown'][gt_class]['missed'] += 1

        # Count false positives: predictions that don't match any ground truth
        for i, (pred_class, pred_x, pred_y, pred_w, pred_h) in enumerate(pred_labels):
            if i not in matched_pred_indices:
                # This prediction doesn't match any ground truth
                file_result['false_positives'] += 1
                results['false_positives'] += 1

        results['file_analysis'].append(file_result)

    # Count objects in missed files
    for file_name in missed_files:
        gt_labels = load_yolo_labels(gt_files[file_name])
        results['total_gt_objects'] += len(gt_labels)
        for gt_class, _, _, _, _ in gt_labels:
            if gt_class not in results['class_breakdown']:
                results['class_breakdown'][gt_class] = {
                    'gt_count': 0, 'correct': 0, 'incorrect': 0, 'missed': 0
                }
            results['class_breakdown'][gt_class]['gt_count'] += 1
            results['class_breakdown'][gt_class]['missed'] += 1

    return results


def print_comparison_report(results: Dict, iou_threshold: float = 0.5):
    """Print a simplified comparison report."""
    print("="*50)
    print("YOLO LABEL COMPARISON REPORT")
    print("="*50)

    print(f"\n1. FILE COUNT ANALYSIS:")
    print(f"   Ground Truth Files: {results['gt_file_count']}")
    print(f"   Prediction Files:   {results['pred_file_count']}")
    print(f"   Missed Detections:  {results['missed_detections']} files")

    print(f"\n2. CORRECT DETECTION RATE:")
    if results['total_gt_objects'] > 0:
        correct_detection_rate = (results['correct_detections'] / results['total_gt_objects']) * 100
        print(f"   Overall: {results['correct_detections']}/{results['total_gt_objects']} ({correct_detection_rate:.1f}%)")
        print(f"   (IoU > {iou_threshold} AND correct classification)")
    else:
        print("   No ground truth objects found")

    print(f"\n3. CORRECT DETECTION RATE BY CLASS:")
    for class_id, stats in results['class_breakdown'].items():
        if stats['gt_count'] > 0:
            class_correct_rate = (stats['correct'] / stats['gt_count']) * 100
            print(f"   Class {class_id}: {stats['correct']}/{stats['gt_count']} ({class_correct_rate:.1f}%)")
        else:
            print(f"   Class {class_id}: 0/0 (0.0%)")

    print(f"\n4. FILES WITH MULTIPLE PREDICTIONS:")
    multi_pred_files = [f for f in results['file_analysis'] if f['pred_count'] > 1]
    if multi_pred_files:
        print(f"   {len(multi_pred_files)} files have multiple predictions:")
        for f in multi_pred_files:
            print(f"     {f['filename']}: {f['pred_count']} predictions")
    else:
        print("   No files with multiple predictions")


def main():
    parser = argparse.ArgumentParser(description='Compare YOLO format ground truth and predicted labels')
    parser.add_argument('gt_dir', help='Directory containing ground truth label files')
    parser.add_argument('pred_dir', help='Directory containing predicted label files')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching predictions to ground truth - must be strictly exceeded (default: 0.5)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed per-file analysis')

    args = parser.parse_args()

    if not os.path.exists(args.gt_dir):
        print(f"Error: Ground truth directory '{args.gt_dir}' does not exist")
        sys.exit(1)

    if not os.path.exists(args.pred_dir):
        print(f"Error: Predictions directory '{args.pred_dir}' does not exist")
        sys.exit(1)

    print(f"Comparing labels with IoU threshold: {args.iou_threshold}")
    print(f"Ground Truth: {args.gt_dir}")
    print(f"Predictions:  {args.pred_dir}")

    results = compare_labels(args.gt_dir, args.pred_dir, args.iou_threshold)
    print_comparison_report(results, args.iou_threshold)

    if args.verbose and results['file_analysis']:
        print(f"\n6. DETAILED PER-FILE ANALYSIS:")
        print(f"{'Filename':<30} {'GT':<3} {'Pred':<4} {'Match':<5} {'Correct':<7} {'Wrong':<5} {'FP':<3}")
        print("-" * 70)
        for f in results['file_analysis']:
            print(f"{f['filename']:<30} {f['gt_count']:<3} {f['pred_count']:<4} "
                  f"{f['matched_predictions']:<5} {f['correct_detections']:<7} {f['incorrect_class']:<5} {f['false_positives']:<3}")


if __name__ == "__main__":
    main()