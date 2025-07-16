#!/usr/bin/env python3

"""
spatial_processor.py

Spatial indexing utilities for efficient multi-object crop annotation.
Uses KD-trees and intelligent clustering to reduce redundant crops while
ensuring all annotations are captured.

Author: Peter Millitz
Created: 2025-07-15
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional


class SpatialCropProcessor:
    """Efficient spatial processing for multi-object crop annotation."""
    
    def __init__(self, crop_size: int, min_crop_distance: int = None):
        """
        Initialize the spatial processor.
        
        Args:
            crop_size: Size of square crops (e.g., 1024)
            min_crop_distance: Minimum distance between crop centers to prevent
                             redundancy (default: crop_size // 2)
        """
        self.crop_size = crop_size
        self.min_crop_distance = min_crop_distance or (crop_size // 2)
        
    def process_scene_annotations(self, 
                                  annotations_df: pd.DataFrame, 
                                  img_height: int, 
                                  img_width: int) -> List[Dict]:
        """
        Process all annotations for a single scene/image to determine optimal
        crop locations and annotation assignments.
        
        Args:
            annotations_df: DataFrame with annotations for one scene
            img_height, img_width: Image dimensions
            
        Returns:
            List of crop dictionaries with format:
            {
                'center': (row, col),
                'annotations': [list of annotation indices],
                'primary_annotation_idx': int (for naming the crop)
            }
        """
        if annotations_df.empty:
            return []
        
        # Reset index to ensure we have consistent indices
        annotations_df = annotations_df.reset_index(drop=False)
        
        # Step 1: Extract annotation centers and build spatial index
        centers = self._extract_annotation_centers(annotations_df)
        kdtree = cKDTree(centers)
        
        # Step 2: Determine crop locations using greedy clustering
        crop_locations = self._determine_crop_locations(
            centers, kdtree, img_height, img_width, annotations_df
        )
        
        # Step 3: Assign annotations to crops
        crops = self._assign_annotations_to_crops(
            annotations_df, crop_locations, kdtree
        )
        
        return crops
    
    def _extract_annotation_centers(self, annotations_df: pd.DataFrame) -> np.ndarray:
        """
        Extract center coordinates for all annotations.
        
        Args:
            annotations_df: DataFrame containing annotations
            
        Returns:
            numpy array of shape (n_annotations, 2) with [row, col] centers
        """
        centers = []
        
        for idx, ann in annotations_df.iterrows():
            # Priority: use detect_scene coordinates if available
            if "detect_scene_row" in ann and "detect_scene_column" in ann:
                center_row = int(ann["detect_scene_row"])
                center_col = int(ann["detect_scene_column"])
            else:
                # Fallback to bounding box center
                # Note: Y-axis is inverted in annotations (top > bottom)
                center_row = int((ann["top"] + ann["bottom"]) / 2)
                center_col = int((ann["left"] + ann["right"]) / 2)
            
            centers.append([center_row, center_col])
        
        return np.array(centers)
    
    def _determine_crop_locations(self, 
                                  centers: np.ndarray, 
                                  kdtree: cKDTree,
                                  img_height: int, 
                                  img_width: int,
                                  annotations_df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Determine optimal crop locations using greedy clustering.
        Ensures minimum distance between crops to reduce redundancy.
        
        Args:
            centers: Array of annotation centers
            kdtree: KD-tree built from centers
            img_height, img_width: Image dimensions
            annotations_df: Original annotations (for confidence sorting)
            
        Returns:
            List of (row, col) tuples for crop centers
        """
        n_annotations = len(centers)
        processed = np.zeros(n_annotations, dtype=bool)
        crop_locations = []
        
        # Sort by confidence if available (process high confidence first)
        confidence_order = self._get_confidence_order(annotations_df)
        
        for idx in confidence_order:
            if processed[idx]:
                continue
                
            center = centers[idx]
            
            # Check if this location is too close to existing crops
            too_close = False
            for existing_crop in crop_locations:
                dist = np.sqrt((center[0] - existing_crop[0])**2 + 
                              (center[1] - existing_crop[1])**2)
                if dist < self.min_crop_distance:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Check if crop would be too close to image edges
            half_size = self.crop_size // 2
            if (center[0] - half_size < -self.crop_size//4 or 
                center[0] + half_size > img_height + self.crop_size//4 or
                center[1] - half_size < -self.crop_size//4 or 
                center[1] + half_size > img_width + self.crop_size//4):
                # Skip crops that would be mostly padding
                continue
            
            # Valid crop location found
            crop_locations.append(tuple(center))
            
            # Mark all annotations within crop range as processed
            # This is the key optimization - prevents redundant crops
            nearby_indices = kdtree.query_ball_point(center, r=self.crop_size // 2)
            processed[nearby_indices] = True
        
        return crop_locations
    
    def _get_confidence_order(self, annotations_df: pd.DataFrame) -> List[int]:
        """
        Get processing order based on confidence levels.
        High confidence annotations are processed first.
        """
        if 'confidence' in annotations_df.columns:
            # Define confidence priority
            confidence_priority = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            priorities = annotations_df['confidence'].map(
                lambda x: confidence_priority.get(x, 3)
            ).values
            return np.argsort(priorities)
        else:
            # No confidence info - process in order
            return list(range(len(annotations_df)))
    
    def _assign_annotations_to_crops(self, 
                                     annotations_df: pd.DataFrame,
                                     crop_locations: List[Tuple[int, int]],
                                     kdtree: cKDTree) -> List[Dict]:
        """
        Assign all annotations to their respective crops.
        Each annotation can appear in multiple crops if they overlap.
        
        Args:
            annotations_df: DataFrame with annotations
            crop_locations: List of crop center coordinates
            kdtree: KD-tree of annotation centers
            
        Returns:
            List of crop dictionaries
        """
        crops = []
        
        for crop_center in crop_locations:
            half_size = self.crop_size // 2
            
            # Query for all points within the crop
            # Use slightly larger radius then filter precisely
            candidate_indices = kdtree.query_ball_point(
                crop_center, 
                r=half_size * np.sqrt(2),  # Diagonal distance
                p=np.inf  # L-infinity norm for square region
            )
            
            # Filter to ensure annotations are truly within crop bounds
            annotations_in_crop = []
            for idx in candidate_indices:
                ann_center = kdtree.data[idx]
                if (abs(ann_center[0] - crop_center[0]) <= half_size and
                    abs(ann_center[1] - crop_center[1]) <= half_size):
                    # Get the original index from the dataframe
                    original_idx = annotations_df.iloc[idx]['index']
                    annotations_in_crop.append(original_idx)
            
            if annotations_in_crop:
                # Determine primary annotation (closest to crop center)
                distances = []
                for ann_idx in annotations_in_crop:
                    # Find position in our working dataframe
                    pos = annotations_df[annotations_df['index'] == ann_idx].index[0]
                    dist = np.linalg.norm(kdtree.data[pos] - crop_center)
                    distances.append(dist)
                
                primary_idx = annotations_in_crop[np.argmin(distances)]
                
                crops.append({
                    'center': crop_center,
                    'annotations': annotations_in_crop,
                    'primary_annotation_idx': primary_idx
                })
        
        return crops
    
    def get_crop_statistics(self, crops: List[Dict]) -> Dict:
        """
        Calculate statistics about the crop distribution.
        
        Args:
            crops: List of crop dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not crops:
            return {
                'num_crops': 0,
                'total_annotations': 0,
                'avg_annotations_per_crop': 0,
                'max_annotations_per_crop': 0,
                'min_annotations_per_crop': 0
            }
        
        annotations_per_crop = [len(crop['annotations']) for crop in crops]
        
        return {
            'num_crops': len(crops),
            'total_annotations': sum(annotations_per_crop),
            'avg_annotations_per_crop': np.mean(annotations_per_crop),
            'max_annotations_per_crop': max(annotations_per_crop),
            'min_annotations_per_crop': min(annotations_per_crop),
            'std_annotations_per_crop': np.std(annotations_per_crop)
        }


def process_scene_with_spatial_indexing(
    scene_annotations: pd.DataFrame,
    img_array: np.ndarray,
    crop_size: int,
    out_img_dir: Path,
    out_lbl_dir: Path,
    swath_idx: int,
    saved_filenames: Set[str],
    spatial_processor: SpatialCropProcessor,
    logger=None,
    create_padded_crop=None  # Function from main module
) -> Dict[str, int]:
    """
    Process all annotations for a scene using spatial indexing.
    Creates crops with multiple annotations per label file.
    
    This is the main integration function that bridges the spatial processor
    with the existing create_crop.py infrastructure.
    
    Args:
        scene_annotations: DataFrame with annotations for this scene
        img_array: The image array
        crop_size: Size of crops to create
        out_img_dir: Output directory for image files
        out_lbl_dir: Output directory for label files
        swath_idx: Swath index for filename generation
        saved_filenames: Set to track saved filenames
        spatial_processor: SpatialCropProcessor instance
        logger: Logger instance
        create_padded_crop: Function from main module for creating crops
        
    Returns:
        Dictionary with processing statistics
    """
    if create_padded_crop is None:
        raise ValueError("create_padded_crop function must be provided")
    
    # Get optimal crop locations and assignments
    crops = spatial_processor.process_scene_annotations(
        scene_annotations,
        img_array.shape[0],
        img_array.shape[1]
    )
    
    if logger:
        stats = spatial_processor.get_crop_statistics(crops)
        logger.print(f"  Spatial analysis: {stats['num_crops']} crops for "
                    f"{len(scene_annotations)} annotations "
                    f"(avg {stats['avg_annotations_per_crop']:.1f} annotations/crop)")
    
    stats = {
        'crops_created': 0,
        'annotations_processed': 0,
        'annotations_skipped': 0,
        'crops_padded': 0
    }
    
    for crop_info in crops:
        crop_center = crop_info['center']
        annotation_indices = crop_info['annotations']
        primary_idx = crop_info['primary_annotation_idx']
        
        # Use primary annotation for filename
        primary_ann = scene_annotations.loc[primary_idx]
        filename_base = f"{primary_ann['detect_id']}_swath{swath_idx}"
        
        if filename_base in saved_filenames:
            continue
        saved_filenames.add(filename_base)
        
        # Create the crop using the main module's function
        crop, metadata = create_padded_crop(
            img_array, crop_center[0], crop_center[1], crop_size, pad_value=0
        )
        
        if metadata['padding_applied']:
            stats['crops_padded'] += 1
        
        # Process all annotations for YOLO labels
        yolo_labels = []
        actual_top, actual_bottom, actual_left, actual_right = metadata['actual_image_region']
        pad_top, pad_bottom, pad_left, pad_right = metadata['padding_amounts']
        
        annotations_included = 0
        for ann_idx in annotation_indices:
            ann = scene_annotations.loc[ann_idx]
            
            # Transform bounding box to crop coordinates
            # Remember: annotations use inverted Y-axis
            box_left_in_crop = float(ann["left"]) - actual_left + pad_left
            box_right_in_crop = float(ann["right"]) - actual_left + pad_left
            box_top_in_crop = float(ann["bottom"]) - actual_top + pad_top  # inverted
            box_bottom_in_crop = float(ann["top"]) - actual_top + pad_top  # inverted
            
            # Apply shrinking if needed
            bbox_outside_crop = (box_left_in_crop < 0 or box_top_in_crop < 0 or
                               box_right_in_crop > crop_size or box_bottom_in_crop > crop_size)
            
            if bbox_outside_crop:
                shrunk_left = max(0, box_left_in_crop)
                shrunk_top = max(0, box_top_in_crop)
                shrunk_right = min(crop_size, box_right_in_crop)
                shrunk_bottom = min(crop_size, box_bottom_in_crop)
                
                # Check shrinking limits
                if (abs(box_left_in_crop - shrunk_left) > 5 or
                    abs(box_top_in_crop - shrunk_top) > 5 or
                    abs(shrunk_right - box_right_in_crop) > 5 or
                    abs(shrunk_bottom - box_bottom_in_crop) > 5):
                    stats['annotations_skipped'] += 1
                    if logger:
                        logger.print(f"    Warning: Skipped annotation {ann['detect_id']} - "
                                   f"bounding box extends too far outside crop")
                    continue
                
                box_left_in_crop = shrunk_left
                box_top_in_crop = shrunk_top
                box_right_in_crop = shrunk_right
                box_bottom_in_crop = shrunk_bottom
            
            # Convert to YOLO format
            xc = (box_left_in_crop + box_right_in_crop) / 2 / crop_size
            yc = (box_top_in_crop + box_bottom_in_crop) / 2 / crop_size
            w = (box_right_in_crop - box_left_in_crop) / crop_size
            h = (box_bottom_in_crop - box_top_in_crop) / crop_size
            
            # Determine class
            class_id = 1 if pd.notna(ann.get("is_fishing")) and ann["is_fishing"] else 0
            
            # Validate and add label
            if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                yolo_labels.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                annotations_included += 1
            else:
                if logger:
                    logger.print(f"    Warning: Invalid YOLO coordinates for {ann['detect_id']}")
                stats['annotations_skipped'] += 1
        
        # Save crop and labels if we have valid annotations
        if yolo_labels:
            image_path = out_img_dir / f"{filename_base}.npy"
            label_path = out_lbl_dir / f"{filename_base}.txt"
            
            try:
                np.save(image_path, crop)
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_labels) + "\n")
                
                if logger:
                    status_str = " [PADDED]" if metadata['padding_applied'] else ""
                    logger.print(f"    Saved: {filename_base} "
                               f"[{annotations_included} annotations]{status_str}")
                
                stats['crops_created'] += 1
                stats['annotations_processed'] += annotations_included
                
            except Exception as e:
                if logger:
                    logger.print(f"    Error saving {filename_base}: {e}")
    
    return stats
