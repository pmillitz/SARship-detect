# sar_png_augmentation.py - v1 (adapted from sar_slc_augmentation.py v18)
# Augmentation module for processed PNG SAR data (3-channel: VH-amp, VV-amp, PCF) with lazy loading

import numpy as np
import cv2
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import random
from scipy.ndimage import shift as ndimage_shift


class SARPNGAugmentation:
    """
    Augmentation module for processed PNG SAR data.
    Works with 3-channel float32 arrays (VH-amplitude, VV-amplitude, PCF).
    Optimized with lazy loading for better memory efficiency.
    """
    
    def __init__(self,
                 # Geometric augmentation probabilities by class
                 geometric_probs: Dict[int, Dict[str, float]] = None,
                 # SAR-specific augmentation probabilities by class  
                 sar_probs: Dict[int, Dict[str, float]] = None,
                 # Mosaic augmentation settings
                 mosaic_prob: Dict[int, float] = None,
                 # General settings
                 min_visibility: float = 0.3):
        """
        Args:
            geometric_probs: Per-class probabilities for geometric augmentations
                {0: {'hflip': 0.3, 'vflip': 0.3, 'rotate': 0.2, 'translate': 0.2},
                 1: {'hflip': 0.6, 'vflip': 0.6, 'rotate': 0.5, 'translate': 0.4}}
            sar_probs: Per-class probabilities for SAR-specific augmentations
                {0: {'amplitude_scale': 0.0, 'amplitude_speckle': 0.0, 'gaussian_filter': 0.0},
                 1: {'amplitude_scale': 0.1, 'amplitude_speckle': 0.1, 'gaussian_filter': 0.2}}
            mosaic_prob: Per-class probability of creating mosaic
                {0: 0.1, 1: 0.3}
            min_visibility: Minimum fraction of bbox that must remain visible
        """
        
        # Default probabilities if not provided
        if geometric_probs is None:
            geometric_probs = {
                0: {'hflip': 0.3, 'vflip': 0.3, 'rotate': 0.2, 'translate': 0.2},
                1: {'hflip': 0.6, 'vflip': 0.6, 'rotate': 0.5, 'translate': 0.4}
            }
        
        if sar_probs is None:
            # SAR augmentations default to 0 probability for amplitude data
            sar_probs = {
                0: {'amplitude_scale': 0.0, 'amplitude_speckle': 0.0, 'gaussian_filter': 0.0},
                1: {'amplitude_scale': 0.1, 'amplitude_speckle': 0.1, 'gaussian_filter': 0.2}
            }
            
        if mosaic_prob is None:
            mosaic_prob = {0: 0.1, 1: 0.3}
        
        self.geometric_probs = geometric_probs
        self.sar_probs = sar_probs
        self.mosaic_prob = mosaic_prob
        self.min_visibility = min_visibility
        
        # Initialize augmentation functions
        self._init_augmentations()
        
        # Statistics tracking
        self.stats = {
            'processed': 0,
            'augmented': 0,
            'by_class': {0: 0, 1: 0},
            'augmentation_counts': {}
        }
        
    def _init_augmentations(self):
        """Initialize augmentation functions for complex data."""
        # Geometric transforms that also update bounding boxes
        self.geometric_transforms = {
            'hflip': self._create_complex_hflip(),
            'vflip': self._create_complex_vflip(),
            'rotate': self._create_complex_rotate(),
            'translate': self._create_complex_translate()
        }
        
        # SAR-specific transforms (amplitude domain, no bbox changes)
        self.sar_transforms = {
            'amplitude_scale': self._create_amplitude_scale(),
            'amplitude_speckle': self._create_amplitude_speckle(),
            'gaussian_filter': self._create_amplitude_gaussian_filter()
        }
    
    def _create_complex_hflip(self):
        """Create horizontal flip transform for 3-channel data."""
        def transform(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            flipped_data = np.fliplr(data)
            flipped_labels = labels.copy()
            
            if len(flipped_labels) > 0:
                # Flip x_center coordinates
                flipped_labels[:, 1] = 1.0 - flipped_labels[:, 1]
            
            return flipped_data, flipped_labels
        return transform
    
    def _create_complex_vflip(self):
        """Create vertical flip transform for 3-channel data."""
        def transform(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            flipped_data = np.flipud(data)
            flipped_labels = labels.copy()
            
            if len(flipped_labels) > 0:
                # Flip y_center coordinates
                flipped_labels[:, 2] = 1.0 - flipped_labels[:, 2]
            
            return flipped_data, flipped_labels
        return transform
    
    def _create_complex_rotate(self):
        """Create 90-degree rotation transform for 3-channel data."""
        def transform(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # Random 90-degree rotations (1, 2, or 3 times)
            k = random.randint(1, 3)
            # np.rot90 rotates counter-clockwise by default, works on first 2 axes
            rotated_data = np.rot90(data, k=k, axes=(0, 1))
            rotated_labels = labels.copy()
            
            if len(rotated_labels) > 0:
                for _ in range(k):
                    # Rotate 90 degrees counter-clockwise to match np.rot90
                    # For counter-clockwise: new_x = old_y, new_y = 1 - old_x
                    temp_x = rotated_labels[:, 1].copy()
                    temp_y = rotated_labels[:, 2].copy()
                    temp_w = rotated_labels[:, 3].copy()
                    temp_h = rotated_labels[:, 4].copy()
                    
                    rotated_labels[:, 1] = temp_y        # new_x = old_y
                    rotated_labels[:, 2] = 1.0 - temp_x  # new_y = 1 - old_x
                    rotated_labels[:, 3] = temp_h        # Swap width and height
                    rotated_labels[:, 4] = temp_w
            
            return rotated_data, rotated_labels
        return transform
    
    def _create_complex_translate(self):
        """Create translation transform for 3-channel data."""
        def transform(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            h, w, c = data.shape
            
            # Random translation within 10% of image size
            max_shift_x = int(0.1 * w)
            max_shift_y = int(0.1 * h)
            shift_x = random.randint(-max_shift_x, max_shift_x)  # Fixed typo
            shift_y = random.randint(-max_shift_y, max_shift_y)
            
            # Translate the 3-channel data (shift each channel)
            translated_data = np.zeros_like(data)
            for ch in range(c):
                translated_data[:, :, ch] = ndimage_shift(data[:, :, ch], shift=(shift_y, shift_x), mode='constant', cval=0.0)
            
            # Update labels
            translated_labels = labels.copy()
            if len(translated_labels) > 0:
                # Convert normalized shift to label coordinates
                norm_shift_x = shift_x / w
                norm_shift_y = shift_y / h
                
                translated_labels[:, 1] += norm_shift_x
                translated_labels[:, 2] += norm_shift_y
                
                # Filter out boxes that are now outside the image
                valid_boxes = []
                for label in translated_labels:
                    cls, xc, yc, bw, bh = label
                    
                    # Calculate edges
                    x_min = xc - bw/2
                    y_min = yc - bh/2
                    x_max = xc + bw/2
                    y_max = yc + bh/2
                    
                    # Check visibility
                    if x_max > 0 and x_min < 1 and y_max > 0 and y_min < 1:
                        # Clip to image bounds
                        x_min_clipped = max(0, x_min)
                        y_min_clipped = max(0, y_min)
                        x_max_clipped = min(1, x_max)
                        y_max_clipped = min(1, y_max)
                        
                        # Calculate visibility
                        visible_w = x_max_clipped - x_min_clipped
                        visible_h = y_max_clipped - y_min_clipped
                        visibility = (visible_w * visible_h) / (bw * bh)
                        
                        if visibility >= self.min_visibility:
                            # Update to clipped coordinates
                            new_xc = (x_min_clipped + x_max_clipped) / 2
                            new_yc = (y_min_clipped + y_max_clipped) / 2
                            new_bw = x_max_clipped - x_min_clipped
                            new_bh = y_max_clipped - y_min_clipped
                            valid_boxes.append([cls, new_xc, new_yc, new_bw, new_bh])
                
                translated_labels = np.array(valid_boxes, dtype=np.float32) if valid_boxes else np.zeros((0, 5))
            
            return translated_data, translated_labels
        return transform
    
    # Phase shift not applicable for PNG data (phase already extracted)
    # def _create_phase_shift(self): - REMOVED
    
    def _create_amplitude_scale(self):
        """Create amplitude scaling transform for complex data."""
        def transform(data: np.ndarray) -> np.ndarray:
            # Scale amplitude by random factor
            scale_factor = np.random.uniform(0.7, 1.3)
            return data * scale_factor
        return transform
    
    def _create_amplitude_speckle(self):
        """Create speckle noise for amplitude data."""
        def transform(data: np.ndarray) -> np.ndarray:
            # Add multiplicative Gaussian noise to each channel
            noise_level = 0.05
            h, w, c = data.shape

            # Generate noise for each channel
            noise = np.random.normal(1.0, noise_level, (h, w, c))

            # Apply multiplicative noise
            noisy_data = data * noise

            # Clip to valid range [0, 1]
            return np.clip(noisy_data, 0.0, 1.0)
        return transform
    
    def _create_amplitude_gaussian_filter(self, sigma_range=(0.5, 2.0)):
        """
        Create Gaussian filter for amplitude data.
        Applies smoothing to each channel independently.

        Args:
            sigma_range: Tuple of (min_sigma, max_sigma) for random sigma selection
        """
        def transform(data: np.ndarray) -> np.ndarray:
            from scipy.ndimage import gaussian_filter

            # Random sigma for varying smoothing strength
            sigma = np.random.uniform(sigma_range[0], sigma_range[1])

            # Apply Gaussian filter to VH and VV magnitude channels only (channels 0 and 1)
            # Channel 0: VH magnitude (should be filtered)
            # Channel 1: VV magnitude (should be filtered)
            # Channel 2: PCF (should NOT be filtered)
            filtered_data = data.copy()
            filtered_data[:, :, 0] = gaussian_filter(data[:, :, 0], sigma=sigma)  # VH magnitude
            filtered_data[:, :, 1] = gaussian_filter(data[:, :, 1], sigma=sigma)  # VV magnitude
            # Channel 2 (PCF) remains unchanged
            return filtered_data
        return transform
    
    def _get_class_distribution(self, labels: np.ndarray) -> Dict[int, float]:
        """Get class distribution in image."""
        if len(labels) == 0:
            return {0: 0, 1: 0}
        
        classes, counts = np.unique(labels[:, 0].astype(int), return_counts=True)
        distribution = {int(c): int(count) for c, count in zip(classes, counts)}
        
        # Ensure both classes are represented
        for c in [0, 1]:
            if c not in distribution:
                distribution[c] = 0
                
        return distribution
    
    def _should_augment(self, labels: np.ndarray) -> Tuple[bool, int]:
        """
        Determine if image should be augmented and which class to prioritize.
        Returns: (should_augment, priority_class)
        """
        distribution = self._get_class_distribution(labels)
        
        # Prioritize minority class (is_fishing)
        if distribution[1] > 0:
            return True, 1
        elif distribution[0] > 0:
            # Still augment majority class but less frequently
            return random.random() < 0.5, 0
        else:
            return False, -1
    
    def augment_single(self,
                      slc_data: np.ndarray,
                      labels: np.ndarray,
                      force_augment: bool = False) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Augment a single PNG SAR image based on class distribution.

        Args:
            slc_data: PNG SAR data (H, W, 3) dtype=float32 with 3 channels (VH-amp, VV-amp, PCF)
            labels: YOLO format labels (N, 5) with [class, x_center, y_center, w, h]
            force_augment: Force augmentation regardless of class distribution

        Returns:
            List of (augmented_png_data, augmented_labels, augmentation_description)
        """
        results = []
        self.stats['processed'] += 1
        
        # Ensure correct data type and shape
        assert len(slc_data.shape) == 3 and slc_data.shape[2] == 3, f"Expected 3-channel data (H,W,3), got {slc_data.shape}"
        assert slc_data.dtype == np.float32, f"Expected float32 data, got {slc_data.dtype}"
        
        # Determine augmentation strategy
        if force_augment:
            should_augment = True
            priority_class = 1 if any(labels[:, 0] == 1) else 0
        else:
            should_augment, priority_class = self._should_augment(labels)
        
        if not should_augment:
            return [(slc_data, labels, "none")]
        
        self.stats['augmented'] += 1
        self.stats['by_class'][priority_class] += 1
        
        # Get probabilities for this class
        geo_probs = self.geometric_probs.get(priority_class, {})
        sar_probs = self.sar_probs.get(priority_class, {})
        
        # Quick Win #2: Batch random number generation
        # Pre-generate all random numbers needed for this augmentation
        n_geo_transforms = len(self.geometric_transforms)
        n_sar_transforms = len(self.sar_transforms)
        all_randoms = np.random.random(n_geo_transforms + n_sar_transforms)
        random_idx = 0
        
        # Apply augmentations
        applied_augmentations = []
        aug_data = slc_data.copy()
        aug_labels = labels.copy()
        
        # Apply geometric augmentations
        for aug_name, aug_transform in self.geometric_transforms.items():
            if all_randoms[random_idx] < geo_probs.get(aug_name, 0):
                aug_data, aug_labels = aug_transform(aug_data, aug_labels)
                applied_augmentations.append(aug_name)
                self._update_stats(aug_name)
            random_idx += 1
        
        # Apply SAR-specific augmentations (data only)
        for aug_name, aug_transform in self.sar_transforms.items():
            if all_randoms[random_idx] < sar_probs.get(aug_name, 0):
                aug_data = aug_transform(aug_data)
                applied_augmentations.append(aug_name)
                self._update_stats(aug_name)
            random_idx += 1
        
        # Create description
        description = "_".join(applied_augmentations) if applied_augmentations else "none"
        
        results.append((aug_data, aug_labels, description))
        
        return results
    
    def create_mosaic(self,
                     images_labels: List[Tuple[np.ndarray, np.ndarray]],
                     output_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create mosaic augmentation from 4 PNG SAR images.

        Args:
            images_labels: List of 4 (png_data, labels) tuples
            output_size: Output size (H, W), if None uses first image size

        Returns:
            (mosaic_png_data, mosaic_labels)
        """
        assert len(images_labels) == 4, "Mosaic requires exactly 4 images"
        
        if output_size is None:
            output_size = images_labels[0][0].shape[:2]  # Take only H, W from (H, W, C)

        h, w = output_size
        mosaic_data = np.zeros((h, w, 3), dtype=np.float32)
        mosaic_labels = []
        
        # Random center point
        cx = random.randint(w//4, 3*w//4)
        cy = random.randint(h//4, 3*h//4)
        
        # Place each image in a quadrant
        positions = [
            (0, 0, cx, cy),           # Top-left
            (cx, 0, w, cy),           # Top-right
            (0, cy, cx, h),           # Bottom-left
            (cx, cy, w, h)            # Bottom-right
        ]

        for i, ((png_data, labels), (x1, y1, x2, y2)) in enumerate(zip(images_labels, positions)):
            # Resize 3-channel data to fit quadrant
            quad_h, quad_w = y2 - y1, x2 - x1

            # Resize 3-channel data directly
            resized = cv2.resize(png_data, (quad_w, quad_h))

            mosaic_data[y1:y2, x1:x2] = resized
            
            # Adjust labels - work in normalized coordinates
            if len(labels) > 0:
                for label in labels:
                    cls, xc, yc, bw, bh = label
                    
                    # Map normalized coordinates to quadrant position in mosaic
                    new_xc = xc * (x2 - x1) / w + x1 / w
                    new_yc = yc * (y2 - y1) / h + y1 / h
                    
                    # Scale dimensions by quadrant size relative to full mosaic
                    new_bw = bw * (x2 - x1) / w
                    new_bh = bh * (y2 - y1) / h
                    
                    # Clip to quadrant boundaries
                    x_min = max(x1/w, new_xc - new_bw/2)
                    y_min = max(y1/h, new_yc - new_bh/2)
                    x_max = min(x2/w, new_xc + new_bw/2)
                    y_max = min(y2/h, new_yc + new_bh/2)
                    
                    # Recalculate center and dimensions after clipping
                    if x_max > x_min and y_max > y_min:
                        new_xc = (x_min + x_max) / 2
                        new_yc = (y_min + y_max) / 2
                        new_bw = x_max - x_min
                        new_bh = y_max - y_min
                        
                        # Validate final box size
                        if new_bw >= 0.001 and new_bh >= 0.001:
                            mosaic_labels.append([cls, new_xc, new_yc, new_bw, new_bh])
        
        mosaic_labels = np.array(mosaic_labels, dtype=np.float32) if mosaic_labels else np.zeros((0, 5))
        self._update_stats('mosaic')
        
        return mosaic_data, mosaic_labels
    
    def _load_single_file(self, data_path: Path, labels_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single PNG SAR data file and its labels."""
        # Load PNG SAR data (3-channel: VH-amp, VV-amp, PCF)
        png_data = cv2.imread(str(data_path), cv2.IMREAD_UNCHANGED)
        if png_data is None:
            raise ValueError(f"Could not load PNG file: {data_path}")

        # Convert BGR to RGB to match expected channel ordering
        # Expected: [0]=VH-amp, [1]=VV-amp, [2]=PCF
        if len(png_data.shape) == 3 and png_data.shape[2] == 3:
            png_data = cv2.cvtColor(png_data, cv2.COLOR_BGR2RGB)

        # Convert to float32 and normalize from [0, 255] to [0, 1]
        png_data = png_data.astype(np.float32) / 255.0

        # Ensure 3-channel format
        if len(png_data.shape) == 2:
            # Convert grayscale to 3-channel by replicating
            png_data = np.stack([png_data, png_data, png_data], axis=2)
        elif png_data.shape[2] == 4:
            # Remove alpha channel if present
            png_data = png_data[:, :, :3]

        assert len(png_data.shape) == 3 and png_data.shape[2] == 3, f"Expected 3-channel data, got shape {png_data.shape}"
        
        # Load labels with faster parsing
        label_path = labels_dir / f"{data_path.stem}.txt"
        
        if label_path.exists() and label_path.stat().st_size > 0:
            try:
                # Quick Win #1: Replace np.loadtxt with faster parsing
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    labels = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            labels.append([float(x) for x in parts])
                    
                    if labels:
                        labels = np.array(labels, dtype=np.float32)
                        # Validate label format
                        assert labels.shape[1] == 5, f"Labels should have 5 columns, got {labels.shape[1]}"
                        assert np.all(labels[:, 0] >= 0), "Class indices should be non-negative"
                        assert np.all((labels[:, 1:] >= 0) & (labels[:, 1:] <= 1)), "Coordinates should be in [0,1]"
                    else:
                        labels = np.zeros((0, 5), dtype=np.float32)
                else:
                    labels = np.zeros((0, 5), dtype=np.float32)
            except Exception as e:
                print(f"Error loading labels for {data_path.stem}: {e}")
                labels = np.zeros((0, 5), dtype=np.float32)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)
        
        return png_data, labels
    
    def _scan_labels_only(self, input_dir: Path, labels_dir: Path, file_extension: str) -> Dict[str, List[Path]]:
        """Scan only label files to categorize data files by class content."""
        data_files = sorted(Path(input_dir).glob(f'*{file_extension}'))
        file_categories = {'all': [], 'class_0': [], 'class_1': [], 'minority': [], 'majority': [], 'no_labels': []}
        
        print("Scanning labels for file categorization...")
        for data_path in tqdm(data_files, desc="Scanning"):
            label_path = labels_dir / f"{data_path.stem}.txt"
            
            if label_path.exists() and label_path.stat().st_size > 0:
                try:
                    # Quick Win #1: Use faster parsing for label scanning too
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    if lines:
                        labels = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                labels.append([float(x) for x in parts])
                        
                        if labels:
                            labels = np.array(labels, dtype=np.float32)
                            classes = set(labels[:, 0].astype(int))
                            if 1 in classes:
                                file_categories['class_1'].append(data_path)
                                file_categories['minority'].append(data_path)
                            if 0 in classes:
                                file_categories['class_0'].append(data_path)
                                file_categories['majority'].append(data_path)
                        else:
                            file_categories['no_labels'].append(data_path)
                    else:
                        file_categories['no_labels'].append(data_path)
                except:
                    file_categories['no_labels'].append(data_path)
            else:
                file_categories['no_labels'].append(data_path)
            
            file_categories['all'].append(data_path)
        
        return file_categories
    
    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         labels_dir: str,
                         output_labels_dir: str,
                         augmentations_per_image: Dict[int, int] = None,
                         file_extension: str = '.png',
                         enable_mosaic: bool = True):
        """
        Process entire directory with optimized loading strategy.
        
        Args:
            input_dir: Directory containing input PNG SAR data files
            output_dir: Directory to save augmented PNG SAR data
            labels_dir: Directory containing YOLO format labels
            output_labels_dir: Directory to save augmented labels
            augmentations_per_image: Number of augmentations per class
            file_extension: Image file extension (.png for processed SAR data)
            enable_mosaic: Whether to create mosaic augmentations
        """
        if augmentations_per_image is None:
            augmentations_per_image = {0: 1, 1: 3}
        
        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_labels_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert paths to Path objects
        input_dir = Path(input_dir)
        labels_dir = Path(labels_dir)
        
        if enable_mosaic:
            # Use existing implementation with all files loaded
            self._process_directory_with_mosaic(
                input_dir, output_dir, labels_dir, output_labels_dir,
                augmentations_per_image, file_extension
            )
        else:
            # Use lazy loading - process one file at a time
            self._process_directory_lazy(
                input_dir, output_dir, labels_dir, output_labels_dir,
                augmentations_per_image, file_extension
            )
    
    def _process_directory_lazy(self, input_dir: Path, output_dir: str, labels_dir: Path,
                               output_labels_dir: str, augmentations_per_image: Dict[int, int],
                               file_extension: str):
        """Process files one at a time without loading all into memory."""
        data_files = sorted(input_dir.glob(f'*{file_extension}'))
        
        # Initialize statistics
        augmentation_count = 0
        empty_label_count = 0
        class_distribution = {
            'original': {'class_0_only': 0, 'class_1_only': 0, 'both_classes': 0, 'no_labels': 0},
            'augmented': {'class_0_only': 0, 'class_1_only': 0, 'both_classes': 0, 'no_labels': 0}
        }
        
        print(f"Processing {len(data_files)} PNG SAR data files (lazy loading mode)...")
        
        for data_path in tqdm(data_files, desc="Processing"):
            # Load single file
            png_data, labels = self._load_single_file(data_path, labels_dir)

            # Always save original
            base_stem = data_path.stem.replace('_proc', '')  # Remove _proc if present
            self._save_data_label(png_data, labels, output_dir, output_labels_dir,
                                f"{base_stem}_original_proc", file_extension)
            
            # Track original class distribution
            self._update_class_distribution(labels, class_distribution['original'])
            
            # Determine number of augmentations based on class content
            class_dist = self._get_class_distribution(labels)
            if class_dist[1] > 0:  # Has minority class
                n_augs = augmentations_per_image.get(1, 3)
            else:
                n_augs = augmentations_per_image.get(0, 1)
            
            # Generate augmentations
            for i in range(n_augs):
                aug_results = self.augment_single(png_data, labels, force_augment=True)
                
                for j, (aug_data, aug_labels, desc) in enumerate(aug_results):
                    if desc == "none":
                        continue
                    
                    base_stem = data_path.stem.replace('_proc', '')  # Remove _proc if present
                    filename = f"{base_stem}_aug{i}_{desc}_proc"
                    self._save_data_label(aug_data, aug_labels, output_dir,
                                        output_labels_dir, filename, file_extension)
                    augmentation_count += 1
                    
                    if len(aug_labels) == 0:
                        empty_label_count += 1
                    self._update_class_distribution(aug_labels, class_distribution['augmented'])
            
            # Clear from memory
            del png_data
        
        # Print summary
        self._print_summary(len(data_files), augmentation_count, empty_label_count, class_distribution)
    
    def _process_directory_with_mosaic(self, input_dir: Path, output_dir: str, labels_dir: Path,
                                      output_labels_dir: str, augmentations_per_image: Dict[int, int],
                                      file_extension: str):
        """Process directory with mosaic support - loads all files for mosaic creation."""
        # First, scan labels to categorize files
        file_categories = self._scan_labels_only(input_dir, labels_dir, file_extension)
        
        print(f"\nFile categorization:")
        print(f"Files with class 0 (is_vessel): {len(file_categories['class_0'])}")
        print(f"Files with class 1 (is_fishing): {len(file_categories['class_1'])}")
        print(f"Files with no labels: {len(file_categories['no_labels'])}")
        
        # Initialize statistics
        augmentation_count = 0
        empty_label_count = 0
        class_distribution = {
            'original': {'class_0_only': 0, 'class_1_only': 0, 'both_classes': 0, 'no_labels': 0},
            'augmented': {'class_0_only': 0, 'class_1_only': 0, 'both_classes': 0, 'no_labels': 0}
        }
        
        # Process each file individually first
        print(f"\nProcessing augmentations...")
        for data_path in tqdm(file_categories['all'], desc="Augmenting"):
            # Load single file
            png_data, labels = self._load_single_file(data_path, labels_dir)
            
            # Always save original
            base_stem = data_path.stem.replace('_proc', '')  # Remove _proc if present
            self._save_data_label(png_data, labels, output_dir, output_labels_dir,
                                f"{base_stem}_original_proc", file_extension)
            
            # Track original class distribution
            self._update_class_distribution(labels, class_distribution['original'])
            
            # Determine number of augmentations
            class_dist = self._get_class_distribution(labels)
            if class_dist[1] > 0:
                n_augs = augmentations_per_image.get(1, 3)
            else:
                n_augs = augmentations_per_image.get(0, 1)
            
            # Generate augmentations
            for i in range(n_augs):
                aug_results = self.augment_single(png_data, labels, force_augment=True)
                
                for j, (aug_data, aug_labels, desc) in enumerate(aug_results):
                    if desc == "none":
                        continue
                    
                    base_stem = data_path.stem.replace('_proc', '')  # Remove _proc if present
                    filename = f"{base_stem}_aug{i}_{desc}_proc"
                    self._save_data_label(aug_data, aug_labels, output_dir,
                                        output_labels_dir, filename, file_extension)
                    augmentation_count += 1
                    
                    if len(aug_labels) == 0:
                        empty_label_count += 1
                    self._update_class_distribution(aug_labels, class_distribution['augmented'])
            
            # Clear from memory after processing
            del png_data
        
        # Create mosaics efficiently
        print(f"\nCreating mosaic augmentations...")
        self._create_mosaics_efficient(file_categories, input_dir, labels_dir, output_dir,
                                      output_labels_dir, file_extension, class_distribution)
        
        # Print summary
        self._print_summary(len(file_categories['all']), augmentation_count, empty_label_count, class_distribution)
    
    def _create_mosaics_efficient(self, file_categories: Dict[str, List[Path]], input_dir: Path,
                                 labels_dir: Path, output_dir: str, output_labels_dir: str,
                                 file_extension: str, class_distribution: Dict):
        """Create mosaics by loading only required files."""
        minority_prob = self.mosaic_prob.get(1, 0.3)
        majority_prob = self.mosaic_prob.get(0, 0.1)
        
        # Calculate number of mosaics to create
        n_mosaics_minority = max(
            1 if minority_prob > 0 and len(file_categories['minority']) >= 2 else 0,
            int(len(file_categories['minority']) * minority_prob)
        )
        n_mosaics_majority = max(
            1 if majority_prob > 0 and len(file_categories['all']) >= 4 else 0,
            int(len(file_categories['majority']) * majority_prob)
        )
        
        # Cap the number of mosaics
        n_mosaics_minority = min(n_mosaics_minority, 50)
        n_mosaics_majority = min(n_mosaics_majority, 20)
        
        print(f"Creating {n_mosaics_minority} mosaics for minority class")
        print(f"Creating {n_mosaics_majority} mosaics for majority class")
        
        mosaic_count = 0
        
        # Create mosaics for minority class
        for i in range(n_mosaics_minority):
            if len(file_categories['minority']) >= 2:
                # Select files for mosaic
                selected_paths = []
                
                # Get 2 minority class files
                minority_sample = random.sample(file_categories['minority'], 
                                              min(2, len(file_categories['minority'])))
                selected_paths.extend(minority_sample)
                
                # Fill remaining with any files
                remaining_files = [f for f in file_categories['all'] if f not in selected_paths]
                if len(remaining_files) >= 2:
                    selected_paths.extend(random.sample(remaining_files, 2))
                else:
                    continue
                
                # Load only these 4 files
                mosaic_data = []
                for path in selected_paths:
                    png_data, labels = self._load_single_file(path, labels_dir)
                    mosaic_data.append((png_data, labels))
                
                # Create and save mosaic
                mosaic_png, mosaic_labels = self.create_mosaic(mosaic_data)
                filename = f"mosaic_minority_{mosaic_count}_proc"
                self._save_data_label(mosaic_png, mosaic_labels, output_dir,
                                    output_labels_dir, filename, file_extension)
                
                # Track mosaic class distribution
                self._update_class_distribution(mosaic_labels, class_distribution['augmented'])
                mosaic_count += 1
                
                # Clear memory
                del mosaic_data
        
        # Create mosaics for majority class
        for i in range(n_mosaics_majority):
            if len(file_categories['all']) >= 4:
                # Random selection of 4 files
                selected_paths = random.sample(file_categories['all'], 4)
                
                # Load only these 4 files
                mosaic_data = []
                for path in selected_paths:
                    png_data, labels = self._load_single_file(path, labels_dir)
                    mosaic_data.append((png_data, labels))
                
                # Create and save mosaic
                mosaic_png, mosaic_labels = self.create_mosaic(mosaic_data)
                filename = f"mosaic_majority_{mosaic_count}_proc"
                self._save_data_label(mosaic_png, mosaic_labels, output_dir,
                                    output_labels_dir, filename, file_extension)
                
                # Track mosaic class distribution
                self._update_class_distribution(mosaic_labels, class_distribution['augmented'])
                mosaic_count += 1
                
                # Clear memory
                del mosaic_data
        
        print(f"Created {mosaic_count} total mosaics")
    
    def _print_summary(self, n_original: int, augmentation_count: int, empty_label_count: int,
                      class_distribution: Dict):
        """Print processing summary."""
        print(f"\nAugmentation Summary:")
        print(f"Original PNG SAR data files: {n_original}")
        print(f"Augmented PNG SAR data files created: {augmentation_count}")
        print(f"Augmented files with empty labels: {empty_label_count}")
        print(f"Total PNG SAR data files after augmentation: {n_original + augmentation_count}")
        
        # Print class distribution
        print(f"\nClass Distribution Analysis:")
        print(f"Original data:")
        print(f"  Class 0 (is_vessel) only: {class_distribution['original']['class_0_only']}")
        print(f"  Class 1 (is_fishing) only: {class_distribution['original']['class_1_only']}")
        print(f"  Both classes: {class_distribution['original']['both_classes']}")
        print(f"  No labels: {class_distribution['original']['no_labels']}")
        
        print(f"\nAugmented data:")
        print(f"  Class 0 (is_vessel) only: {class_distribution['augmented']['class_0_only']}")
        print(f"  Class 1 (is_fishing) only: {class_distribution['augmented']['class_1_only']}")
        print(f"  Both classes: {class_distribution['augmented']['both_classes']}")
        print(f"  No labels: {class_distribution['augmented']['no_labels']}")
        
        # Calculate total class occurrences
        orig_class_0 = class_distribution['original']['class_0_only'] + class_distribution['original']['both_classes']
        orig_class_1 = class_distribution['original']['class_1_only'] + class_distribution['original']['both_classes']
        aug_class_0 = class_distribution['augmented']['class_0_only'] + class_distribution['augmented']['both_classes']
        aug_class_1 = class_distribution['augmented']['class_1_only'] + class_distribution['augmented']['both_classes']
        
        total_class_0 = orig_class_0 + aug_class_0
        total_class_1 = orig_class_1 + aug_class_1
        
        print(f"\nTotal class occurrences:")
        print(f"Original - Class 0: {orig_class_0}, Class 1: {orig_class_1} (Ratio {orig_class_0/max(orig_class_1, 1):.2f}:1)")
        print(f"After augmentation - Class 0: {total_class_0}, Class 1: {total_class_1} (Ratio {total_class_0/max(total_class_1, 1):.2f}:1)")
    
    def _save_data_label(self, png_data, labels, data_dir, label_dir, filename, extension):
        """Save PNG SAR data and corresponding labels."""
        # Save PNG data - convert back to uint8 [0, 255] range
        if extension == '.png':
            # Clip to [0, 1] range and convert to uint8
            png_data_uint8 = np.clip(png_data * 255.0, 0, 255).astype(np.uint8)

            # Convert RGB back to BGR for OpenCV saving
            # Our data is in RGB format (VH, VV, PCF) but cv2.imwrite expects BGR
            if len(png_data_uint8.shape) == 3 and png_data_uint8.shape[2] == 3:
                png_data_bgr = cv2.cvtColor(png_data_uint8, cv2.COLOR_RGB2BGR)
            else:
                png_data_bgr = png_data_uint8

            cv2.imwrite(str(Path(data_dir) / f"{filename}.png"), png_data_bgr)
        elif extension == '.npy':
            # Save as float32 numpy array
            np.save(Path(data_dir) / f"{filename}.npy", png_data)
        else:
            # Alternative: save as compressed npz
            np.savez_compressed(Path(data_dir) / f"{filename}.npz", data=png_data)
        
        # Save labels
        label_path = Path(label_dir) / f"{filename}.txt"
        
        if len(labels) > 0:
            np.savetxt(label_path, labels, fmt='%d %.6f %.6f %.6f %.6f')
        else:
            label_path.touch()  # Create empty file
    
    def _update_stats(self, aug_type: str):
        """Update augmentation statistics."""
        if aug_type not in self.stats['augmentation_counts']:
            self.stats['augmentation_counts'][aug_type] = 0
        self.stats['augmentation_counts'][aug_type] += 1
    
    def get_stats(self) -> Dict:
        """Get augmentation statistics."""
        return self.stats
    
    def _update_class_distribution(self, labels: np.ndarray, dist_dict: Dict):
        """Update class distribution statistics."""
        if len(labels) == 0:
            dist_dict['no_labels'] += 1
        else:
            classes = set(labels[:, 0].astype(int))
            if classes == {0}:
                dist_dict['class_0_only'] += 1
            elif classes == {1}:
                dist_dict['class_1_only'] += 1
            elif classes == {0, 1}:
                dist_dict['both_classes'] += 1
    
    def save_config(self, filepath: str):
        """Save augmentation configuration."""
        config = {
            'geometric_probs': self.geometric_probs,
            'sar_probs': self.sar_probs,
            'mosaic_prob': self.mosaic_prob,
            'min_visibility': self.min_visibility
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_config(cls, filepath: str):
        """Load augmentation configuration."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls(**config)

