#!/usr/bin/env python3

"""
visualise_crops.py

Author: Peter Millitz
Enhanced: 2025-08-17

Visualises SAR image crops stored as .npy files (complex64, 2D) with optional YOLO-format bounding box overlays.

Amplitude clipping applied by default to remove extreme outliers. Default clipping uses
1% and 99% percentiles of the training dataset (1.00 and 71.51).

Options:
  -h, --help: show this help message and exit
  -i, --images IMAGES: directory containing .npy image files (complex64, 2D)
  -l, --labels LABELS: directory containing .txt label files (YOLO format)
  -m, --mode MODE: initial display mode (magnitude, log_magnitude)
  -n, --samples SAMPLES: number of randomly sampled images to view (default = 50)
  -s, --single IMAGE: view single image by filename
  --class-names: custom class names (default: is_vessel, is_fishing)
  --amp-clip-params AMP_MIN AMP_MAX: custom amplitude clipping (default: 1.00 71.51)

Navigation:
 →  : move to next crop
 ←  : move to previous crop
 ↑  : next display mode
 ↓  : previous display mode
 Esc: exit the viewer
"""

import numpy as np
import matplotlib
import random
from pathlib import Path
import argparse

# Try to use interactive backend
try:
    matplotlib.use('Qt5Agg')
except:
    try:
        matplotlib.use('TkAgg')
    except:
        print("Warning: No interactive backend available. GUI may not work properly.")

matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# Configuration constants
DEFAULT_EPSILON = 1e-10
DEFAULT_WINDOW_POSITION = "+400+200"
DEFAULT_CLASS_NAMES = ["is_vessel", "is_fishing"]
# Default amplitude clipping using 1% and 99% percentiles of training set
DEFAULT_AMP_MIN = 1.00  # 1% percentile
DEFAULT_AMP_MAX = 71.51  # 99% percentile

def validate_amp_clip_params(params, verbose=False):
    """
    Validate amplitude clipping parameters.
    
    Parameters:
    -----------
    params : list of float
        [amp_min, amp_max]
    
    Returns:
    --------
    bool
        True if parameters are valid
    """
    if len(params) != 2:
        print(f"Error: amp-clip-params requires exactly 2 values, got {len(params)}")
        return False
    
    amp_min, amp_max = params
    
    # Validate amplitude parameters (unscaled, must be non-negative)
    if amp_min < 0 or amp_max <= 0:
        print(f"Error: amplitude parameters must be non-negative with amp_max > 0, got [{amp_min:.6f}, {amp_max:.6f}]")
        return False
    
    # Handle amp_min = 0.0 case (convert to small positive value to avoid log(0))
    if amp_min == 0.0:
        amp_min = 1e-10  # Use a very small positive value
        params[0] = amp_min  # Update the original list
        if verbose:
            print(f"Note: amp_min of 0.0 converted to {amp_min:.2e} to avoid log(0) issues")
    
    if amp_min >= amp_max:
        print(f"Error: amp_min ({amp_min:.6f}) must be less than amp_max ({amp_max:.6f})")
        return False
    
    if verbose:
        print("Amplitude clipping parameters validated successfully:")
        print(f"  Amplitude (unscaled): [{amp_min:.6f}, {amp_max:.6f}]")
    
    return True

def load_sar_image(image_path):
    """
    Load SAR image data from .npy file.
    
    Args:
        image_path: Path to .npy image file
        
    Returns:
        numpy.ndarray: Complex64 SAR data with shape (H, W)
    """
    image_path = Path(image_path)
    
    if image_path.suffix.lower() != '.npy':
        raise ValueError(f"Only .npy files are supported, got {image_path.suffix}")
    
    data = np.load(image_path)
    if data.dtype != np.complex64 or len(data.shape) != 2:
        raise ValueError(f"Expected complex64 2D array, got {data.dtype} with shape {data.shape}")
    
    return data

class DisplayModeProcessor:
    """Handles different display mode processing for SAR data."""
    
    @staticmethod
    def normalize_image(image, fixed_range):
        """
        Normalize image for display using fixed range.
        
        Args:
            image: Input image array
            fixed_range: Tuple (min_val, max_val) for fixed normalization
        """
        min_val, max_val = fixed_range
        # Clip to fixed range and normalize
        clipped = np.clip(image, min_val, max_val)
        if max_val > min_val:
            return (clipped - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(clipped)
    
    @staticmethod
    def process_mode(complex_data, display_mode, amp_clip_params=None):
        """
        Process complex SAR data according to display mode.
        Always applies amplitude clipping using either provided parameters or default percentiles.
        
        Args:
            complex_data: Input complex SAR data
            display_mode: 'magnitude' or 'log_magnitude'
            amp_clip_params: Optional tuple (amp_min, amp_max) for amplitude clipping
        
        Returns:
            tuple: (processed_image, colormap, mode_title)
        """
        magnitude = np.abs(complex_data)
        
        # Use provided parameters or defaults
        if amp_clip_params is not None:
            amp_min, amp_max = amp_clip_params
            is_default_clipping = False
        else:
            amp_min, amp_max = DEFAULT_AMP_MIN, DEFAULT_AMP_MAX
            is_default_clipping = True
        
        if display_mode == 'magnitude':
            # Always clip amplitude
            clipped_magnitude = np.clip(magnitude, amp_min, amp_max)
            fixed_range = (amp_min, amp_max)
            
            if is_default_clipping:
                title = f"Magnitude (default clipped [{amp_min:.2f}, {amp_max:.2f}])"
            else:
                title = f"Magnitude (clipped [{amp_min:.2f}, {amp_max:.2f}])"
            
            return DisplayModeProcessor.normalize_image(clipped_magnitude, fixed_range), 'gray', title
        
        elif display_mode == 'log_magnitude':
            # Always clip amplitude in linear space before dB conversion
            clipped_magnitude = np.clip(magnitude, amp_min, amp_max)
            # Convert to dB
            image = 20 * np.log10(clipped_magnitude + DEFAULT_EPSILON)
            # Fixed dB range for normalization
            amp_db_min = 20 * np.log10(amp_min + DEFAULT_EPSILON)
            amp_db_max = 20 * np.log10(amp_max + DEFAULT_EPSILON)
            fixed_range = (amp_db_min, amp_db_max)
            
            if is_default_clipping:
                title = f"Log Magnitude (default clipped [{amp_min:.2f}, {amp_max:.2f}] → [{amp_db_min:.1f}, {amp_db_max:.1f}] dB)"
            else:
                title = f"Log Magnitude (clipped [{amp_min:.2f}, {amp_max:.2f}] → [{amp_db_min:.1f}, {amp_db_max:.1f}] dB)"
            
            return DisplayModeProcessor.normalize_image(image, fixed_range), 'gray', title
        
        else:
            raise ValueError(f"Unknown display_mode: {display_mode}")

class LabelHandler:
    """Handles YOLO label file loading and bounding box drawing."""
    
    def __init__(self, class_names=None):
        self.class_names = class_names or DEFAULT_CLASS_NAMES
    
    def find_label_file(self, image_path, label_dir):
        """
        Find corresponding label file for .npy image.
        
        Naming pattern: filename.npy → filename.txt
        """
        if label_dir is None:
            return None
        
        image_path = Path(image_path)
        label_dir = Path(label_dir)
        
        # For .npy files: label should have same stem + .txt
        label_name = f"{image_path.stem}.txt"
        label_path = label_dir / label_name
        return label_path if label_path.exists() else None
    
    def load_bounding_boxes(self, label_path, image_shape):
        """Load bounding boxes from YOLO format file."""
        if not label_path or not label_path.exists():
            return []
        
        boxes = []
        try:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 4:
                        print(f"Warning: Invalid label format in {label_path.name}, line {line_num}")
                        continue
                    
                    try:
                        if len(parts) == 5:
                            class_id, xc, yc, w, h = map(float, parts)
                        else:
                            # Assume 4 values: xc, yc, w, h (class_id = 0)
                            xc, yc, w, h = map(float, parts)
                            class_id = 0
                        
                        boxes.append((int(class_id), xc, yc, w, h))
                    except ValueError as e:
                        print(f"Warning: Could not parse line {line_num} in {label_path.name}: {e}")
                        continue
        except IOError as e:
            print(f"Warning: Could not read label file {label_path}: {e}")
        
        return boxes
    
    def draw_bounding_boxes(self, ax, boxes, image_shape):
        """Draw YOLO bounding boxes on the axes."""
        if not boxes:
            return
        
        img_h, img_w = image_shape
        
        for class_id, xc, yc, w, h in boxes:
            # Convert from normalized YOLO format to pixel coordinates
            box_w = w * img_w
            box_h = h * img_h
            box_x = (xc * img_w) - box_w / 2
            box_y = (yc * img_h) - box_h / 2
            
            # Set colors based on class: red for is_vessel (0), lime for is_fishing (1)
            if class_id == 0:  # is_vessel
                edge_color = 'red'
                text_color = 'red'
            elif class_id == 1:  # is_fishing
                edge_color = 'lime'
                text_color = 'lime'
            else:  # unknown class
                edge_color = 'yellow'
                text_color = 'yellow'
            
            # Draw rectangle
            rect = plt.Rectangle((box_x, box_y), box_w, box_h,
                               edgecolor=edge_color, facecolor='none', linewidth=2)
            ax.add_patch(rect)
            
            # Add class label
            if 0 <= class_id < len(self.class_names):
                label_str = self.class_names[class_id]
            else:
                label_str = f"class_{class_id}"
            
            label_x = box_x - 2
            label_y = box_y
            ax.text(label_x, label_y, label_str,
                   color=text_color, fontsize=10,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1))

class SARViewer:
    """Main viewer class for SAR image visualization."""
    
    def __init__(self, class_names=None, amp_clip_params=None):
        self.modes = ['magnitude', 'log_magnitude']
        self.label_handler = LabelHandler(class_names)
        self.amp_clip_params = amp_clip_params
    
    
    def visualise_crop(self, crop_path, label_dir=None, title=None, display_mode='magnitude', fig=None, ax=None, image_count=None):
        """Display a SAR crop with optional YOLO bounding boxes."""
        try:
            # Load image data
            complex_data = load_sar_image(crop_path)
            shape = complex_data.shape
            
            # Process according to display mode
            image, cmap, mode_title = DisplayModeProcessor.process_mode(complex_data, display_mode, self.amp_clip_params)
            
            # Setup or reuse figure
            if fig is None or ax is None:
                fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
            else:
                fig.clear()
                ax = fig.add_subplot(111)
            
            # Display image
            im = ax.imshow(image, origin="upper", cmap=cmap)
            
            # Add colorbar with proper value labels
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Set colorbar labels based on display mode and clipping
            if display_mode == 'magnitude':
                if self.amp_clip_params is not None:
                    amp_min, amp_max = self.amp_clip_params
                else:
                    amp_min, amp_max = DEFAULT_AMP_MIN, DEFAULT_AMP_MAX
                
                cbar.set_label('Magnitude', rotation=270, labelpad=15)
                # Set ticks to show actual magnitude values
                cbar.set_ticks([0.0, 1.0])
                cbar.set_ticklabels([f'{amp_min:.2f}', f'{amp_max:.2f}'])
            
            elif display_mode == 'log_magnitude':
                if self.amp_clip_params is not None:
                    amp_min, amp_max = self.amp_clip_params
                else:
                    amp_min, amp_max = DEFAULT_AMP_MIN, DEFAULT_AMP_MAX
                
                # Calculate corresponding dB values
                db_min = 20 * np.log10(amp_min + DEFAULT_EPSILON)
                db_max = 20 * np.log10(amp_max + DEFAULT_EPSILON)
                cbar.set_label('Log Magnitude (dB)', rotation=270, labelpad=15)
                # Set ticks to show actual dB values
                cbar.set_ticks([0.0, 1.0])
                cbar.set_ticklabels([f'{db_min:.1f}', f'{db_max:.1f}'])
            
            # Add bounding boxes
            label_path = self.label_handler.find_label_file(crop_path, label_dir)
            boxes = self.label_handler.load_bounding_boxes(label_path, shape)
            self.label_handler.draw_bounding_boxes(ax, boxes, shape)
            
            # Set title - just filename with count
            image_name = Path(crop_path).stem
            if image_count:
                title_text = f"{image_name} {image_count}"
            else:
                title_text = image_name
            
            ax.set_title(title_text, fontsize=10, pad=10)
            
            # Add clipping info in top-right corner
            if self.amp_clip_params:
                amp_min, amp_max = self.amp_clip_params
                clip_text = f"Amplitude clipped: [{amp_min:.2f}, {amp_max:.2f}]"
            else:
                clip_text = f"Amplitude clipped: [{DEFAULT_AMP_MIN:.2f}, {DEFAULT_AMP_MAX:.2f}] (default)"
            
            # Position text in top-right corner with semi-transparent background
            ax.text(0.98, 0.98, clip_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            ax.axis("off")
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Error displaying {crop_path}: {e}")
            if ax is not None:
                ax.clear()
                ax.text(0.5, 0.5, f"Error loading image:\n{e}", 
                       ha='center', va='center', transform=ax.transAxes)
    
    def setup_figure(self, title="SAR Viewer"):
        """Create and setup figure window."""
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        fig.canvas.manager.set_window_title(title)
        
        try:
            mngr = fig.canvas.manager
            mngr.window.wm_geometry(DEFAULT_WINDOW_POSITION)
        except:
            pass
        
        return fig, ax
    
    def find_image_files(self, image_dir):
        """Find all .npy image files in directory."""
        image_dir = Path(image_dir)
        npy_files = list(image_dir.glob("*.npy"))
        return sorted(npy_files)
    
    def visualise_single(self, image_dir, image_name, label_dir=None, display_mode='magnitude'):
        """Visualise a single specific image with display mode switching."""
        if not Path(image_dir).exists():
            print(f"Error: Image directory does not exist: {image_dir}")
            return
        
        # Find the specific image file
        image_path = Path(image_dir) / image_name
        if not image_path.exists():
            # Try adding .npy extension if not present
            if not image_name.endswith('.npy'):
                test_path = Path(image_dir) / (image_name + '.npy')
                if test_path.exists():
                    image_path = test_path
        
        if not image_path.exists():
            print(f"Image not found: {image_name}")
            print(f"Searched in: {image_dir}")
            return
        
        print(f"Displaying: {image_path.name}")
        
        # Setup figure and interaction
        fig, ax = self.setup_figure(f'SAR Viewer - {image_path.name}')
        mode_index = [self.modes.index(display_mode) if display_mode in self.modes else 0]
        done = [False]
        
        def on_key(event):
            if event.key == 'up':
                mode_index[0] = (mode_index[0] + 1) % len(self.modes)
                update()
            elif event.key == 'down':
                mode_index[0] = (mode_index[0] - 1) % len(self.modes)
                update()
            elif event.key == 'escape':
                done[0] = True
                plt.close(fig)
        
        def update():
            current_mode = self.modes[mode_index[0]]
            self.visualise_crop(image_path, label_dir, None, current_mode, fig, ax)
        
        cid = fig.canvas.mpl_connect('key_press_event', on_key)
        update()
        
        print("Single Image Mode:")
        print("  ↑ : next display mode")
        print("  ↓ : previous display mode")
        print("  Esc : quit")
        print(f"\nDisplay modes: {', '.join(self.modes)}")
        print(f"Current mode: {self.modes[mode_index[0]]}")
        
        plt.show(block=False)
        
        try:
            while not done[0] and plt.fignum_exists(fig.number):
                plt.pause(0.1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            try:
                fig.canvas.mpl_disconnect(cid)
                plt.close(fig)
            except:
                pass
    
    def visualise_all(self, image_dir, label_dir=None, display_mode='magnitude', n_samples=50):
        """Visualise crop-label pairs with sampling and keyboard navigation."""
        if not Path(image_dir).exists():
            print(f"Error: Image directory does not exist: {image_dir}")
            return
        
        all_image_files = self.find_image_files(image_dir)
        if len(all_image_files) == 0:
            print("No .npy files found in:", image_dir)
            return
        
        # Apply sampling
        total_count = len(all_image_files)
        if n_samples >= total_count:
            sample_files = all_image_files
            print(f"Using all {total_count} images")
        else:
            sample_files = sorted(random.sample(all_image_files, n_samples))
            print(f"Randomly sampled {n_samples} images from {total_count} total")
        
        # Setup figure and interaction
        fig, ax = self.setup_figure('SAR Crop Viewer')
        index = [0]
        mode_index = [self.modes.index(display_mode) if display_mode in self.modes else 0]
        done = [False]
        
        def on_key(event):
            if event.key == 'right':
                if index[0] < len(sample_files) - 1:
                    index[0] += 1
                    update()
            elif event.key == 'left':
                if index[0] > 0:
                    index[0] -= 1
                    update()
            elif event.key == 'up':
                mode_index[0] = (mode_index[0] + 1) % len(self.modes)
                update()
            elif event.key == 'down':
                mode_index[0] = (mode_index[0] - 1) % len(self.modes)
                update()
            elif event.key == 'escape':
                done[0] = True
                plt.close(fig)
        
        def update():
            crop_path = sample_files[index[0]]
            current_mode = self.modes[mode_index[0]]
            image_count = f"({index[0]+1}/{len(sample_files)})"
            self.visualise_crop(crop_path, label_dir, None, current_mode, fig, ax, image_count)
        
        cid = fig.canvas.mpl_connect('key_press_event', on_key)
        update()
        
        print("Navigation:")
        print("  → : next crop")
        print("  ← : previous crop")
        print("  ↑ : next display mode")
        print("  ↓ : previous display mode")
        print("  Esc : quit")
        print(f"\nDisplay modes: {', '.join(self.modes)}")
        print(f"Sample size: {len(sample_files)} images")
        
        plt.show(block=False)
        
        try:
            while not done[0] and plt.fignum_exists(fig.number):
                plt.pause(0.1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            try:
                fig.canvas.mpl_disconnect(cid)
                plt.close(fig)
            except:
                pass

def main():
    parser = argparse.ArgumentParser(
        description='Visualize SAR crops (.npy complex64) with optional YOLO labels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View numpy complex64 files
  %(prog)s -i /path/to/npy_crops -l /path/to/labels
  
  # Single image
  %(prog)s -i /path/to/crops -s image_001 -l /path/to/labels
  
  # With custom amplitude clipping
  %(prog)s -i /path/to/npy_crops -l /path/to/labels --amp-clip-params 33.11 71.51
        """
    )
    
    parser.add_argument('--images', '-i', required=True,
                       help='Directory containing .npy image files (complex64, 2D)')
    parser.add_argument('--labels', '-l', default=None,
                       help='Directory containing .txt label files (YOLO format, optional)')
    parser.add_argument('--mode', '-m', default='magnitude',
                       choices=['magnitude', 'log_magnitude'],
                       help='Initial display mode (default: magnitude)')
    parser.add_argument('--samples', '-n', type=int, default=50,
                       help='Number of images to randomly sample for viewing (default: 50)')
    parser.add_argument('--single', '-s',
                       help='View single image by filename (e.g., "image_001" or "image_001.npy")')
    parser.add_argument('--class-names',
                       help='Custom class names separated by commas (default: is_vessel, is_fishing)')
    parser.add_argument('--amp-clip-params',
                       nargs=2,
                       type=float,
                       metavar=('AMP_MIN', 'AMP_MAX'),
                       help='Custom amplitude clipping parameters: amp_min amp_max. '
                            'Values are unscaled raw amplitude bounds. Default: 1.00 71.51')
    
    args = parser.parse_args()
    
    # Parse class names
    class_names = None
    if args.class_names:
        class_names = [name.strip() for name in args.class_names.split(',')]
    
    # Validate amplitude clipping parameters if provided
    amp_clip_params = None
    if args.amp_clip_params is not None:
        if validate_amp_clip_params(args.amp_clip_params, verbose=True):
            amp_clip_params = tuple(args.amp_clip_params)
        else:
            return 1  # Exit with error
    
    # Create viewer
    viewer = SARViewer(class_names, amp_clip_params)
    
    if args.single:
        viewer.visualise_single(
            image_dir=args.images,
            image_name=args.single,
            label_dir=args.labels,
            display_mode=args.mode
        )
    else:
        viewer.visualise_all(
            image_dir=args.images,
            label_dir=args.labels,
            display_mode=args.mode,
            n_samples=args.samples
        )

if __name__ == "__main__":
    main()
