# sar_dataset.py - custom Dataset class
import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from letterbox_resize import letterbox_resize  # custom function

class SARPreprocessedDataset(Dataset):
    def __init__(self, image_dir, label_dir, imgsz=640):
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir) if fname.endswith('.npy')
        ])
        self.label_paths = sorted([
            os.path.join(label_dir, fname.replace('_proc.npy', '.txt'))
            for fname in os.listdir(image_dir) if fname.endswith('.npy')
        ])
        self.imgsz = imgsz

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        im = np.load(self.image_paths[idx])  # (3, H, W)
        im = np.transpose(im, (1, 2, 0))      # → (H, W, 3)

        # Load labels
        label_path = self.label_paths[idx]
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            labels = np.loadtxt(label_path, ndmin=2, dtype=np.float32)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        # Un-normalize labels BEFORE re-sizing
        h0, w0 = im.shape[:2]
        labels[:, 1] *= w0
        labels[:, 2] *= h0
        labels[:, 3] *= w0
        labels[:, 4] *= h0
        
        # Apply letterbox resize
        im, labels = letterbox_resize(im, labels, new_shape=self.imgsz)

        # Convert back to torch format
        im = torch.from_numpy(im).permute(2, 0, 1).float().contiguous()
        labels = torch.from_numpy(labels).float()

        return im, labels
