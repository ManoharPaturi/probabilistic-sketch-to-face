"""Pre-processing identical to the original Kaggle notebook (OpenCV, INTER_AREA).

sketch: grayscale, 224x224, [0, 255] -> [0, 1]     shape [1, H, W]
face:   RGB,       224x224, [0, 255] -> [-1, 1]    shape [3, H, W]
The model is sensitive to the resize filter: use INTER_AREA as in training.
"""
import cv2
import numpy as np
import torch


def load_sketch(path, size=224):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read sketch: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img.astype(np.float32) / 255.0)[None]


def load_face(path, size=224):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read face: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(np.transpose(img, (2, 0, 1)))


def to_image(t):
    """[-1, 1] tensor [3, H, W] -> uint8 HxWx3 for display."""
    return ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
