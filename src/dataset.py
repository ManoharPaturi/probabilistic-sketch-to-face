"""Paired sketch/face dataset driven by the split CSVs in data/splits/."""
import os
import pandas as pd
from torch.utils.data import Dataset
from .preprocess import load_sketch, load_face


class SketchFaceDataset(Dataset):
    """Returns (sketch 1x224x224 in [0, 1], face 3x224x224 in [-1, 1]).

    The CSVs store absolute Kaggle paths. Pass `root` to remap them to a local
    copy of the dataset: everything up to and including `utkface_aligned_cropped/`
    is replaced by `root`.
    """

    def __init__(self, csv_path, root=None, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.root = root
        self.size = image_size

    def _path(self, p):
        if self.root is None:
            return p
        key = "utkface_aligned_cropped/"
        return os.path.join(self.root, p.split(key, 1)[1]) if key in p else p

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        return load_sketch(self._path(row.sketch_path), self.size), load_face(self._path(row.face_path), self.size)
