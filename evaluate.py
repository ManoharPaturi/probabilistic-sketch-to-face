"""Evaluate a checkpoint on the held-out test split (L1, SSIM, PSNR [+ uncertainty]).

python evaluate.py --model bayesian --weights weights/bayesian_sketch_to_face_final.pth --data-root ...
"""
import argparse, json, os
import torch
from torch.utils.data import DataLoader
from src.dataset import SketchFaceDataset
from src.losses import reconstruction_loss, psnr

from src.utils import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["deterministic", "bayesian"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="data/splits/test.csv")
    ap.add_argument("--samples", type=int, default=8)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.model, a.weights, device)
    dl = DataLoader(SketchFaceDataset(a.split, a.data_root), batch_size=16, num_workers=2)
    tot, unc, n = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "psnr": 0.0}, 0.0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            if a.model == "bayesian":
                pred, _, ys = model.predict(x, a.samples)
                unc += ys.std(0).mean().item() * x.size(0)
            else:
                pred = model(x)
            rec, l1, s = reconstruction_loss(pred, y)
            for k, v in zip(tot, (rec, l1, s, psnr(pred, y))):
                tot[k] += v.item() * x.size(0)
            n += x.size(0)
    res = {k: v / n for k, v in tot.items()}
    if a.model == "bayesian":
        res["mean_uncertainty"] = unc / n
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
