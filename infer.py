"""Generate a face (and, for the Bayesian model, an uncertainty map) from one sketch.

python infer.py --sketch my_sketch.png --weights weights/bayesian_sketch_to_face_final.pth --out result.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from src.preprocess import load_sketch, to_image
from src.utils import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", choices=["deterministic", "bayesian"], default="bayesian")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", default="result.png")
    a = ap.parse_args()

    x = load_sketch(a.sketch).unsqueeze(0)
    model = load_model(a.model, a.weights)
    with torch.no_grad():
        if a.model == "bayesian":
            mean, _, ys = model.predict(x, a.samples)
            unc = ys.std(0).mean(1)[0]  # per-pixel std across samples, averaged over RGB
        else:
            mean, unc = model(x), None

    cols = 3 if unc is not None else 2
    fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4))
    ax[0].imshow(x[0, 0], cmap="gray"); ax[0].set_title("Input sketch")
    ax[1].imshow(to_image(mean[0])); ax[1].set_title("Generated face (mean)")
    if unc is not None:
        im = ax[2].imshow(unc, cmap="viridis"); ax[2].set_title(f"Uncertainty (S={a.samples})")
        fig.colorbar(im, ax=ax[2], fraction=0.046)
    for axis in ax:
        axis.axis("off")
    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("saved", a.out)


if __name__ == "__main__":
    main()
