"""Train the deterministic U-Net or the Bayesian latent U-Net.

python train.py --model bayesian --data-root /path/to/utkface_aligned_cropped
"""
import argparse, json, os
import torch
from torch.utils.data import DataLoader
from src.dataset import SketchFaceDataset
from src.losses import reconstruction_loss, kl_divergence, kl_weight, psnr
from src.models import DeterministicUNet, BayesianUNet, count_parameters


def run_epoch(model, loader, bayesian, beta, device, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    tot = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "psnr": 0.0, "kl": 0.0}
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=device.type == "cuda"):
            if bayesian:
                pred, mu, logvar = model(x)
                kl = kl_divergence(mu.float(), logvar.float())
            else:
                pred, kl = model(x), torch.zeros((), device=device)
            rec, l1, s = reconstruction_loss(pred.float(), y)
            loss = rec + beta * kl
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        b = x.size(0); n += b
        for k, v in zip(tot, (loss, l1, s, psnr(pred.float(), y), kl)):
            tot[k] += v.item() * b
    return {k: v / n for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["deterministic", "bayesian"], default="bayesian")
    ap.add_argument("--data-root", default=None, help="local folder replacing the Kaggle path prefix")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta-max", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mk = lambda f, sh: DataLoader(SketchFaceDataset(os.path.join(a.splits, f), a.data_root),
                                  batch_size=a.batch_size, shuffle=sh, num_workers=2, pin_memory=True)
    train_dl, val_dl = mk("train.csv", True), mk("val.csv", False)

    bayesian = a.model == "bayesian"
    model = (BayesianUNet() if bayesian else DeterministicUNet()).to(device)
    print(f"{a.model}: {count_parameters(model) / 1e6:.2f} M parameters on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    os.makedirs(a.out, exist_ok=True)
    best, history = float("inf"), []
    for epoch in range(1, a.epochs + 1):
        beta = kl_weight(epoch, a.beta_max, a.warmup) if bayesian else 0.0
        tr = run_epoch(model, train_dl, bayesian, beta, device, opt, scaler)
        va = run_epoch(model, val_dl, bayesian, beta, device)
        sched.step(va["loss"])
        history.append({"epoch": epoch, "beta": beta, "lr": opt.param_groups[0]["lr"],
                        "train": tr, "val": va})
        print(f"ep {epoch:02d} beta={beta:.0e} train {tr['loss']:.4f} | val loss {va['loss']:.4f} "
              f"SSIM {va['ssim']:.4f} PSNR {va['psnr']:.2f}")
        state = {"epoch": epoch, "model_state_dict": model.state_dict(), "architecture": a.model}
        torch.save(state, os.path.join(a.out, f"latest_{a.model}.pth"))
        if va["loss"] < best:
            best = va["loss"]
            torch.save(state, os.path.join(a.out, f"best_{a.model}.pth"))
    json.dump(history, open(os.path.join(a.out, f"{a.model}_history.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
