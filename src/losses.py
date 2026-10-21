"""Losses and metrics: L1 + 0.2 (1 - SSIM), KL divergence, KL warm-up, PSNR.

All faces live in [-1, 1] (Tanh output), matching the original Kaggle notebook.
"""
import torch
import torch.nn.functional as F


def _gaussian_window(size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    w = (g.t() @ g).expand(channels, 1, size, size).contiguous()
    return w


def ssim(x, y, data_range=2.0, size=11, sigma=1.5):
    """Mean SSIM (Wang et al., 2004); images are in [-1, 1], so data_range = 2."""
    c = x.shape[1]
    w = _gaussian_window(size, sigma, c, x.device).to(x.dtype)
    mu_x = F.conv2d(x, w, padding=size // 2, groups=c)
    mu_y = F.conv2d(y, w, padding=size // 2, groups=c)
    sxx = F.conv2d(x * x, w, padding=size // 2, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=size // 2, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=size // 2, groups=c) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean()


def reconstruction_loss(pred, target, ssim_weight=0.2):
    """l_rec = mean|y - y_hat| + 0.2 (1 - SSIM(y, y_hat))."""
    l1 = F.l1_loss(pred, target)
    s = ssim(pred, target)
    return l1 + ssim_weight * (1.0 - s), l1, s


def kl_divergence(mu, logvar, reduction="sum"):
    """KL(N(mu, sigma^2) || N(0, I)).

    reduction="sum": summed over the d = 256*14*14 latent units, averaged over the batch.
    reduction="mean": averaged over every latent unit (a much weaker regulariser).
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.flatten(1).sum(1).mean() if reduction == "sum" else kl.mean()


def kl_weight(epoch, beta_max=1e-3, warmup_epochs=10):
    """Linear KL warm-up: beta(t) = beta_max * min(1, t / T_w), epochs counted from 1."""
    return beta_max * min(1.0, epoch / warmup_epochs)


def psnr(pred, target, data_range=2.0):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(data_range ** 2 / mse.clamp_min(1e-12))
