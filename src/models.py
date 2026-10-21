"""Model definitions for probabilistic sketch-to-face generation.

Both classes reproduce the exact parameter names and shapes of the trained
checkpoints, so `load_state_dict(..., strict=True)` works on the released weights.
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """[Conv3x3 -> BatchNorm -> ReLU] x 2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------------------------------
# Model A: deterministic U-Net baseline (5.14 M parameters)
# --------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # baseline concatenates [skip, up] (Bayesian decoder uses [up, skip])
        return self.conv(torch.cat([skip, self.up(x)], dim=1))


class DeterministicUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, widths=(32, 64, 128, 256)):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.pool = nn.MaxPool2d(2)
        self.enc1 = EncoderBlock(in_ch, w1)
        self.enc2 = EncoderBlock(w1, w2)
        self.enc3 = EncoderBlock(w2, w3)
        self.enc4 = EncoderBlock(w3, w4)
        self.bottleneck = DoubleConv(w4, w4)
        self.dec4 = DecoderBlock(w4, w4, w4)
        self.dec3 = DecoderBlock(w4, w3, w3)
        self.dec2 = DecoderBlock(w3, w2, w2)
        self.dec1 = DecoderBlock(w2, w1, w1)
        self.output_layer = nn.Sequential(nn.Conv2d(w1, out_ch, 1), nn.Tanh())

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        d = self.dec4(b, s4)
        d = self.dec3(d, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)
        return self.output_layer(d)


# --------------------------------------------------------------------------
# Model B: Bayesian latent-variable U-Net (6.62 M parameters)
# --------------------------------------------------------------------------
class BayesianEncoder(nn.Module):
    def __init__(self, in_ch=1, widths=(64, 128, 256, 256), latent_ch=256):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_ch, w1)
        self.enc2 = DoubleConv(w1, w2)
        self.enc3 = DoubleConv(w2, w3)
        self.enc4 = DoubleConv(w3, w4)
        self.bottleneck = DoubleConv(w4, latent_ch)
        self.mu_layer = nn.Conv2d(latent_ch, latent_ch, 1)
        self.logvar_layer = nn.Conv2d(latent_ch, latent_ch, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        h = self.bottleneck(self.pool(s4))
        return self.mu_layer(h), self.logvar_layer(h), (s1, s2, s3, s4)


class BayesianDecoder(nn.Module):
    def __init__(self, out_ch=3, skip_widths=(64, 128, 256, 256), latent_ch=256):
        super().__init__()
        k1, k2, k3, k4 = skip_widths
        self.up4 = nn.ConvTranspose2d(latent_ch, 256, 2, stride=2)
        self.dec4 = DoubleConv(256 + k4, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(128 + k3, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(64 + k2, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(32 + k1, 32)
        self.output_layer = nn.Sequential(nn.Conv2d(32, out_ch, 3, padding=1), nn.Tanh())

    def forward(self, z, skips):
        s1, s2, s3, s4 = skips
        d = self.dec4(torch.cat([self.up4(z), s4], 1))
        d = self.dec3(torch.cat([self.up3(d), s3], 1))
        d = self.dec2(torch.cat([self.up2(d), s2], 1))
        d = self.dec1(torch.cat([self.up1(d), s1], 1))
        return self.output_layer(d)


class BayesianUNet(nn.Module):
    """x -> q(z|x) = N(mu, diag sigma^2) -> z -> p(y|x,z), with U-Net skips."""

    def __init__(self, in_ch=1, out_ch=3):
        super().__init__()
        self.encoder = BayesianEncoder(in_ch)
        self.decoder = BayesianDecoder(out_ch)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x, sample=True):
        mu, logvar, skips = self.encoder(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decoder(z, skips), mu, logvar

    @torch.no_grad()
    def predict(self, x, num_samples=8):
        """Monte-Carlo posterior predictive: mean face and per-pixel variance."""
        mu, logvar, skips = self.encoder(x)
        ys = torch.stack([self.decoder(self.reparameterize(mu, logvar), skips)
                          for _ in range(num_samples)])
        return ys.mean(0), ys.var(0, unbiased=False), ys


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
