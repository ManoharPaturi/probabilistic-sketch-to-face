# Results

All numbers come from the original Kaggle run and are stored in [`results/`](../results/).

## Test set (2,371 unseen pairs)

| Model | Params | Loss ↓ | L1 ↓ | SSIM ↑ | PSNR (dB) ↑ |
|---|---|---|---|---|---|
| Deterministic U-Net | 5.14 M | 0.1629 | 0.1115 | 0.7430 | 22.28 |
| **Bayesian latent U-Net** | 6.62 M | **0.1566** | **0.1066** | **0.7500** | **22.66** |
| Difference | | −3.84 % | −4.36 % | +0.94 % | +0.38 dB |

Uncertainty statistics (Bayesian model, S = 8 samples, std in [-1, 1] space): mean 0.00225, median 0.00215, max 0.0120.

## Training behaviour

![Validation SSIM and test metrics](../results/figures/chart_val.png)

* Both models converge in 30 epochs. The plateau scheduler halved the learning rate four times.
* Validation curves fluctuate early on (BatchNorm statistics + mixed precision) and settle once the learning rate drops.
* The deterministic model reaches a **lower training loss** but not a better validation score. Train-vs-test SSIM gap: **0.066** (deterministic) vs **0.027** (Bayesian).

## Qualitative

![Outputs, uncertainty and samples](../results/figures/demo.png)

* Face geometry (eyes, nose, smile, jaw) is reproduced faithfully thanks to the skip connections.
* Skin tone and hair colour are plausible guesses; outputs are slightly smooth.
* Uncertainty concentrates at eye corners, nostrils, mouth corners and hair/background edges.
* The eight samples are nearly identical (see the posterior-collapse note below).

## Comparison with recent literature (indicative only)

| Method | Venue | Dataset | SSIM | PSNR (dB) |
|---|---|---|---|---|
| CA-GAN (Yu et al.), as reported by Ren et al. | IEEE T-Cyb 2021 | CUHK | 0.702 | 18.47 |
| Ren, Yang & Wei | Sensors 2022 | CUHK | 0.688 | 17.90 |
| FaceMamba (Tang et al.) | Sci. Rep. 2024 | CUFS | 0.700 | 15.66 |
| **This work (Bayesian U-Net)** | | UTKFace | **0.750** | **22.66** |

The datasets differ: CUHK/CUFS are a few hundred hand-drawn forensic sketches, while ours are 23.7 k synthetic sketches. This is **not** a head-to-head comparison.

## Interpreting the gain

Because the posterior collapsed (μ ≈ 0, σ ≈ 1), the latent variable carries no information about the input. The Bayesian model's small advantage is most likely due to its **wider encoder** (64-128-256-256 vs 32-64-128-256) rather than to probabilistic modelling. A fair ablation would train a deterministic U-Net with the same widths.

Next: [Reproducibility & limitations](reproducibility.md)
