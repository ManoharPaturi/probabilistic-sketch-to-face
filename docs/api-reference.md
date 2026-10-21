# API reference

All tensors follow PyTorch's `[batch, channels, height, width]` layout. Sketches are in [0, 1]; faces are in [-1, 1].

## `src/models.py`

### `DoubleConv(in_ch, out_ch)`
`[Conv3×3 (no bias) → BatchNorm → ReLU] × 2`. Attribute `block` (an `nn.Sequential`).

### `DeterministicUNet(in_ch=1, out_ch=3, widths=(32, 64, 128, 256))`
Baseline U-Net. `forward(x) → y` with `x: [B, 1, 224, 224]`, `y: [B, 3, 224, 224]` in [-1, 1].

### `BayesianUNet(in_ch=1, out_ch=3)`
Latent-variable U-Net with sub-modules `encoder: BayesianEncoder` and `decoder: BayesianDecoder`.

| Method | Returns | Notes |
|---|---|---|
| `forward(x, sample=True)` | `(y, mu, logvar)` | `sample=False` decodes z = μ (no noise) |
| `predict(x, num_samples=8)` | `(mean, var, samples)` | no-grad Monte-Carlo inference; `samples: [S, B, 3, H, W]` |
| `reparameterize(mu, logvar)` (static) | `z` | `mu + exp(0.5·logvar) * randn_like(mu)` |

Per-pixel uncertainty as reported in the project: `samples.std(0)` (optionally `.mean(1)` over RGB).

### `BayesianEncoder(in_ch=1, widths=(64, 128, 256, 256), latent_ch=256)`
`forward(x) → (mu, logvar, (s1, s2, s3, s4))`; `mu`, `logvar`: `[B, 256, 14, 14]`.

### `BayesianDecoder(out_ch=3, skip_widths=(64, 128, 256, 256), latent_ch=256)`
`forward(z, skips) → y`.

### `count_parameters(model) → int`

## `src/losses.py`

| Function | Description |
|---|---|
| `ssim(x, y, data_range=2.0, size=11, sigma=1.5)` | mean SSIM with a Gaussian window |
| `reconstruction_loss(pred, target, ssim_weight=0.2)` | returns `(l1 + w·(1−ssim), l1, ssim)` |
| `kl_divergence(mu, logvar, reduction="sum")` | KL to N(0, I); `"sum"` over latent units then batch-mean (as in training), `"mean"` per unit |
| `kl_weight(epoch, beta_max=1e-3, warmup_epochs=10)` | linear warm-up β(t); epochs start at 1 |
| `psnr(pred, target, data_range=2.0)` | peak signal-to-noise ratio in dB |

## `src/preprocess.py`

| Function | Description |
|---|---|
| `load_sketch(path, size=224)` | OpenCV grayscale, `INTER_AREA` resize → `[1, H, W]` in [0, 1] |
| `load_face(path, size=224)` | OpenCV RGB, `INTER_AREA` resize → `[3, H, W]` in [-1, 1] |
| `to_image(t)` | `[3, H, W]` in [-1, 1] → `H×W×3` NumPy array in [0, 1] |

## `src/dataset.py`

### `SketchFaceDataset(csv_path, root=None, image_size=224)`
Returns `(sketch, face)` pairs from a split CSV. `root` remaps the stored Kaggle paths (see [Training](training.md#dataset)).

## `src/utils.py`

### `load_model(kind, weights, device="cpu")`
`kind` is `"bayesian"` or `"deterministic"`. Loads a checkpoint (either a raw state dict or a dict with `model_state_dict`) with `strict=True` and returns the model in eval mode.

## Command-line scripts

| Script | Required arguments | Optional |
|---|---|---|
| `infer.py` | `--sketch`, `--weights` | `--model {bayesian,deterministic}`, `--samples 8`, `--out result.png` |
| `evaluate.py` | `--model`, `--weights` | `--data-root`, `--split data/splits/test.csv`, `--samples 8` |
| `train.py` | none | `--model`, `--data-root`, `--splits`, `--epochs 30`, `--batch-size 16`, `--lr 2e-4`, `--beta-max 1e-3`, `--warmup 10`, `--out checkpoints` |
