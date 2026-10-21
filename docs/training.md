# Training and evaluation

## Dataset

* **Source:** [UTKFace](https://susanqq.github.io/UTKFace/) aligned and cropped faces (23,708 images; ages 0–116, both genders, five ethnicity groups) [Zhang, Song & Qi, CVPR 2017].
* **Sketches:** one grayscale pencil-style sketch per face, paired by file name (`<name>.jpg.chip.jpg` ↔ `<name>.jpg.chip_sketch.jpg`).
* **Split:** fixed random 80 / 10 / 10 split, stored in `data/splits/`:

| File | Pairs |
|---|---|
| `train.csv` | 18,966 |
| `val.csv` | 2,371 |
| `test.csv` | 2,371 |

Each row has `pair_id, face_path, sketch_path, split, target_size`. The paths are the original Kaggle paths, for example:

```
/kaggle/input/datasets/manoharpaturi/my-paired-sketch-dataset/utkface_aligned_cropped/UTKFace/<name>.jpg.chip.jpg
/kaggle/input/datasets/manoharpaturi/my-paired-sketch-dataset/utkface_aligned_cropped/sketches/<name>.jpg.chip_sketch.jpg
```

To run locally, arrange your copy as `<root>/UTKFace/…` and `<root>/sketches/…` and pass `--data-root <root>`. Everything up to and including `utkface_aligned_cropped/` is replaced by `<root>`. On Kaggle with the original dataset attached, omit `--data-root`.

## Pre-processing

Implemented in [`src/preprocess.py`](../src/preprocess.py), identical to the training notebook:

| | Sketch | Face |
|---|---|---|
| Read | OpenCV, grayscale | OpenCV, BGR → RGB |
| Resize | 224 × 224, `INTER_AREA` | 224 × 224, `INTER_AREA` |
| Scale | /255 → [0, 1] | /255 → [0, 1] → ×2 − 1 → [-1, 1] |
| Tensor | 1 × 224 × 224 | 3 × 224 × 224 |

No data augmentation was used.

## Hyper-parameters

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 2 × 10⁻⁴, betas (0.9, 0.999), weight decay 10⁻⁵ |
| LR schedule | ReduceLROnPlateau on validation loss (factor 0.5, patience 3, min 10⁻⁶) |
| Batch size | 16 (1,186 steps per epoch) |
| Epochs | 30 |
| Precision | mixed (FP16 autocast + GradScaler) on CUDA |
| Loss | L1 + 0.2 (1 − SSIM) [+ β(t) · KL for the Bayesian model] |
| KL schedule | β(t) = 10⁻³ · min(1, t/10) |
| Model selection | lowest validation loss (deterministic: epoch 24, Bayesian: epoch 29) |
| Hardware | single Kaggle GPU |

## Train

```bash
python train.py --model bayesian      --data-root /path/to/utkface_aligned_cropped
python train.py --model deterministic --data-root /path/to/utkface_aligned_cropped
```

Options: `--epochs`, `--batch-size`, `--lr`, `--beta-max`, `--warmup`, `--splits`, `--out`.
Checkpoints go to `checkpoints/` (`best_<model>.pth`, `latest_<model>.pth`) together with `<model>_history.json`.

## Evaluate

```bash
python evaluate.py --model bayesian \
                   --weights weights/bayesian_sketch_to_face_final.pth \
                   --data-root /path/to/utkface_aligned_cropped
```

Prints loss, L1, SSIM and PSNR on the test split (plus mean uncertainty for the Bayesian model). All metrics are computed in [-1, 1] space: SSIM uses `data_range = 2` (11×11 Gaussian window, σ = 1.5), and PSNR uses a maximum value of 2.

Note: `train.py` and `evaluate.py` re-implement the notebook's training loop; the released numbers come from the original Kaggle run. Re-training will give similar but not bit-identical results.

Next: [Results](results.md)
