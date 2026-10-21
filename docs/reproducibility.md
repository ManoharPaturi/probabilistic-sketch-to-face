# Reproducibility and limitations

## Verified

* Both released checkpoints load into `src/models.py` with `strict=True`.
* With the pre-processing below, the Bayesian model reproduces the saved test-set figures. The mean per-pixel std over 8 samples is 0.0026, against 0.00225 reported over the full test set.
* A fresh clone plus a release download runs `infer.py` end to end.

## Pre-processing must match training

The model's colour output is **sensitive to the input**: in our checks, changing sketch pixels by 0.2 % on average visibly shifted the generated skin tone. Always use [`src/preprocess.py`](../src/preprocess.py):

* OpenCV `cv2.imread` (grayscale for sketches, BGR → RGB for faces)
* `cv2.resize(…, (224, 224), interpolation=cv2.INTER_AREA)`
* sketch / 255 → [0, 1]; face / 255 × 2 − 1 → [-1, 1]

Using PIL or torchvision bilinear resizing instead gives noticeably different colours.

## Checksums (release v1.0)

```
5e9db02ce58e73fd9fec8a5a3e320e1003b4d624fcec9e78568cec73029ca587  bayesian_sketch_to_face_final.pth
c4cece61e88cdf1963cb4dc38a3aaaaaafaabdcbf251538ad5fb8a918d919d36  deterministic_sketch_to_face_final.pth
```

## Posterior collapse

On test sketches the trained encoder outputs **μ ≈ 0 (|μ| ≈ 10⁻⁵) and σ ≈ 1.000**, so q_φ(z | x) ≈ p(z) = N(0, I) and the KL term is ≈ 0. Consequences:

1. **z does not depend on the sketch.** It is pure prior noise.
2. **The uncertainty map is still meaningful, but narrower than intended.** It measures how much the decoder's output moves when the latent noise changes. The spatial pattern does depend on the sketch (through the skip connections), and it highlights ambiguous regions, but it is not a learned input-dependent posterior.
3. **Samples are nearly identical.** The mean pixel std is about 0.002 on a [-1, 1] scale.
4. **The accuracy gain is probably architectural** (wider encoder), not probabilistic.

Why it happens: the skip connections give the decoder everything it needs to reconstruct the face, so the cheapest way to minimise the KL term is to ignore z entirely.

Planned fixes:

* **free bits**: only penalise KL above a minimum number of nats per latent group
* **cyclical** β schedules, or a larger β_max with a longer warm-up
* weaker skips (dropout on skip features, or skips only at high resolution) so that z must carry appearance information
* an ablation against a deterministic U-Net of the same width

## Other limitations

* Sketches are synthetic (generated from the photos); results on hand-drawn forensic sketches (CUFS / CUFSF) are untested.
* No FID / LPIPS or identity-preservation (face recognition) metrics yet.
* The accept-vs-review decision threshold τ is not calibrated.
* `train.py` re-implements the notebook's training loop and has not been re-run end to end; the released weights come from the original notebook.
* UTKFace images are not redistributed here; see its own licence terms.
