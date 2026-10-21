# Getting started

## 1. Requirements

* Python 3.9 or newer
* PyTorch 2.1 or newer (a GPU is only needed for training; inference runs on CPU)

```bash
git clone https://github.com/ManoharPaturi/probabilistic-sketch-to-face.git
cd probabilistic-sketch-to-face
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download the trained weights

The weights are attached to the [v1.0 release](https://github.com/ManoharPaturi/probabilistic-sketch-to-face/releases/tag/v1.0) and are not stored in git.

```bash
gh release download v1.0 -D weights -R ManoharPaturi/probabilistic-sketch-to-face
```

Or download both `.pth` files from the release page by hand and place them in `weights/`:

```
weights/
├── bayesian_sketch_to_face_final.pth        (26.5 MB, 6.62 M parameters)
└── deterministic_sketch_to_face_final.pth   (61.8 MB, 5.14 M parameters, includes optimiser state)
```

Check the files against the SHA-256 sums in the release notes if you want to be sure they downloaded correctly.

## 3. Generate a face from a sketch

```bash
python infer.py --sketch path/to/sketch.png \
                --weights weights/bayesian_sketch_to_face_final.pth \
                --out result.png
```

`result.png` shows three panels: the input sketch, the generated face (mean of 8 samples) and the per-pixel uncertainty map.

For the deterministic baseline (no uncertainty map):

```bash
python infer.py --model deterministic --sketch path/to/sketch.png \
                --weights weights/deterministic_sketch_to_face_final.pth --out result_det.png
```

### What makes a good input sketch

The models were trained on UTKFace-style inputs: an aligned, cropped, roughly frontal face drawn as dark pencil-like strokes on a white background. Any image size works (it is resized to 224×224), but crop tightly around the face. Inputs that look very different (thick cartoon lines, colour, a full upper body) will give poor results. Skin tone in particular is a guess: the sketch contains no colour information.

## 4. Use it from Python

```python
import torch
from src.preprocess import load_sketch, to_image
from src.utils import load_model

model = load_model("bayesian", "weights/bayesian_sketch_to_face_final.pth")
x = load_sketch("sketch.png").unsqueeze(0)          # [1, 1, 224, 224] in [0, 1]
mean, var, samples = model.predict(x, num_samples=8) # faces in [-1, 1]
face = to_image(mean[0])                             # H×W×3 float array in [0, 1]
uncertainty = samples.std(0).mean(1)[0]              # 224×224 per-pixel std
```

Next: [Methodology](methodology.md) · [API reference](api-reference.md)
