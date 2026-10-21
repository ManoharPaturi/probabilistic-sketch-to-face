import torch
from .models import DeterministicUNet, BayesianUNet


def load_model(kind, weights, device="cpu"):
    model = BayesianUNet() if kind == "bayesian" else DeterministicUNet()
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
