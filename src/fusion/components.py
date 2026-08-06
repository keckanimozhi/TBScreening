"""Load the already-trained, frozen X-ray and cough models and run
single-item inference, for late fusion (src/fusion/train.py combines
their output probabilities rather than retraining either backbone)."""

import torch
from PIL import Image

from src.common.config import MODELS_DIR
from src.cough.features import extract_log_mel
from src.xray.dataset import build_transforms as xray_transform
from src.xray.model import build_model as build_xray_model
from src.cough.model import build_model as build_cough_model

DEVICE = torch.device("cpu")


def load_xray_model():
    model = build_xray_model().to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / "xray" / "resnet18_best.pt", map_location=DEVICE))
    model.eval()
    return model


def load_cough_model():
    model = build_cough_model().to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / "cough" / "cnn_best.pt", map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def predict_xray(model, filepath: str) -> float:
    image = Image.open(filepath).convert("RGB")
    tensor = xray_transform(train=False)(image).unsqueeze(0).to(DEVICE)
    logit = model(tensor).squeeze()
    return torch.sigmoid(logit).item()


@torch.no_grad()
def predict_cough(model, filepath: str) -> float:
    mel = extract_log_mel(filepath)
    tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
    logit = model(tensor).squeeze()
    return torch.sigmoid(logit).item()
