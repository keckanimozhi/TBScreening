"""Grad-CAM explainability for the chest X-ray TB classifier.

Implemented directly with forward/backward hooks on `layer4` (the last
convolutional block) rather than an external grad-cam library, since the
model is a single frozen-mostly ResNet18 and the technique is a handful of
lines -- one less third-party dependency to pin/version.

Usage:
    .venv/Scripts/python.exe -m src.xray.gradcam
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image

from src.common.config import IMAGE_SIZE, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.xray.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from src.xray.model import build_model


class GradCAM:
    """Grad-CAM for a target conv layer of a binary (single-logit) model."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor) -> np.ndarray:
        """image_tensor: (1, C, H, W). Returns a (H, W) CAM in [0, 1],
        explaining evidence toward the TB (positive) class."""
        self.model.zero_grad()
        logit = self.model(image_tensor).squeeze()
        logit.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def unnormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) normalized tensor -> (H, W, C) uint8 array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = image_tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def overlay_heatmap(image_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    heatmap = cm.jet(cam)[:, :, :3]  # (H, W, 3) in [0,1]
    heatmap = (heatmap * 255).astype(np.uint8)
    blended = (image_uint8 * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def gradcam_overlay_for_image(model: torch.nn.Module, filepath: str) -> Image.Image:
    """Single-image convenience wrapper reused by the Streamlit app (Task
    11) -- returns just the overlay for one file, without the batch
    scoring/logging that `main()` below does over the whole test set."""
    gradcam = GradCAM(model, model.layer4)
    transform = build_transforms(train=False)

    image = Image.open(filepath).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    cam = gradcam(tensor)

    display_img = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
    return overlay_heatmap(display_img, cam)


def main():
    device = torch.device("cpu")  # Grad-CAM here is single-image, CPU is fine.

    model = build_model().to(device)
    ckpt_path = MODELS_DIR / "xray" / "resnet18_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    gradcam = GradCAM(model, model.layer4)
    transform = build_transforms(train=False)

    test_df = pd.read_csv(PROCESSED_DATA_DIR / "xray_test.csv")

    out_dir = REPORTS_DIR / "xray" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in test_df.iterrows():
        image = Image.open(row["filepath"]).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            prob = torch.sigmoid(model(tensor).squeeze()).item()
        pred_label = "TB" if prob >= 0.5 else "Normal"
        true_label = "TB" if row["label"] == 1 else "Normal"

        cam = gradcam(tensor)

        display_img = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        overlay = overlay_heatmap(display_img, cam)

        out_name = f"{row['filename'].replace('.png', '')}_true-{true_label}_pred-{pred_label}_p{prob:.2f}.png"
        overlay.save(out_dir / out_name)

    print(f"Saved {len(test_df)} Grad-CAM overlays to {out_dir}")


if __name__ == "__main__":
    main()
