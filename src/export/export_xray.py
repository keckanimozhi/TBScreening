"""Export the trained X-ray ResNet18 to ONNX for on-device inference via
ONNX Runtime Mobile (Task 15), and verify the exported graph produces the
same predictions as the original PyTorch model on the held-out test set.

Usage:
    .venv/Scripts/python.exe -m src.export.export_xray
"""

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from src.common.config import IMAGE_SIZE, MODELS_DIR, PROCESSED_DATA_DIR
from src.xray.dataset import build_transforms
from src.xray.model import build_model
from PIL import Image

ONNX_OUT = MODELS_DIR / "onnx" / "xray.onnx"


def export():
    model = build_model()
    model.load_state_dict(torch.load(MODELS_DIR / "xray" / "resnet18_best.pt", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_OUT),
        input_names=["image"],
        output_names=["logit"],
        opset_version=13,
        dynamic_axes=None,  # fixed batch=1, fixed 224x224 -- simplest/most compatible for mobile
        dynamo=False,  # legacy TorchScript-based exporter: stable for a static-shape inference CNN,
                        # and avoids the newer dynamo exporter's console-encoding crash on Windows
    )
    print(f"Exported ONNX model to {ONNX_OUT}")
    return model


def verify(torch_model):
    transform = build_transforms(train=False)
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "xray_test.csv")

    session = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    max_abs_diff = 0.0
    for _, row in test_df.iterrows():
        image = Image.open(row["filepath"]).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            torch_logit = torch_model(tensor).squeeze().item()

        onnx_logit = session.run(None, {input_name: tensor.numpy().astype(np.float32)})[0].squeeze()
        diff = abs(torch_logit - float(onnx_logit))
        max_abs_diff = max(max_abs_diff, diff)

    print(f"Max abs logit difference (PyTorch vs ONNX) over {len(test_df)} test images: {max_abs_diff:.6f}")
    assert max_abs_diff < 1e-3, "ONNX export diverges from PyTorch model beyond tolerance"
    print("ONNX export verified: matches PyTorch model within tolerance.")


def main():
    model = export()
    verify(model)


if __name__ == "__main__":
    main()
