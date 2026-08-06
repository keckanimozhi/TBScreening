"""Export the trained cough CNN to ONNX (Task 15), verified against
PyTorch on the held-out test set.

Usage:
    .venv/Scripts/python.exe -m src.export.export_cough
"""

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from src.common.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.cough.features import extract_log_mel
from src.cough.model import build_model

ONNX_OUT = MODELS_DIR / "onnx" / "cough.onnx"


def export():
    model = build_model()
    model.load_state_dict(torch.load(MODELS_DIR / "cough" / "cnn_best.pt", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 1, 64, 173)  # (batch, channel, n_mels, n_frames)
    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_OUT),
        input_names=["log_mel"],
        output_names=["logit"],
        opset_version=13,
        dynamic_axes=None,
        dynamo=False,
    )
    print(f"Exported ONNX model to {ONNX_OUT}")
    return model


def verify(torch_model):
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "cough_test.csv")

    session = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    max_abs_diff = 0.0
    for _, row in test_df.iterrows():
        mel = extract_log_mel(row["filepath"])
        tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            torch_logit = torch_model(tensor).squeeze().item()

        onnx_logit = session.run(None, {input_name: tensor.numpy().astype(np.float32)})[0].squeeze()
        diff = abs(torch_logit - float(onnx_logit))
        max_abs_diff = max(max_abs_diff, diff)

    print(f"Max abs logit difference (PyTorch vs ONNX) over {len(test_df)} test clips: {max_abs_diff:.6f}")
    assert max_abs_diff < 1e-3, "ONNX export diverges from PyTorch model beyond tolerance"
    print("ONNX export verified: matches PyTorch model within tolerance.")


def main():
    model = export()
    verify(model)


if __name__ == "__main__":
    main()
