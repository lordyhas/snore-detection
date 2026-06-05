"""
Snore Detection - export a trained mobile model to ONNX for on-device inference.

Author: Hassan Kajila
"""

import argparse
import logging
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.config import MODEL_DIR, PROCESSED_DATA_FILE
from src.models.registry import WAVEFORM_MODELS, build_model


logger = logging.getLogger(__name__)

OPSET_VERSION = 18


class _ProbabilityModel(torch.nn.Module):
    """Adds the sigmoid, so the app reads a probability directly."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        return torch.sigmoid(self.model(waveform))


def export_model(model_name: str = "mobile_small") -> str:
    """Export models/<model_name>.pth to models/snore_<model_name>.onnx and check it."""
    if model_name not in WAVEFORM_MODELS:
        raise ValueError(
            f"Only waveform models can be exported: {', '.join(WAVEFORM_MODELS)}."
        )
    checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}; "
            f"run `python -m src.train --model {model_name}` first."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sample_rate = int(checkpoint["sample_rate"])
    num_samples = int(sample_rate * checkpoint["duration"])
    model = build_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    wrapper = _ProbabilityModel(model).eval()

    onnx_path = os.path.join(MODEL_DIR, f"snore_{model_name}.onnx")
    torch.onnx.export(
        wrapper,
        (torch.zeros(2, num_samples),),
        onnx_path,
        input_names=["waveform"],
        output_names=["probability"],
        dynamic_shapes={"waveform": {0: torch.export.Dim("batch")}},
        opset_version=OPSET_VERSION,
        dynamo=True,
        external_data=False,
    )

    # Keep what the app needs to know next to the weights
    onnx_model = onnx.load(onnx_path)
    del onnx_model.metadata_props[:]
    for key, value in {
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "input": "mono waveform, float32 in [-1, 1]",
        "output": "snoring probability in [0, 1]",
    }.items():
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = key, str(value)
    onnx.save(onnx_model, onnx_path)

    # Check the exported model against PyTorch on the validation set
    with np.load(PROCESSED_DATA_FILE, allow_pickle=False) as data:
        x_val = data["x_val"].astype(np.float32)
        y_val = data["y_val"]
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_probabilities = session.run(None, {"waveform": x_val})[0].reshape(-1)
    with torch.no_grad():
        torch_probabilities = wrapper(torch.from_numpy(x_val)).reshape(-1).numpy()
    max_difference = float(np.abs(onnx_probabilities - torch_probabilities).max())
    accuracy = float(((onnx_probabilities >= 0.5) == (y_val == 1)).mean())
    logger.info(
        "Exported %s (%.0f KB) - validation accuracy %.4f, "
        "max difference with PyTorch %.2e",
        onnx_path,
        os.path.getsize(onnx_path) / 1024,
        accuracy,
        max_difference,
    )
    if not max_difference <= 1e-2:  # also catches NaN
        raise RuntimeError("The exported model does not match the PyTorch model.")
    return onnx_path


if __name__ == "__main__":
    # the exporter prints symbols that a legacy Windows console cannot encode
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Export a mobile model to ONNX.")
    parser.add_argument(
        "--model", dest="model_name", choices=list(WAVEFORM_MODELS), default="mobile_small"
    )
    args = parser.parse_args()
    print(f"Exported model saved to {export_model(args.model_name)}")
