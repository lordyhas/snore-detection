"""
Snore Detection - inference on new recordings with an exported ONNX model.

Author: Hassan Kajila
"""

import argparse
import os

import librosa
import numpy as np
import onnxruntime as ort

from src.config import MODEL_DIR
from src.models.registry import WAVEFORM_MODELS


def _windows(waveform: np.ndarray, num_samples: int, hop_samples: int) -> np.ndarray:
    """Cut a recording into fixed-length windows, zero-padding the last one."""
    if waveform.size <= num_samples:
        starts = [0]
    else:
        starts = list(range(0, waveform.size - num_samples + 1, hop_samples))
        if starts[-1] + num_samples < waveform.size:
            starts.append(starts[-1] + hop_samples)
    windows = np.zeros((len(starts), num_samples), dtype=np.float32)
    for row, start in enumerate(starts):
        chunk = waveform[start:start + num_samples]
        windows[row, :chunk.size] = chunk
    return windows


def predict_file(session: ort.InferenceSession, file_path: str) -> np.ndarray:
    """Return the snoring probability of each window of a recording."""
    metadata = session.get_modelmeta().custom_metadata_map
    sample_rate = int(metadata["sample_rate"])
    num_samples = int(metadata["num_samples"])
    waveform, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    windows = _windows(waveform, num_samples, hop_samples=num_samples // 2)
    return session.run(None, {"waveform": windows})[0].reshape(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect snoring in audio files.")
    parser.add_argument("files", nargs="+", help="Audio files (WAV, MP3, ...).")
    parser.add_argument(
        "--model", dest="model_name", choices=list(WAVEFORM_MODELS), default="mobile_small"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    onnx_path = os.path.join(MODEL_DIR, f"snore_{args.model_name}.onnx")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    for file_path in args.files:
        probabilities = predict_file(session, file_path)
        snoring = float((probabilities >= args.threshold).mean())
        verdict = "snoring" if snoring >= 0.5 else "no snoring"
        print(
            f"{os.path.basename(file_path)}: {verdict} "
            f"({snoring:.0%} of {probabilities.size} windows, "
            f"mean probability {probabilities.mean():.2f})"
        )
