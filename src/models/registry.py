"""
Snore Detection - model names available for training, export and inference.

Author: Hassan Kajila
"""

from src.models.audio_cnn import AudioCNN
from src.models.mobile_cnn import SnoreMobileNet


MODELS = {
    "audio_cnn": AudioCNN,
    "mobile_small": lambda: SnoreMobileNet(width=32),
    "mobile_tiny": lambda: SnoreMobileNet(width=16),
}

# Models that take the raw waveform and compute their own spectrogram
WAVEFORM_MODELS = ("mobile_small", "mobile_tiny")


def build_model(name: str):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'; choose from {', '.join(MODELS)}.")
    return MODELS[name]()
