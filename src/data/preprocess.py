import os
import tempfile

import librosa
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DATA_FILE,
    RANDOM_SEED,
    RAW_DATA_DIR,
)


def _load_class_audio(
    class_dir: str, label: int, sample_rate: int, duration: float
) -> tuple[list[np.ndarray], list[int]]:
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"Audio class directory does not exist: {class_dir}")

    files = sorted(
        os.path.join(class_dir, name)
        for name in os.listdir(class_dir)
        if name.lower().endswith(".wav")
    )
    if not files:
        raise ValueError(f"No WAV files found in audio class directory: {class_dir}")

    target_samples = int(sample_rate * duration)
    waveforms: list[np.ndarray] = []
    labels: list[int] = []
    for file_path in files:
        try:
            waveform, _ = librosa.load(file_path, sr=sample_rate, mono=True)
        except Exception as error:
            raise RuntimeError(f"Could not load audio file {file_path}") from error

        waveform = np.asarray(waveform[:target_samples], dtype=np.float32)
        if waveform.size < target_samples:
            waveform = np.pad(waveform, (0, target_samples - waveform.size))
        waveforms.append(waveform)
        labels.append(label)

    return waveforms, labels


def preprocess_data(
    sample_rate: int | None = None,
    duration: float | None = None,
    validation_split: float | None = None,
) -> str:
    """Split raw class folders and save fixed-length audio arrays for training."""
    sample_rate = 16000 if sample_rate is None else sample_rate
    duration = 3.0 if duration is None else duration
    validation_split = 0.2 if validation_split is None else validation_split
    if sample_rate <= 0 or duration <= 0:
        raise ValueError("sample_rate and duration must be positive.")
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be greater than 0 and less than 1.")

    waveforms_0, labels_0 = _load_class_audio(
        os.path.join(RAW_DATA_DIR, "0"), 0, sample_rate, duration
    )
    waveforms_1, labels_1 = _load_class_audio(
        os.path.join(RAW_DATA_DIR, "1"), 1, sample_rate, duration
    )

    waveforms = np.stack(waveforms_0 + waveforms_1)
    labels = np.asarray(labels_0 + labels_1, dtype=np.int64)
    try:
        x_train, x_val, y_train, y_val = train_test_split(
            waveforms,
            labels,
            test_size=validation_split,
            random_state=RANDOM_SEED,
            stratify=labels,
        )
    except ValueError as error:
        raise ValueError(
            "Could not create a stratified train/validation split; "
            "each class must contain enough WAV files."
        ) from error

    processed_data_dir = os.path.dirname(PROCESSED_DATA_FILE)
    os.makedirs(processed_data_dir, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=processed_data_dir, delete=False
        ) as temp_file:
            temp_path = temp_file.name
            np.savez_compressed(
                temp_file,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                sample_rate=np.asarray(sample_rate),
                duration=np.asarray(duration),
            )
        os.replace(temp_path, PROCESSED_DATA_FILE)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return PROCESSED_DATA_FILE


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess the raw audio dataset.")
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--validation-split", type=float, default=None)
    args = parser.parse_args()
    print(
        "Preprocessed dataset saved to "
        f"{preprocess_data(args.sample_rate, args.duration, args.validation_split)}"
    )