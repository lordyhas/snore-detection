"""
Snore Detection - training loop with early stopping.

Author: Hassan Kajila
"""

import logging
import os
import random
import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.config import (
    MODEL_DIR,
    PROCESSED_DATA_FILE,
    RANDOM_SEED,
)
from src.data.data_manager import AudioDataset
from src.models.registry import MODELS, WAVEFORM_MODELS, build_model


logger = logging.getLogger(__name__)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    with torch.set_grad_enabled(is_training):
        for specs, labels in loader:
            specs = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(specs)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions.extend((logits.detach().view(-1) >= 0).cpu().int().tolist())
            targets.extend(labels.detach().view(-1).cpu().int().tolist())

    if not targets:
        raise ValueError("The training or validation dataset is empty.")

    return total_loss / len(targets), {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
    }


def train_model(
    epochs: int | None = None,
    *,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    early_stopping_patience: int | None = None,
    num_workers: int | None = None,
    sample_rate: int | None = None,
    duration: float | None = None,
    n_mels: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    model_name: str = "audio_cnn",
) -> str:
    """Train a model and save the checkpoint with the best validation loss."""
    if not os.path.isfile(PROCESSED_DATA_FILE):
        raise FileNotFoundError(
            f"Preprocessed data not found at {PROCESSED_DATA_FILE}; "
            "run src.data.preprocess.preprocess_data() first."
        )

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    with np.load(PROCESSED_DATA_FILE, allow_pickle=False) as data:
        required_arrays = ("x_train", "y_train", "x_val", "y_val")
        required_metadata = ("sample_rate", "duration")
        missing_keys = [
            key for key in required_arrays + required_metadata if key not in data
        ]
        if missing_keys:
            raise ValueError(
                f"Preprocessed dataset is missing fields: {', '.join(missing_keys)}"
            )
        x_train, y_train, x_val, y_val = (
            data[key].copy() for key in required_arrays
        )
        dataset_sample_rate = int(data["sample_rate"])
        dataset_duration = float(data["duration"])
    if sample_rate is not None and sample_rate != dataset_sample_rate:
        raise ValueError(
            f"sample_rate={sample_rate} does not match the preprocessed dataset "
            f"({dataset_sample_rate}); preprocess the data again with the desired rate."
        )
    if duration is not None and duration != dataset_duration:
        raise ValueError(
            f"duration={duration} does not match the preprocessed dataset "
            f"({dataset_duration}); preprocess the data again with the desired duration."
        )
    sample_rate = dataset_sample_rate
    duration = dataset_duration
    batch_size = 32 if batch_size is None else batch_size
    epochs = 50 if epochs is None else epochs
    learning_rate = 0.001 if learning_rate is None else learning_rate
    weight_decay = 0.01 if weight_decay is None else weight_decay
    early_stopping_patience = (
        8 if early_stopping_patience is None else early_stopping_patience
    )
    num_workers = 0 if num_workers is None else num_workers
    n_mels = 128 if n_mels is None else n_mels
    n_fft = 1024 if n_fft is None else n_fft
    hop_length = 512 if hop_length is None else hop_length

    if batch_size < 1 or epochs < 1 or early_stopping_patience < 1:
        raise ValueError("batch_size, epochs and early_stopping_patience must be positive.")
    if learning_rate <= 0 or weight_decay < 0 or num_workers < 0:
        raise ValueError(
            "learning_rate must be positive; weight_decay and num_workers cannot be negative."
        )
    if n_mels < 1 or n_fft < 1 or hop_length < 1:
        raise ValueError("n_mels, n_fft and hop_length must be positive.")

    if len(x_train) != len(y_train) or len(x_val) != len(y_val):
        raise ValueError("Preprocessed audio and label counts do not match.")

    train_dataset = AudioDataset(
        x_train,
        y_train,
        sample_rate=sample_rate,
        duration=duration,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        train=True,
        features=model_name not in WAVEFORM_MODELS,
    )
    val_dataset = AudioDataset(
        x_val,
        y_val,
        sample_rate=sample_rate,
        duration=duration,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        train=False,
        features=model_name not in WAVEFORM_MODELS,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = build_model(model_name).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3
    )

    num_epochs = epochs
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics = None
    epochs_without_improvement = 0
    best_state = None
    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_metrics = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info(
            "Epoch %d/%d - train loss: %.4f, val loss: %.4f, "
            "val accuracy: %.4f, val F1: %.4f",
            epoch,
            num_epochs,
            train_loss,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["f1"],
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                logger.info("Stopping early after epoch %d.", epoch)
                break

    if best_state is None:
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    model.load_state_dict(best_state)
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_name": model_name,
            "sample_rate": sample_rate,
            "duration": duration,
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "best_epoch": best_epoch,
            "validation_loss": best_val_loss,
            "validation_metrics": best_val_metrics,
        },
        model_path,
    )
    logger.info(
        "Saved best model to %s (epoch %d, val loss %.4f). "
        "Best validation metrics: %s",
        model_path,
        best_epoch,
        best_val_loss,
        best_val_metrics,
    )
    return model_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train the audio classification CNN.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-rate", "--sr",dest="sample_rate", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--n-mels", type=int, default=None)
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument(
        "--model", dest="model_name", choices=list(MODELS), default="audio_cnn"
    )
    args = parser.parse_args()
    print(f"Trained model saved to {train_model(**vars(args))}")