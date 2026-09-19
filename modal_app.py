"""
Snore Detection - run preprocessing, training and export on Modal (cloud GPU).

    modal run modal_app.py --upload                 # first time: send data/raw, then train
    modal run modal_app.py --model mobile_tiny      # train (and export) one model

Author: Hassan Kajila
"""

import logging
import os

import modal

from src.models.registry import MODELS


APP_NAME = "snore-detection"
VOLUME_PATH = "/vol"
LOCAL_RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

app = modal.App(APP_NAME)

# Keeps the dataset and the trained models between runs
volume = modal.Volume.from_name(f"{APP_NAME}-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "numpy>=2.5",
        "scikit-learn>=1.9",
        "librosa>=1.0",
        "torch>=2.11",
        "torchaudio>=2.11",
        "onnx>=1.23",
        "onnxruntime>=1.31",
        "onnxscript>=0.7",
    )
    # src/config.py reads these, so the project code runs unchanged on the volume
    .env({
        "SNORE_DATA_DIR": f"{VOLUME_PATH}/data",
        "SNORE_MODEL_DIR": f"{VOLUME_PATH}/models",
    })
    .add_local_python_source("src")
)


@app.function(image=image, gpu="T4", volumes={VOLUME_PATH: volume}, timeout=60 * 60)
def train(
    model_name: str = "mobile_tiny",
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> dict:
    """Preprocess, train and (for the mobile models) export; return the produced files."""
    import torch

    from src.config import RAW_DATA_DIR
    from src.data.preprocess import preprocess_data
    from src.models.registry import WAVEFORM_MODELS
    from src.train import train_model

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,
    )
    # the ONNX exporter is very verbose at INFO level
    for noisy in ("onnxscript", "onnx_ir"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger = logging.getLogger(APP_NAME)
    if not os.path.isdir(RAW_DATA_DIR):
        raise FileNotFoundError(
            "No audio data on the Modal volume yet; run "
            "`modal run modal_app.py --upload` once to send data/raw."
        )
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("Running on %s.", device)

    preprocess_data()
    paths = [
        train_model(
            epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_name=model_name,
        )
    ]
    if model_name in WAVEFORM_MODELS:
        from src.export import export_model

        paths.append(export_model(model_name))

    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
    files = {}
    for path in paths:
        with open(path, "rb") as file:
            files[os.path.basename(path)] = file.read()
    if not modal.is_local():
        volume.commit()
    return {
        "device": device,
        "best_epoch": checkpoint["best_epoch"],
        "validation_loss": checkpoint["validation_loss"],
        "validation_metrics": checkpoint["validation_metrics"],
        "files": files,
    }


def upload_raw_data() -> None:
    """Send the local data/raw folder to the Modal volume."""
    if not os.path.isdir(LOCAL_RAW_DATA_DIR):
        raise FileNotFoundError(f"Local dataset not found: {LOCAL_RAW_DATA_DIR}")
    print(f"Uploading {LOCAL_RAW_DATA_DIR} to the Modal volume...")
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(LOCAL_RAW_DATA_DIR, "/data/raw")


@app.local_entrypoint()
def main(
    model: str = "mobile_tiny",
    epochs: int = 0,
    batch_size: int = 0,
    learning_rate: float = 0.0,
    upload: bool = False,
    output_dir: str = os.path.join("models", "modal"),
):
    """Train on Modal and download the results. 0 keeps the default of src/train.py."""
    if model not in MODELS:
        raise SystemExit(f"Unknown model '{model}'; choose from {', '.join(MODELS)}.")
    if upload:
        upload_raw_data()

    result = train.remote(
        model_name=model,
        epochs=epochs or None,
        batch_size=batch_size or None,
        learning_rate=learning_rate or None,
    )

    os.makedirs(output_dir, exist_ok=True)
    for name, content in result["files"].items():
        with open(os.path.join(output_dir, name), "wb") as file:
            file.write(content)
    metrics = result["validation_metrics"]
    print(
        f"\n{model} trained on {result['device']}: best epoch {result['best_epoch']}, "
        f"validation loss {result['validation_loss']:.4f}, "
        f"accuracy {metrics['accuracy']:.4f}, F1 {metrics['f1']:.4f}"
    )
    print(f"Saved to {output_dir}: {', '.join(result['files'])}")
