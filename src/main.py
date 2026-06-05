"""
Snore Detection - command line entry point (preprocess + train).

Author: Hassan Kajila
"""

import argparse
import logging

from src.data.preprocess import preprocess_data
from src.train import train_model


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess the audio dataset and train the snore classifier."
    )
    parser.add_argument("--sample-rate", "--sr", dest="sample_rate", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--validation-split", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--n-mels", type=int)
    parser.add_argument("--n-fft", type=int)
    parser.add_argument("--hop-length", type=int)
    return parser


def main(argv: list[str] | None = None) -> str:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    args = build_parser().parse_args(argv)

    logger.info("Starting audio preprocessing.")
    dataset_path = preprocess_data(
        sample_rate=args.sample_rate,
        duration=args.duration,
        validation_split=args.validation_split,
    )
    logger.info("Preprocessed dataset saved to %s.", dataset_path)

    logger.info("Starting model training.")
    model_path = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    logger.info("Training completed; model saved to %s.", model_path)
    return model_path


if __name__ == "__main__":
    main()
