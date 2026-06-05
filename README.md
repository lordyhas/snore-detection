# Snore Detection

This project detects snoring sounds in audio recordings with a convolutional neural
network. One-second clips labelled snoring / non-snoring are converted to mel
spectrograms and classified by a small CNN written in PyTorch.

## Project structure

```
├── data/                    # Not versioned, see "Dataset"
│   ├── raw/
│   │   ├── 0/               # Non-snoring audio files
│   │   └── 1/               # Snoring audio files
│   ├── processed/           # dataset.npz (created by the preprocessing step)
│   └── inference/           # Extra recordings to try the model on
├── models/                  # Saved checkpoints (created by training, not versioned)
├── docs/
├── notebooks/               # Data visualisation and scratch notebooks
├── src/
│   ├── data/
│   │   ├── preprocess.py    # Loading, fixed-length clips, train/validation split
│   │   └── data_manager.py  # PyTorch dataset: mel spectrogram + SpecAugment
│   ├── features/
│   │   └── utils.py         # librosa helpers used by the notebooks
│   ├── models/
│   │   ├── audio_cnn.py     # The classifier
│   │   └── cnn_blocks.py    # Residual, dense and bottleneck blocks
│   ├── config.py            # Paths and random seed
│   ├── train.py             # Training loop
│   ├── main.py              # Preprocess + train in one command
│   └── inference.py         # To be implemented
├── README.md
└── requirements.txt
```

## Dataset

The dataset consists of two folders:

- `data/raw/1/`: 500 snoring audio files, each 1 second long.
- `data/raw/0/`: 500 non-snoring audio files, each 1 second long. These include
  background sounds like a baby crying, a clock ticking, a door opening, etc.

The audio files are not stored in this repository. Download them from Kaggle
([Dataset Link](https://www.kaggle.com/code/mpwolke/snoring-wav/input)) and place the
`0/` and `1/` folders in `data/raw/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the commands from the project root.

```bash
# Preprocess the raw audio, then train
python -m src.main

# Or run the two steps separately
python -m src.data.preprocess
python -m src.train --epochs 50 --batch-size 32
```

The main options are `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`,
`--early-stopping-patience`, `--sample-rate`, `--duration`, `--n-mels`, `--n-fft` and
`--hop-length`. The best checkpoint is saved to `models/audio_cnn.pth`.

## Project workflow

1. **Preprocessing** - each WAV file is resampled to 16 kHz mono, cropped or padded to
   1 second, and the dataset is split 80/20 (stratified) into training and validation
   sets saved in `data/processed/dataset.npz`.
2. **Features** - clips are converted on the fly to 128-band mel spectrograms in dB.
   Frequency and time masking (SpecAugment) are applied to the training set only.
3. **Model** - four convolution blocks (32, 64, 128, 256 channels), global average
   pooling and a small classifier producing one logit (about 1.2 M parameters).
4. **Training** - AdamW with binary cross-entropy, learning rate reduced on plateau,
   early stopping on the validation loss. The weights of the best epoch are saved.

## Results

With the default settings, training stops after 24 epochs and keeps epoch 16:

| Metric (validation, 200 clips) | Value |
|--------------------------------|-------|
| Loss                           | 0.023 |
| Accuracy                       | 0.99  |
| Precision                      | 1.00  |
| Recall                         | 0.98  |
| F1                             | 0.99  |

These numbers come from the validation split, which is also used for early stopping;
there is no separate test set yet. The one-second clips may be cut from the same
recordings, so performance on new recordings is expected to be lower.

---

By Hassan Kajila (lordyhas)
