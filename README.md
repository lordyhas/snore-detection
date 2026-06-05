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

-----------------------------------------------------------------------------------

By Hassan K (lordyhas)
