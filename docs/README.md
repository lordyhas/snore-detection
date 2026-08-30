# Documentation

A short course in deep learning for audio with PyTorch, built around the code of this
repository. Each chapter introduces the theory, shows where it appears in the project,
lists common pitfalls and ends with exercises.

The chapters are meant to be read with the source code open. Numbers quoted from
training runs come from single runs on the project's dataset and are indications, not
benchmarks.

## Prerequisites

Python, basic linear algebra (vectors, matrices) and calculus (derivatives, the chain
rule). No prior knowledge of signal processing or PyTorch is assumed.

## Contents

### Part I - From sound to tensors

| #  | Chapter                                                    | Main code                    |
|----|------------------------------------------------------------|------------------------------|
| 1  | [Digital audio](01-digital-audio.md)                       | `src/data/preprocess.py`     |
| 2  | [Spectrograms](02-spectrograms.md)                         | `src/models/mobile_cnn.py`   |
| 3  | [Mel and log scales](03-mel-and-log-scales.md)             | `src/models/mobile_cnn.py`   |
| 4  | [Tensors and shapes](04-tensors-and-shapes.md)             | whole pipeline               |
| 5  | [Datasets and data loaders](05-datasets-and-dataloaders.md)| `src/data/data_manager.py`   |

### Part II - Convolutional networks

| #  | Chapter                                                    | Main code                    |
|----|------------------------------------------------------------|------------------------------|
| 6  | [Convolution](06-convolution.md)                           | `src/models/audio_cnn.py`    |
| 7  | [Normalisation, activation and pooling](07-normalization-activation-pooling.md) | `src/models/audio_cnn.py` |
| 8  | [Walk-through of the audio CNN](08-audio-cnn-walkthrough.md) | `src/models/audio_cnn.py`  |
| 9  | [Depthwise separable convolutions](09-depthwise-separable-convolutions.md) | `src/models/mobile_cnn.py` |
| 10 | [Residual and dense blocks](10-residual-and-dense-blocks.md) | `src/models/cnn_blocks.py` |

### Part III - Training

| #  | Chapter                                                    | Main code                    |
|----|------------------------------------------------------------|------------------------------|
| 11 | [Loss functions](11-loss-functions.md)                     | `src/train.py`               |
| 12 | [Autograd and backpropagation](12-autograd-and-backpropagation.md) | `src/train.py`       |
| 13 | [Optimisers](13-optimizers.md)                             | `src/train.py`               |
| 14 | [Learning rate schedules and early stopping](14-learning-rate-and-early-stopping.md) | `src/train.py` |
| 15 | [Regularisation and augmentation](15-regularization-and-augmentation.md) | `src/data/data_manager.py` |
| 16 | [Anatomy of the training loop](16-training-loop.md)        | `src/train.py`               |

### Part IV - Evaluation and scientific method

| #  | Chapter                                                    | Main code                    |
|----|------------------------------------------------------------|------------------------------|
| 17 | [Evaluation metrics](17-evaluation-metrics.md)             | `src/train.py`               |
| 18 | [Generalisation and data leakage](18-generalization-and-data-leakage.md) | `src/data/preprocess.py` |
| 19 | [Reproducibility](19-reproducibility.md)                   | `src/train.py`, `src/config.py` |

### Part V - Deployment

| #  | Chapter                                                    | Main code                    |
|----|------------------------------------------------------------|------------------------------|
| 20 | [Exporting to ONNX](20-exporting-to-onnx.md)               | `src/export.py`              |
| 21 | [On-device inference](21-on-device-inference.md)           | `src/inference.py`           |

### Part VI - Going further

| #  | Chapter                                                    |
|----|------------------------------------------------------------|
| 22 | [Debugging neural networks](22-debugging-neural-networks.md) |
| 23 | [Experiments to try](23-experiments.md)                    |
| 24 | [Glossary](24-glossary.md)                                 |
| 25 | [References](25-references.md)                             |

## Suggested paths

- **New to deep learning**: read in order, and do at least one exercise per chapter.
- **Know the theory, new to PyTorch**: chapters 4, 5, 12 and 16, then 22.
- **Interested in the mobile models**: chapters 2, 3, 9, 20 and 21.
- **Want to improve the project**: chapters 17 and 18, then 23.

## Conventions

- Mathematics is written in LaTeX and rendered by GitHub.
- Shapes are written `(batch, channels, height, width)`; for spectrograms, height is
  frequency and width is time.
- Commands are run from the project root, for example `python -m src.train`.
