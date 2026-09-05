# Documentation

A short course in deep learning for audio with PyTorch, built around the code of this
repository. It is organised in six parts. Each part is one document, opens with an
introduction and a table of contents, and closes with its key points.

Inside a part, every section follows the same plan: learning goals, theory, where the
idea appears in the project, common pitfalls, exercises and further reading.

The course is meant to be read with the source code open. Numbers quoted from training
runs come from single runs on the project's dataset and are indications, not benchmarks.

## Prerequisites

Python, basic linear algebra (vectors, matrices) and calculus (derivatives, the chain
rule). No prior knowledge of signal processing or PyTorch is assumed.

## Contents

### [Part I - From sound to tensors](01-audio-and-tensors.md)

How a recording becomes the input of a network.

| Section | Topic                                                                         | Main code                  |
|---------|-------------------------------------------------------------------------------|----------------------------|
| 1.1     | [Digital audio](01-audio-and-tensors.md#11-digital-audio)                     | `src/data/preprocess.py`   |
| 1.2     | [Spectrograms](01-audio-and-tensors.md#12-spectrograms)                       | `src/models/mobile_cnn.py` |
| 1.3     | [Mel and log scales](01-audio-and-tensors.md#13-mel-and-log-scales)           | `src/models/mobile_cnn.py` |
| 1.4     | [Tensors and shapes](01-audio-and-tensors.md#14-tensors-and-shapes)           | whole pipeline             |
| 1.5     | [Datasets and data loaders](01-audio-and-tensors.md#15-datasets-and-data-loaders) | `src/data/data_manager.py` |

### [Part II - Convolutional networks](02-convolutional-networks.md)

The layers, then the two families of models of the project.

| Section | Topic                                                                         | Main code                  |
|---------|-------------------------------------------------------------------------------|----------------------------|
| 2.1     | [Convolution](02-convolutional-networks.md#21-convolution)                    | `src/models/audio_cnn.py`  |
| 2.2     | [Normalisation, activation and pooling](02-convolutional-networks.md#22-normalisation-activation-and-pooling) | `src/models/audio_cnn.py` |
| 2.3     | [Walk-through of the audio CNN](02-convolutional-networks.md#23-walk-through-of-the-audio-cnn) | `src/models/audio_cnn.py` |
| 2.4     | [Depthwise separable convolutions](02-convolutional-networks.md#24-depthwise-separable-convolutions) | `src/models/mobile_cnn.py` |
| 2.5     | [Residual and dense blocks](02-convolutional-networks.md#25-residual-and-dense-blocks) | `src/models/cnn_blocks.py` |

### [Part III - Training](03-training.md)

What is minimised, how, and how to keep the model from memorising.

| Section | Topic                                                                         | Main code                  |
|---------|-------------------------------------------------------------------------------|----------------------------|
| 3.1     | [Loss functions](03-training.md#31-loss-functions)                            | `src/train.py`             |
| 3.2     | [Autograd and backpropagation](03-training.md#32-autograd-and-backpropagation)| `src/train.py`             |
| 3.3     | [Optimisers](03-training.md#33-optimisers)                                    | `src/train.py`             |
| 3.4     | [Learning rate schedules and early stopping](03-training.md#34-learning-rate-schedules-and-early-stopping) | `src/train.py` |
| 3.5     | [Regularisation and augmentation](03-training.md#35-regularisation-and-augmentation) | `src/data/data_manager.py` |
| 3.6     | [Anatomy of the training loop](03-training.md#36-anatomy-of-the-training-loop)| `src/train.py`             |

### [Part IV - Evaluation and scientific method](04-evaluation.md)

What a score means and how far to trust it.

| Section | Topic                                                                         | Main code                  |
|---------|-------------------------------------------------------------------------------|----------------------------|
| 4.1     | [Evaluation metrics](04-evaluation.md#41-evaluation-metrics)                  | `src/train.py`             |
| 4.2     | [Generalisation and data leakage](04-evaluation.md#42-generalisation-and-data-leakage) | `src/data/preprocess.py` |
| 4.3     | [Reproducibility](04-evaluation.md#43-reproducibility)                        | `src/train.py`, `src/config.py` |

### [Part V - Deployment](05-deployment.md)

From a PyTorch model to a file that runs on a phone.

| Section | Topic                                                                         | Main code                  |
|---------|-------------------------------------------------------------------------------|----------------------------|
| 5.1     | [Exporting to ONNX](05-deployment.md#51-exporting-to-onnx)                    | `src/export.py`            |
| 5.2     | [On-device inference](05-deployment.md#52-on-device-inference)                | `src/inference.py`         |

### [Part VI - Practice](06-practice.md)

What to do when it does not work, and what to try next.

| Section | Topic                                                                         |
|---------|-------------------------------------------------------------------------------|
| 6.1     | [Debugging neural networks](06-practice.md#61-debugging-neural-networks)      |
| 6.2     | [Experiments to try](06-practice.md#62-experiments-to-try)                    |

### Reference

- [Glossary](glossary.md): every term, with the section that explains it.
- [References](references.md): textbooks, papers grouped by part, software
  documentation.

## Suggested paths

- **New to deep learning**: read the parts in order, and do at least one exercise per
  section.
- **Know the theory, new to PyTorch**: sections 1.4, 1.5, 3.2 and 3.6, then 6.1.
- **Interested in the mobile models**: sections 1.2, 1.3 and 2.4, then part V.
- **Want to improve the project**: part IV, then section 6.2.

## How to study

1. Read a section with the matching source file open.
2. Before running an exercise, write down what you expect. The gap between the
   prediction and the result is where the learning happens.
3. At the end of a part, reread its key points and check that you could explain each
   one without the text.
4. Keep a log of your experiments, as described in section 6.2.

## Conventions

- Mathematics is written in LaTeX and rendered by GitHub.
- Shapes are written `(batch, channels, height, width)`; for spectrograms, height is
  frequency and width is time.
- Commands are run from the project root, for example `python -m src.train`.
