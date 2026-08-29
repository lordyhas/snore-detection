# 7. Normalisation, activation and pooling

## Learning goals

- Explain why a network needs non-linear activations.
- Describe batch normalisation and its two modes.
- Distinguish max pooling, average pooling and global pooling.

## Activation functions

A stack of convolutions without anything in between is still a single linear map: the
composition of linear functions is linear. Non-linear **activations** are what let depth
add expressive power. The standard choice is the rectified linear unit:

$$\text{ReLU}(x) = \max(0, x).$$

It is cheap, and its gradient is exactly 1 for positive inputs, which avoids the
*vanishing gradient* of sigmoid and tanh in deep networks (their derivative is at most
0.25 and 1 respectively, and tends to 0 when they saturate).

`nn.ReLU(inplace=True)` overwrites its input to save memory. It is safe after a batch
normalisation, as in this project.

## Batch normalisation

For each channel, batch normalisation (Ioffe and Szegedy, 2015) standardises the
activations and then applies a learned scale $\gamma$ and shift $\beta$:

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad
  y = \gamma \, \hat{x} + \beta .$$

The crucial point is where $\mu$ and $\sigma^2$ come from:

| Mode             | Statistics used                        | Set by          |
|------------------|----------------------------------------|-----------------|
| Training         | mean and variance of the current batch | `model.train()` |
| Evaluation       | running averages stored in the layer   | `model.eval()`  |

In training mode the output for one example depends on the other examples of the batch.
In evaluation mode the layer is a fixed affine function. Forgetting `model.eval()` is one
of the most common bugs in PyTorch: predictions then change with the batch composition,
and fail for a batch of one.

Batch normalisation allows higher learning rates, reduces the sensitivity to
initialisation and has a mild regularising effect. Each layer has $2C$ learned
parameters ($\gamma$, $\beta$) and $2C$ non-learned running statistics, which are saved
in the `state_dict` as *buffers*.

## Pooling

**Max pooling** keeps the largest value of each window. `nn.MaxPool2d(2)` halves both
dimensions and gives a small invariance to shifts.

**Global average pooling** averages each channel over all positions:
`nn.AdaptiveAvgPool2d(1)` turns `(batch, 256, H, W)` into `(batch, 256, 1, 1)` for any
`H` and `W`. Compared with flattening the feature map, it has no parameters, so it cannot
overfit, and it makes the network independent of the input size.

## In this project

The pattern of `ConvBlock` (`src/models/audio_cnn.py`) is the classic one:

```python
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
nn.BatchNorm2d(out_channels),
nn.ReLU(inplace=True),
```

In `src/train.py`, a single line switches the mode for the whole network:

```python
model.train(is_training)   # True for the training pass, False for validation
```

The mobile models reduce the resolution with strided convolutions instead of max
pooling: one layer does the work of two, which matters when every operation counts.

## Pitfalls

- With very small batches, the batch statistics are noisy and batch normalisation
  degrades. Group normalisation or layer normalisation are the usual alternatives.
- When exporting or measuring a model, always call `.eval()` first. A training-mode
  export captures batch statistics and random masks (see
  [chapter 20](20-exporting-to-onnx.md)).

## Exercises

1. Show that two linear layers without activation, $W_2 (W_1 x)$, are equivalent to one.
2. Feed the same clip to a trained model alone and inside a batch of 32, first in
   `eval()` mode and then in `train()` mode. Compare the outputs.
3. List the keys of `model.state_dict()` for one `BatchNorm2d` layer and identify which
   ones are parameters and which ones are buffers.

## Further reading

- Ioffe and Szegedy (2015), see [References](25-references.md).
- Next: [8. Walk-through of the audio CNN](08-audio-cnn-walkthrough.md).
