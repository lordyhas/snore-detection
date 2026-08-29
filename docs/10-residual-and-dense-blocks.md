# 10. Residual and dense blocks

## Learning goals

- Explain the degradation problem of deep networks and how skip connections address it.
- Compare residual, dense and bottleneck designs.
- Read `src/models/cnn_blocks.py`, which is not used by the current models and is
  available for experiments.

## The degradation problem

Adding layers to a plain network should never hurt: the extra layers could learn the
identity. Yet in practice, beyond a certain depth, even the *training* error goes up.
The cause is optimisation, not overfitting: gradients are multiplied by one Jacobian per
layer on the way back, and the signal degrades.

## Residual blocks

He et al. (2016) proposed to let each block learn a *correction* to its input instead of
a full transformation:

$$y = \text{ReLU}\big(F(x) + x\big).$$

If the best thing to do is nothing, the block only has to drive $F$ to zero. And the
gradient reaches earlier layers directly through the addition:

$$\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I .$$

The identity term $I$ gives the gradient a path that no weight can shrink. Networks of
hundreds of layers became trainable.

The addition requires $F(x)$ and $x$ to have the same shape. When the block changes the
number of channels or the stride, the shortcut uses a 1×1 convolution to adapt:

```python
self.shortcut = nn.Sequential()
if stride != 1 or in_channels != out_channels:
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
    )
```

An empty `nn.Sequential()` is the identity.

## Dense blocks

DenseNet (Huang et al., 2017) replaces the addition by a **concatenation**:

```python
output = torch.cat([output, x], 1)   # along the channel dimension
```

Each layer receives the feature maps of all the previous ones. Features are reused
instead of recomputed, and the gradient again has short paths. The price is that the
number of channels grows with every layer: `DenseBlock` in this project returns
`2 * in_channels` channels.

## Bottleneck blocks

A 3×3 convolution on many channels is expensive. The bottleneck design surrounds it with
two 1×1 convolutions: the first reduces the number of channels, the 3×3 convolution
works on the reduced representation, the second restores the width.

For 256 channels and a reduction factor 4:

| Design                    | Parameters                                   |
|---------------------------|----------------------------------------------|
| One 3×3 convolution       | $9 \times 256 \times 256 = 589\,824$         |
| 1×1, 3×3, 1×1 bottleneck  | $16\,384 + 36\,864 + 16\,384 = 69\,632$      |

`BottleneckBlock` in this project implements the three convolutions. Two things differ
from the ResNet bottleneck and should be known before using it: the 3×3 convolution has
no padding, so the height and width shrink by 2, and the block has neither skip
connection nor batch normalisation.

## Summary

| Block      | Combination     | Output channels       | Main benefit                |
|------------|-----------------|-----------------------|-----------------------------|
| Residual   | addition        | `out_channels`        | very deep networks          |
| Dense      | concatenation   | `2 * in_channels`     | feature reuse               |
| Bottleneck | none here       | `in_channels`         | fewer parameters per layer  |

## Would they help here?

Skip connections pay off in deep networks. `AudioCNN` has 8 convolution layers and
trains without difficulty, so there is no problem for them to solve yet. They become
relevant if you make the model much deeper, for example on a larger and harder dataset.

## Exercises

1. Pass `torch.zeros(1, 32, 16, 16)` through `ResidualBlock(32, 64, stride=2)`,
   `DenseBlock(32)` and `BottleneckBlock(32)`. Predict the three output shapes first.
2. Write `ResidualAudioCNN`, a copy of `AudioCNN` in which each `ConvBlock` is replaced
   by a `ResidualBlock` with `stride=2`. Register it in `src/models/registry.py` and
   train it.
3. Add `padding=1` and a skip connection to `BottleneckBlock` so that it matches the
   original ResNet design.
4. Why does `ResidualBlock.forward` use `output += self.shortcut(x)` *before* the last
   ReLU and not after?

## Further reading

- He et al. (2016); Huang et al. (2017). See [References](25-references.md).
- Next: [11. Loss functions](11-loss-functions.md).
