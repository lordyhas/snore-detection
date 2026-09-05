# Part II - Convolutional networks

This part explains the building blocks of the two families of models in this
repository, then reads the models themselves. It starts with the convolution and the
layers that always accompany it, walks through `AudioCNN` parameter by parameter,
and shows how the mobile models reach a similar accuracy with a hundred times fewer
parameters. The last section covers skip connections, which the current models do
not need but deeper ones would.

**Contents**

- [2.1 Convolution](#21-convolution)
- [2.2 Normalisation, activation and pooling](#22-normalisation-activation-and-pooling)
- [2.3 Walk-through of the audio CNN](#23-walk-through-of-the-audio-cnn)
- [2.4 Depthwise separable convolutions](#24-depthwise-separable-convolutions)
- [2.5 Residual and dense blocks](#25-residual-and-dense-blocks)
- [Key points](#key-points)

## 2.1 Convolution

### Learning goals

- Define the 2-D convolution used in neural networks and its hyperparameters.
- Compute output sizes, parameter counts and receptive fields by hand.
- Explain why convolutions suit spectrograms, and where the analogy with images breaks.

### Theory

#### The operation

A convolution layer slides a small **kernel** over the input and computes a weighted sum
at each position. For an input with $C_{in}$ channels and a $k \times k$ kernel, output
channel $o$ is

$$y_o[i, j] = b_o + \sum_{c=1}^{C_{in}} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1}
  W_{o,c}[u, v] \; x_c[i + u, \, j + v].$$

Two ideas make it far more economical than a fully connected layer:

- **local connectivity**: each output looks at a small neighbourhood;
- **weight sharing**: the same kernel is used at every position, so a pattern learned
  in one place is recognised everywhere (*translation equivariance*).

#### Hyperparameters and output size

With padding $p$ and stride $s$, an input of size $n$ gives

$$n_{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1 .$$

`kernel_size=3, padding=1, stride=1` keeps the size unchanged; `stride=2` halves it.

#### Parameter count

$$\text{parameters} = k^2 \, C_{in} \, C_{out} \; (+ \, C_{out} \text{ biases}).$$

The count does not depend on the size of the image: this is why a convolutional network
accepts inputs of any size.

#### Receptive field

The receptive field of a unit is the region of the input that can influence it. It grows
with depth. Tracking the *jump* $j$ (distance in input pixels between two neighbouring
units), each layer updates

$$r \leftarrow r + (k - 1) \, j, \qquad j \leftarrow j \cdot s .$$

### In this project

`ConvBlock` in `src/models/audio_cnn.py` stacks two 3×3 convolutions and a 2×2 max
pooling. The convolutions use `bias=False`, because the batch normalisation that follows
has its own bias and would cancel any other.

Parameter count of the first block (1 → 32 channels):

| Layer                 | Computation      | Parameters |
|-----------------------|------------------|------------|
| Conv 3×3, 1 → 32      | $9 \times 1 \times 32$  | 288   |
| BatchNorm (32)        | $2 \times 32$    | 64         |
| Conv 3×3, 32 → 32     | $9 \times 32 \times 32$ | 9 216 |
| BatchNorm (32)        | $2 \times 32$    | 64         |
| **Total**             |                  | **9 632**  |

Receptive field through the four blocks (start with $r = 1$, $j = 1$):

| After block | $r$ | $j$ |
|-------------|-----|-----|
| 1           | 6   | 2   |
| 2           | 16  | 4   |
| 3           | 36  | 8   |
| 4           | 76  | 16  |

A unit of the last feature map sees a 76 × 76 region. The input is 128 mel bands by 32
frames: every unit sees the whole second of audio and more than half of the frequency
axis.

### Convolutions on spectrograms

A spectrogram is not a photograph. Shifting a pattern along the **time** axis leaves its
meaning unchanged: a snore is a snore whenever it starts. Shifting it along the
**frequency** axis changes the sound: a pattern at 100 Hz and the same pattern at 4 kHz
are different events. Convolutions treat both axes the same way. They still work well in
practice, partly because the deeper layers and the final pooling can learn where things
are, but this is a modelling approximation worth keeping in mind.

### Pitfalls

- Forgetting `padding=1` with a 3×3 kernel shrinks the map by 2 at every layer.
  `BottleneckBlock` in `src/models/cnn_blocks.py` does exactly that in its middle
  convolution; check whether you want it before using the block.
- An odd size with stride 2 is rounded: 101 frames become 51, then 26, then 13.

### Exercises

1. Compute the parameter count of the second `ConvBlock` (32 → 64). Check it with
   `sum(p.numel() for p in block.parameters())`.
2. Apply the output-size formula to the mobile model: start from 40 × 101 and apply
   three convolutions with `k=3, p=1, s=2`.
3. Reproduce the receptive-field table. What would it be with 5×5 kernels?

### Further reading

- Goodfellow, Bengio and Courville, *Deep Learning*, chapter 9.
- Dumoulin and Visin, "A guide to convolution arithmetic for deep learning" (2016).

## 2.2 Normalisation, activation and pooling

### Learning goals

- Explain why a network needs non-linear activations.
- Describe batch normalisation and its two modes.
- Distinguish max pooling, average pooling and global pooling.

### Activation functions

A stack of convolutions without anything in between is still a single linear map: the
composition of linear functions is linear. Non-linear **activations** are what let depth
add expressive power. The standard choice is the rectified linear unit:

$$\text{ReLU}(x) = \max(0, x).$$

It is cheap, and its gradient is exactly 1 for positive inputs, which avoids the
*vanishing gradient* of sigmoid and tanh in deep networks (their derivative is at most
0.25 and 1 respectively, and tends to 0 when they saturate).

`nn.ReLU(inplace=True)` overwrites its input to save memory. It is safe after a batch
normalisation, as in this project.

### Batch normalisation

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

### Pooling

**Max pooling** keeps the largest value of each window. `nn.MaxPool2d(2)` halves both
dimensions and gives a small invariance to shifts.

**Global average pooling** averages each channel over all positions:
`nn.AdaptiveAvgPool2d(1)` turns `(batch, 256, H, W)` into `(batch, 256, 1, 1)` for any
`H` and `W`. Compared with flattening the feature map, it has no parameters, so it cannot
overfit, and it makes the network independent of the input size.

### In this project

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

### Pitfalls

- With very small batches, the batch statistics are noisy and batch normalisation
  degrades. Group normalisation or layer normalisation are the usual alternatives.
- When exporting or measuring a model, always call `.eval()` first. A training-mode
  export captures batch statistics and random masks (see
  [section 5.1](05-deployment.md#51-exporting-to-onnx)).

### Exercises

1. Show that two linear layers without activation, $W_2 (W_1 x)$, are equivalent to one.
2. Feed the same clip to a trained model alone and inside a batch of 32, first in
   `eval()` mode and then in `train()` mode. Compare the outputs.
3. List the keys of `model.state_dict()` for one `BatchNorm2d` layer and identify which
   ones are parameters and which ones are buffers.

### Further reading

- Ioffe and Szegedy (2015), see [References](references.md).

## 2.3 Walk-through of the audio CNN

### Learning goals

- Read an `nn.Module` and predict the shape of every intermediate tensor.
- Account for every parameter of a real model.
- Understand the design pattern "feature extractor, pooling, classifier".

### How a module is written

A PyTorch model is a class deriving from `nn.Module` with two methods:

- `__init__` creates the layers. Assigning a module to `self` registers its parameters;
- `forward` describes the computation. Calling `model(x)` runs `forward` (plus hooks),
  so never call `forward` directly.

`nn.Sequential` chains modules whose output feeds the next one.

### The architecture

`AudioCNN` in `src/models/audio_cnn.py` follows the VGG recipe: stack simple blocks,
halve the resolution and double the channels at each stage. Spatial detail is traded for
semantic detail.

Shapes for a 1-second clip (128 mel bands, 32 frames):

| Stage                     | Output shape       | Parameters |
|---------------------------|--------------------|------------|
| Input                     | `(1, 128, 32)`     |            |
| `ConvBlock(1, 32)`        | `(32, 64, 16)`     | 9 632      |
| `ConvBlock(32, 64)`       | `(64, 32, 8)`      | 55 552     |
| `ConvBlock(64, 128)`      | `(128, 16, 4)`     | 221 696    |
| `ConvBlock(128, 256)`     | `(256, 8, 2)`      | 885 760    |
| `AdaptiveAvgPool2d(1)`    | `(256, 1, 1)`      | 0          |
| `Flatten` and `Dropout`   | `(256,)`           | 0          |
| `Linear(256, 64)`, ReLU   | `(64,)`            | 16 448     |
| `Dropout`, `Linear(64, 1)`| `(1,)`             | 65         |
| **Total**                 |                    | **1 189 153** |

Three observations.

**Where the parameters are.** The last block holds 74 % of them. The cost of a
convolution grows with the *product* of its input and output channels, so the wide, late
layers dominate. This is the starting point of
[section 2.4](#24-depthwise-separable-convolutions).

**Where the computation is.** The opposite holds for the number of operations: early
layers work on large maps. The first block applies its kernels at 128 × 32 positions,
the last one at 16 × 4.

**One output.** The model returns a single number, a *logit*, not a probability.
The sigmoid is applied by the loss during training
([section 3.1](03-training.md#31-loss-functions)) and by the caller at inference.

### The classifier head

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(dropout),
    nn.Linear(256, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout),
    nn.Linear(64, num_outputs),
)
```

After global pooling, each clip is summarised by 256 numbers. A small two-layer
perceptron maps them to the decision. Dropout is placed here, on the dense layers, where
it is most useful ([section 3.5](03-training.md#35-regularisation-and-augmentation)).

### Is the model the right size?

There are 800 training clips and 1.19 million parameters: about 1500 parameters per
example. Classical statistics would predict heavy overfitting. In practice the
validation accuracy reaches 0.99, thanks to weight sharing, batch normalisation, dropout,
augmentation and early stopping. But `mobile_tiny`, with 9 889 parameters, reaches 0.98
on the same split. On this dataset the large model is far from necessary, which is a
hint that the task, as posed by this dataset, is easy
([section 4.2](04-evaluation.md#42-generalisation-and-data-leakage)).

### Exercises

1. Run the following and compare with the table:

   ```python
   from src.models.audio_cnn import AudioCNN
   model = AudioCNN()
   for name, module in model.features.named_children():
       print(name, sum(p.numel() for p in module.parameters()))
   ```

2. Pass `torch.zeros(1, 1, 128, 32)` through `model.features` and print the shape after
   each block.
3. The model accepts a 3-second clip (94 frames) without modification. Which layer makes
   this possible?
4. Replace the two-layer head by a single `Linear(256, 1)`. How many parameters are
   saved? Does the validation accuracy change?

### Further reading

- Simonyan and Zisserman, "Very Deep Convolutional Networks for Large-Scale Image
  Recognition" (2015).

## 2.4 Depthwise separable convolutions

### Learning goals

- Factorise a convolution into a depthwise and a pointwise step.
- Quantify the savings in parameters.
- Read the mobile models of this project.

### Theory

A standard 3×3 convolution does two jobs at once: it looks at the **spatial**
neighbourhood and it **mixes channels**. A depthwise separable convolution (popularised
by MobileNets, Howard et al., 2017) does them one after the other.

1. **Depthwise convolution**: one 3×3 kernel *per input channel*, with no mixing between
   channels. In PyTorch this is `groups=in_channels`.
2. **Pointwise convolution**: a 1×1 convolution that mixes the channels at each position.

Parameter counts, without biases:

$$\underbrace{k^2 \, C_{in} \, C_{out}}_{\text{standard}}
\qquad \text{versus} \qquad
\underbrace{k^2 \, C_{in}}_{\text{depthwise}} + \underbrace{C_{in} \, C_{out}}_{\text{pointwise}}$$

The ratio is

$$\frac{k^2 C_{in} + C_{in} C_{out}}{k^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{k^2} .$$

With $k = 3$ and many channels, the cost approaches one ninth of the standard
convolution. For 64 → 64 channels: 36 864 parameters become 576 + 4 096 = 4 672, a
factor 7.9. The number of multiplications is reduced in the same proportion.

The factorisation is a restriction: not every standard convolution can be written this
way. Experience shows that the loss in accuracy is small compared with the savings.

### In this project

`DepthwiseSeparableBlock` in `src/models/mobile_cnn.py`:

```python
nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
          groups=in_channels, bias=False),          # depthwise
nn.BatchNorm2d(in_channels),
nn.ReLU(inplace=True),
nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),   # pointwise
nn.BatchNorm2d(out_channels),
nn.ReLU(inplace=True),
```

`SnoreMobileNet` starts with one ordinary convolution (with a single input channel there
is nothing to factorise), followed by four separable blocks. The `width` argument scales
every layer: 16 for `mobile_tiny`, 32 for `mobile_small`.

Shapes and parameters of `mobile_tiny` (`width=16`):

| Stage                          | Output shape     | Parameters |
|--------------------------------|------------------|------------|
| `LogMelFrontend`               | `(1, 40, 101)`   | 0          |
| Conv 3×3, stride 2, BN         | `(16, 20, 51)`   | 176        |
| Separable 16 → 32, stride 2    | `(32, 10, 26)`   | 752        |
| Separable 32 → 32              | `(32, 10, 26)`   | 1 440      |
| Separable 32 → 64, stride 2    | `(64, 5, 13)`    | 2 528      |
| Separable 64 → 64              | `(64, 5, 13)`    | 4 928      |
| Global pooling, `Linear(64, 1)`| `(1,)`           | 65         |
| **Total**                      |                  | **9 889**  |

The front-end has no *learned* parameter: the Hann window and the mel filter bank are
fixed and stored as buffers.

#### Comparison

| Model          | Parameters | Validation accuracy |
|----------------|------------|---------------------|
| `audio_cnn`    | 1 189 153  | 0.99                |
| `mobile_small` | 35 137     | 0.975               |
| `mobile_tiny`  | 9 889      | 0.98                |

The difference between 0.975 and 0.99 on 200 validation clips is three clips: it is
within the noise of such a small set, and the three models should be considered
equivalent on this split. The smallest one is not worse; do not conclude that it is
better either.

### Pitfalls

- `groups=in_channels` requires the output channels of that layer to be a multiple of
  the input channels; here they are equal.
- Depthwise convolutions are cheap in parameters but not always fast: their speed
  depends on how well the runtime optimises them. Measure on the target device.

### Exercises

1. Check the "Parameters" column for the block 32 → 64 by hand: depthwise, batch
   normalisation, pointwise, batch normalisation.
2. Compute the saving ratio for 128 → 256 channels with the formula, then by counting.
3. Build `SnoreMobileNet(width=8)`. How many parameters? Train it with a new entry in
   `src/models/registry.py` and see how small the model can get.
4. How many parameters would `mobile_tiny` have if its four separable blocks were
   standard 3×3 convolutions?

### Further reading

- Howard et al. (2017); Zhang et al. (2017), "Hello Edge". See
  [References](references.md).

## 2.5 Residual and dense blocks

### Learning goals

- Explain the degradation problem of deep networks and how skip connections address it.
- Compare residual, dense and bottleneck designs.
- Read `src/models/cnn_blocks.py`, which is not used by the current models and is
  available for experiments.

### The degradation problem

Adding layers to a plain network should never hurt: the extra layers could learn the
identity. Yet in practice, beyond a certain depth, even the *training* error goes up.
The cause is optimisation, not overfitting: gradients are multiplied by one Jacobian per
layer on the way back, and the signal degrades.

### Residual blocks

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

### Dense blocks

DenseNet (Huang et al., 2017) replaces the addition by a **concatenation**:

```python
output = torch.cat([output, x], 1)   # along the channel dimension
```

Each layer receives the feature maps of all the previous ones. Features are reused
instead of recomputed, and the gradient again has short paths. The price is that the
number of channels grows with every layer: `DenseBlock` in this project returns
`2 * in_channels` channels.

### Bottleneck blocks

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

### Summary

| Block      | Combination     | Output channels       | Main benefit                |
|------------|-----------------|-----------------------|-----------------------------|
| Residual   | addition        | `out_channels`        | very deep networks          |
| Dense      | concatenation   | `2 * in_channels`     | feature reuse               |
| Bottleneck | none here       | `in_channels`         | fewer parameters per layer  |

### Would they help here?

Skip connections pay off in deep networks. `AudioCNN` has 8 convolution layers and
trains without difficulty, so there is no problem for them to solve yet. They become
relevant if you make the model much deeper, for example on a larger and harder dataset.

### Exercises

1. Pass `torch.zeros(1, 32, 16, 16)` through `ResidualBlock(32, 64, stride=2)`,
   `DenseBlock(32)` and `BottleneckBlock(32)`. Predict the three output shapes first.
2. Write `ResidualAudioCNN`, a copy of `AudioCNN` in which each `ConvBlock` is replaced
   by a `ResidualBlock` with `stride=2`. Register it in `src/models/registry.py` and
   train it.
3. Add `padding=1` and a skip connection to `BottleneckBlock` so that it matches the
   original ResNet design.
4. Why does `ResidualBlock.forward` use `output += self.shortcut(x)` *before* the last
   ReLU and not after?

### Further reading

- He et al. (2016); Huang et al. (2017). See [References](references.md).
