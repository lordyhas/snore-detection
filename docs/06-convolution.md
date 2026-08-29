# 6. Convolution

## Learning goals

- Define the 2-D convolution used in neural networks and its hyperparameters.
- Compute output sizes, parameter counts and receptive fields by hand.
- Explain why convolutions suit spectrograms, and where the analogy with images breaks.

## Theory

### The operation

A convolution layer slides a small **kernel** over the input and computes a weighted sum
at each position. For an input with $C_{in}$ channels and a $k \times k$ kernel, output
channel $o$ is

$$y_o[i, j] = b_o + \sum_{c=1}^{C_{in}} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1}
  W_{o,c}[u, v] \; x_c[i + u, \, j + v].$$

Two ideas make it far more economical than a fully connected layer:

- **local connectivity**: each output looks at a small neighbourhood;
- **weight sharing**: the same kernel is used at every position, so a pattern learned
  in one place is recognised everywhere (*translation equivariance*).

### Hyperparameters and output size

With padding $p$ and stride $s$, an input of size $n$ gives

$$n_{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1 .$$

`kernel_size=3, padding=1, stride=1` keeps the size unchanged; `stride=2` halves it.

### Parameter count

$$\text{parameters} = k^2 \, C_{in} \, C_{out} \; (+ \, C_{out} \text{ biases}).$$

The count does not depend on the size of the image: this is why a convolutional network
accepts inputs of any size.

### Receptive field

The receptive field of a unit is the region of the input that can influence it. It grows
with depth. Tracking the *jump* $j$ (distance in input pixels between two neighbouring
units), each layer updates

$$r \leftarrow r + (k - 1) \, j, \qquad j \leftarrow j \cdot s .$$

## In this project

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

## Convolutions on spectrograms

A spectrogram is not a photograph. Shifting a pattern along the **time** axis leaves its
meaning unchanged: a snore is a snore whenever it starts. Shifting it along the
**frequency** axis changes the sound: a pattern at 100 Hz and the same pattern at 4 kHz
are different events. Convolutions treat both axes the same way. They still work well in
practice, partly because the deeper layers and the final pooling can learn where things
are, but this is a modelling approximation worth keeping in mind.

## Pitfalls

- Forgetting `padding=1` with a 3×3 kernel shrinks the map by 2 at every layer.
  `BottleneckBlock` in `src/models/cnn_blocks.py` does exactly that in its middle
  convolution; check whether you want it before using the block.
- An odd size with stride 2 is rounded: 101 frames become 51, then 26, then 13.

## Exercises

1. Compute the parameter count of the second `ConvBlock` (32 → 64). Check it with
   `sum(p.numel() for p in block.parameters())`.
2. Apply the output-size formula to the mobile model: start from 40 × 101 and apply
   three convolutions with `k=3, p=1, s=2`.
3. Reproduce the receptive-field table. What would it be with 5×5 kernels?

## Further reading

- Goodfellow, Bengio and Courville, *Deep Learning*, chapter 9.
- Dumoulin and Visin, "A guide to convolution arithmetic for deep learning" (2016).
- Next: [7. Normalisation, activation and pooling](07-normalization-activation-pooling.md).
