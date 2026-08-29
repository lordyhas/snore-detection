# 9. Depthwise separable convolutions

## Learning goals

- Factorise a convolution into a depthwise and a pointwise step.
- Quantify the savings in parameters.
- Read the mobile models of this project.

## Theory

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

## In this project

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

### Comparison

| Model          | Parameters | Validation accuracy |
|----------------|------------|---------------------|
| `audio_cnn`    | 1 189 153  | 0.99                |
| `mobile_small` | 35 137     | 0.975               |
| `mobile_tiny`  | 9 889      | 0.98                |

The difference between 0.975 and 0.99 on 200 validation clips is three clips: it is
within the noise of such a small set, and the three models should be considered
equivalent on this split. The smallest one is not worse; do not conclude that it is
better either.

## Pitfalls

- `groups=in_channels` requires the output channels of that layer to be a multiple of
  the input channels; here they are equal.
- Depthwise convolutions are cheap in parameters but not always fast: their speed
  depends on how well the runtime optimises them. Measure on the target device.

## Exercises

1. Check the "Parameters" column for the block 32 → 64 by hand: depthwise, batch
   normalisation, pointwise, batch normalisation.
2. Compute the saving ratio for 128 → 256 channels with the formula, then by counting.
3. Build `SnoreMobileNet(width=8)`. How many parameters? Train it with a new entry in
   `src/models/registry.py` and see how small the model can get.
4. How many parameters would `mobile_tiny` have if its four separable blocks were
   standard 3×3 convolutions?

## Further reading

- Howard et al. (2017); Zhang et al. (2017), "Hello Edge". See
  [References](25-references.md).
- Next: [10. Residual and dense blocks](10-residual-and-dense-blocks.md).
