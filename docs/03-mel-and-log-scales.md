# 3. Mel and log scales

## Learning goals

- Explain why spectrograms are warped to the mel scale and compressed with a logarithm.
- Write the mel filter bank as a matrix product.
- Understand why input normalisation matters, and how it is done here.

## Theory

### The mel scale

Human hearing resolves low frequencies much more finely than high ones: 100 Hz and
200 Hz sound far apart, 7100 Hz and 7200 Hz almost identical. The **mel scale**
(Stevens, Volkmann and Newman, 1937) is a frequency axis on which equal distances sound
equally far apart. The formula used by torchaudio by default is

$$m = 2595 \, \log_{10}\!\left(1 + \frac{f}{700}\right).$$

A **mel filter bank** is a set of triangular filters, evenly spaced in mels, hence
narrow at low frequencies and wide at high ones. Applying it is a matrix product:

$$M = F^\top P, \qquad F \in \mathbb{R}^{\text{bins} \times \text{mels}}$$

where $P$ is one column of the power spectrogram. With 257 bins and 40 mel bands, the
input becomes 6.4 times smaller while keeping the detail where it matters.

### The logarithm

Sound power spans many orders of magnitude; a loud snore can carry a million times the
power of the background. Perceived loudness is roughly logarithmic, and networks train
better on inputs with a moderate range, so we take the log. In decibels:

$$\text{dB} = 10 \, \log_{10}(\text{power}).$$

Because $\log 0 = -\infty$, a small floor is always added or clamped first.

### Normalisation

Gradient descent behaves best when inputs have zero mean and unit variance. Two options:

- **dataset statistics**: one mean and one standard deviation for all clips;
- **per-instance**: each clip is normalised with its own statistics.

Per-instance normalisation has a useful property for audio. Multiplying the waveform by
a gain $g$ multiplies the power by $g^2$, which *adds* the constant $\log g^2$ to the
log-spectrogram. Subtracting the mean of the clip removes that constant: the model
becomes insensitive to the recording level.

## In this project

`audio_cnn` uses 128 mel bands and `AmplitudeToDB`, without normalisation (the batch
normalisation of the first layer compensates in part).

The mobile models do everything in `LogMelFrontend`:

```python
mel = torch.matmul(power.transpose(1, 2), self.fbanks).transpose(1, 2)
peak = mel.flatten(1).max(dim=1).values.view(-1, 1, 1)
log_mel = torch.log(torch.clamp(mel + 1e-5 * peak, min=1e-10))
centered = log_mel - log_mel.mean(dim=(1, 2), keepdim=True)
std = (centered.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-4).sqrt()
return (centered / std).unsqueeze(1)
```

Three details are worth studying:

1. **A relative floor.** `1e-5 * peak` puts the floor 50 dB under the loudest band of
   the clip. An absolute floor such as `1e-6` would break gain invariance and, as
   explained in [chapter 20](20-exporting-to-onnx.md), made the model depend on
   rounding noise.
2. **The clamp** only matters for a clip that is entirely zero, where `peak` is 0.
3. **The `1e-4` under the square root** bounds the division for a constant clip.

The measured effect: the exported model returns the same probability for a signal and
for the same signal 100 times quieter.

## Pitfalls

- The mel scale is motivated by speech and music perception. For snoring it is a
  reasonable default, not a proven optimum.
- Normalising per instance erases absolute loudness. If loudness were a useful cue it
  would have to be given to the model separately.

## Exercises

1. Convert 100 Hz, 1000 Hz and 8000 Hz to mels. What do you notice?
2. Show that $\log(g^2 P) = \log P + 2 \log g$ and conclude that mean subtraction
   removes the gain.
3. Replace the relative floor by `torch.log(mel + 1e-6)`, retrain `mobile_tiny`, and
   compare the predictions for a clip and for the same clip multiplied by 0.01.

## Further reading

- Stevens, Volkmann and Newman (1937); Davis and Mermelstein (1980). See
  [References](25-references.md).
- Next: [4. Tensors and shapes](04-tensors-and-shapes.md).
