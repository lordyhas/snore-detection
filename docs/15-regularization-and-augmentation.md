# 15. Regularisation and augmentation

## Learning goals

- Define overfitting in terms of the gap between training and validation error.
- Explain dropout, weight decay and data augmentation.
- Read the augmentations of this project and reason about their strength.

## Theory

### Overfitting

A model with enough capacity can memorise its training set, noise included. The symptom
is a low training loss with a higher validation loss. **Regularisation** is anything that
reduces this gap, usually by making memorisation harder than learning the general rule.

This project uses five regularisers; three were covered earlier (weight decay in
[chapter 13](13-optimizers.md), early stopping in
[chapter 14](14-learning-rate-and-early-stopping.md), batch normalisation in
[chapter 7](07-normalization-activation-pooling.md)). The two others follow.

### Dropout

During training, dropout (Srivastava et al., 2014) sets each activation to zero with
probability $p$ and multiplies the survivors by $1 / (1 - p)$, so that the expected value
is unchanged:

$$\tilde h_i = \frac{m_i \, h_i}{1 - p}, \qquad m_i \sim \text{Bernoulli}(1 - p) .$$

No unit can rely on the presence of another one, which discourages fragile
co-adaptations. One interpretation is that each step trains a different sub-network and
the final model averages them. At evaluation time dropout does nothing; like batch
normalisation, it is switched by `model.train()` and `model.eval()`.

### Data augmentation

The most effective regulariser is more data. Augmentation manufactures it by applying
transformations that change the input but not its label. The choice of transformations
encodes knowledge about the problem: a snore remains a snore if it starts 200 ms later.

**SpecAugment** (Park et al., 2019) operates on the spectrogram:

- *frequency masking*: a band of consecutive mel channels is set to zero;
- *time masking*: a span of consecutive frames is set to zero.

The model must then recognise a sound from incomplete evidence, and cannot depend on one
particular band or instant.

## In this project

| Technique          | Where                         | Setting                              |
|--------------------|-------------------------------|--------------------------------------|
| Dropout            | `AudioCNN` classifier         | $p = 0.3$, twice                     |
| Dropout            | `SnoreMobileNet` classifier   | $p = 0.2$                            |
| Weight decay       | `AdamW` in `src/train.py`     | $\lambda = 0.01$                     |
| Frequency masking  | `AudioDataset` (128 bands)    | up to 15 bands                       |
| Time masking       | `AudioDataset` (32 frames)    | up to 6 frames                       |
| Frequency masking  | `SnoreMobileNet` (40 bands)   | up to 6 bands                        |
| Time masking       | `SnoreMobileNet` (101 frames) | up to 15 frames                      |
| Random time shift  | `AudioDataset(features=False)`| circular, any offset                 |

The masks are applied in the dataset for `audio_cnn` and inside the model for the mobile
models, guarded by `if self.training:` so that they disappear at evaluation and in the
exported file.

### Augmentation strength is a hyperparameter

When the clip duration was corrected from 3 seconds to 1, the time mask kept its old
maximum of 30 frames while clips now had 32 frames: a single mask could erase almost the
entire clip, leaving a label with no evidence. The fix scales the mask to the clip:

```python
num_frames = self.num_samples // hop_length + 1
self.time_mask = torchaudio.transforms.TimeMasking(
    time_mask_param=max(1, min(30, num_frames // 5))
)
```

An augmentation that destroys the information defining the label teaches the model
noise. Always look at augmented examples before training.

### The time shift

```python
shift = int(torch.randint(0, self.num_samples, (1,)))
waveform = torch.roll(waveform, shift)
```

`torch.roll` is circular: what leaves on the right re-enters on the left. This creates
an artificial discontinuity at the junction. It is a simple approximation; shifting with
padding or cropping from a longer recording would be more faithful.

## Augmentations not used here

- **Additive noise** at a random signal-to-noise ratio, ideally real bedroom noise.
- **Mixup**: train on convex combinations of two clips and of their labels.
- **Random gain**: useless for the mobile models, whose per-clip normalisation already
  removes gain ([chapter 3](03-mel-and-log-scales.md)).
- **Pitch shifting and time stretching**: to use with care, since they may change what
  makes a snore a snore.

## Pitfalls

- Never augment the validation set.
- torchaudio's masking transforms draw one mask for the whole tensor they receive. In
  the dataset that is one mask per clip; inside the mobile models it is one mask per
  *batch*.

## Exercises

1. Plot one training spectrogram five times with `train=True`. Do the masks look
   reasonable?
2. Train `audio_cnn` with `dropout=0.0` and without masking. Compare the gap between the
   training and validation losses.
3. Implement additive Gaussian noise at a random level in `AudioDataset` for
   `features=False`, and evaluate its effect on the recordings of `data/inference/`.
4. Why is dropout multiplied by $1 / (1 - p)$ during training rather than scaling the
   activations at evaluation time?

## Further reading

- Srivastava et al. (2014); Park et al. (2019). See [References](25-references.md).
- Next: [16. Anatomy of the training loop](16-training-loop.md).
