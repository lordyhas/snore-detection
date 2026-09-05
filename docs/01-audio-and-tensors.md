# Part I - From sound to tensors

A neural network does not hear. It receives an array of numbers, and everything it
can learn depends on how the sound was turned into that array. This part follows a
one-second clip from the audio file to the batch of tensors that enters the model:
sampling, the time-frequency representation, the perceptual scales that make it
compact, and the PyTorch objects that carry it.

**Contents**

- [1.1 Digital audio](#11-digital-audio)
- [1.2 Spectrograms](#12-spectrograms)
- [1.3 Mel and log scales](#13-mel-and-log-scales)
- [1.4 Tensors and shapes](#14-tensors-and-shapes)
- [1.5 Datasets and data loaders](#15-datasets-and-data-loaders)
- [Key points](#key-points)

## 1.1 Digital audio

### Learning goals

- Explain what sampling rate and bit depth mean.
- State the Nyquist limit and why this project works at 16 kHz.
- Read the preprocessing code and predict the shape of its output.

### Theory

A microphone produces a continuous pressure signal $x(t)$. A computer stores a
*sampled* version of it:

$$x[n] = x(n / f_s), \qquad n = 0, 1, 2, \dots$$

where $f_s$ is the **sampling rate** in hertz. One second of audio at 16 kHz is a vector
of 16 000 numbers.

**Nyquist-Shannon theorem.** A signal sampled at $f_s$ can only represent frequencies
below $f_s / 2$ (the *Nyquist frequency*). Content above that limit is not lost, it is
folded back onto lower frequencies (*aliasing*), which is why a resampler applies a
low-pass filter before dropping samples.

**Bit depth.** Each sample is quantised. 16-bit PCM stores integers in
$[-32768, 32767]$. Neural networks work with floats, so the convention is to divide by
32768 and obtain values in $[-1, 1)$.

**Channels.** Stereo files hold two signals. Averaging them gives a mono signal, which is
enough when the position of the source does not matter.

### In this project

The raw clips are 1 second long, sampled at 44.1 kHz. `src/data/preprocess.py` loads each
one with:

```python
waveform, _ = librosa.load(file_path, sr=sample_rate, mono=True)
```

`librosa.load` resamples to `sample_rate` (16 000 by default here), averages the channels
and returns `float32` values in $[-1, 1]$. Every clip is then cropped or zero-padded to
exactly `sample_rate * duration` samples, so the whole dataset fits in one array of shape
`(1000, 16000)`.

Why 16 kHz? The Nyquist frequency is then 8 kHz. Snoring has most of its energy below
about 500 Hz, with harmonics and noise components up to a few kilohertz, so little useful
information is lost, while each clip is almost three times smaller than at 44.1 kHz.
It is also the rate at which the SleepLab Android app records.

### Pitfalls

- **Mismatched sampling rates.** A model trained at 16 kHz and fed 44.1 kHz audio sees
  every frequency shifted down by a factor 2.76. Nothing crashes; the predictions are
  just wrong. The checkpoint therefore stores `sample_rate`, and `src/train.py` refuses
  to train if the requested rate differs from the one of the preprocessed data.
- **Padding.** The first version of this project padded every 1-second clip to 3
  seconds, so two thirds of each input was silence. Always check the real duration of
  your data before choosing a window length.
- **Integer versus float.** PCM-16 values passed without the division by 32768 are
  32 768 times too large for a model trained on floats.

### Exercises

1. How many samples does a 2.5 s clip contain at 16 kHz? At 44.1 kHz?
2. A pure tone at 10 kHz is sampled at 16 kHz without a low-pass filter. At which
   frequency does it appear? (Hint: $f_s - f$.)
3. Open a file from `data/raw/1/` with `librosa.load(path, sr=None)` and print its native
   sampling rate, its length and the minimum and maximum sample values.

### Further reading

- Oppenheim and Schafer, *Discrete-Time Signal Processing*, chapter 4 (sampling).

## 1.2 Spectrograms

### Learning goals

- Describe the short-time Fourier transform (STFT) and its three parameters.
- Compute the shape of a spectrogram from the signal length.
- Explain the trade-off between time resolution and frequency resolution.

### Theory

A waveform tells *when* the air moves but hides *which frequencies* are present. The
discrete Fourier transform (DFT) gives the frequency content of a whole signal, but then
the timing is lost. The **short-time Fourier transform** is the compromise: cut the
signal into short overlapping frames, multiply each by a window, and take the DFT of
each frame.

$$X[k, t] = \sum_{n=0}^{N-1} w[n] \; x[n + tH] \; e^{-2\pi i k n / N}$$

- $N$ = `n_fft`, the frame length in samples;
- $H$ = `hop_length`, the step between two frames;
- $w$ = the window, usually a Hann window, which fades each frame in and out to limit
  *spectral leakage* (energy smeared across neighbouring frequencies).

The **power spectrogram** is $|X[k, t]|^2$: a real, non-negative image with frequency on
one axis and time on the other. The phase is discarded, as it carries little information
for classification.

#### Shapes

For a real signal only the first $N/2 + 1$ frequency bins are distinct. With centred
frames (PyTorch pads $N/2$ samples on each side, `center=True`), a signal of $L$ samples
gives

$$\text{bins} = \frac{N}{2} + 1, \qquad \text{frames} = 1 + \left\lfloor \frac{L}{H} \right\rfloor .$$

#### The resolution trade-off

Bin $k$ corresponds to the frequency $k \, f_s / N$, so the frequency resolution is
$f_s / N$ hertz, and a frame lasts $N / f_s$ seconds. A longer frame separates close
frequencies better but blurs events in time. The product of the two resolutions is
constant: this is a form of the uncertainty principle, and no choice of parameters
escapes it.

### In this project

Two front-ends coexist:

| Model          | `n_fft` | `hop_length` | Frame  | Hop   | Resolution | Bins | Frames (1 s) |
|----------------|---------|--------------|--------|-------|------------|------|--------------|
| `audio_cnn`    | 1024    | 512          | 64 ms  | 32 ms | 15.6 Hz    | 513  | 32           |
| mobile models  | 512     | 160          | 32 ms  | 10 ms | 31.25 Hz   | 257  | 101          |

- For `audio_cnn`, the spectrogram is computed in the dataset
  (`torchaudio.transforms.MelSpectrogram` in `src/data/data_manager.py`).
- For the mobile models it is computed *inside the model*, in `LogMelFrontend`
  (`src/models/mobile_cnn.py`), with a direct call to `torch.stft`:

```python
stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                  window=self.window, return_complex=True)
power = torch.view_as_real(stft).pow(2).sum(-1)  # (batch, n_freqs, frames)
```

`view_as_real` turns each complex number into a pair (real, imaginary); squaring and
summing the pair gives $|X|^2$.

### Pitfalls

- A 1-second clip with `hop_length=512` has only 32 frames. Four max-pooling layers
  reduce 32 to 2: there is little room left. Short clips call for a small hop.
- The window must live on the same device as the signal. Registering it as a *buffer*
  (`register_buffer`) makes it follow the model in `.to(device)`.

### Exercises

1. Verify the "Bins" and "Frames" columns of the table with the formulas above.
2. Which `n_fft` gives a 25 ms frame at 16 kHz? What is the frequency resolution?
3. In the notebook `notebooks/data_visualisation.ipynb`, plot the same clip with
   `n_fft=256` and `n_fft=2048`. Describe what changes.

### Further reading

- Müller, *Fundamentals of Music Processing*, chapter 2.

## 1.3 Mel and log scales

### Learning goals

- Explain why spectrograms are warped to the mel scale and compressed with a logarithm.
- Write the mel filter bank as a matrix product.
- Understand why input normalisation matters, and how it is done here.

### Theory

#### The mel scale

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

#### The logarithm

Sound power spans many orders of magnitude; a loud snore can carry a million times the
power of the background. Perceived loudness is roughly logarithmic, and networks train
better on inputs with a moderate range, so we take the log. In decibels:

$$\text{dB} = 10 \, \log_{10}(\text{power}).$$

Because $\log 0 = -\infty$, a small floor is always added or clamped first.

#### Normalisation

Gradient descent behaves best when inputs have zero mean and unit variance. Two options:

- **dataset statistics**: one mean and one standard deviation for all clips;
- **per-instance**: each clip is normalised with its own statistics.

Per-instance normalisation has a useful property for audio. Multiplying the waveform by
a gain $g$ multiplies the power by $g^2$, which *adds* the constant $\log g^2$ to the
log-spectrogram. Subtracting the mean of the clip removes that constant: the model
becomes insensitive to the recording level.

### In this project

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
   explained in [section 5.1](05-deployment.md#51-exporting-to-onnx), made the model depend on
   rounding noise.
2. **The clamp** only matters for a clip that is entirely zero, where `peak` is 0.
3. **The `1e-4` under the square root** bounds the division for a constant clip.

The measured effect: the exported model returns the same probability for a signal and
for the same signal 100 times quieter.

### Pitfalls

- The mel scale is motivated by speech and music perception. For snoring it is a
  reasonable default, not a proven optimum.
- Normalising per instance erases absolute loudness. If loudness were a useful cue it
  would have to be given to the model separately.

### Exercises

1. Convert 100 Hz, 1000 Hz and 8000 Hz to mels. What do you notice?
2. Show that $\log(g^2 P) = \log P + 2 \log g$ and conclude that mean subtraction
   removes the gain.
3. Replace the relative floor by `torch.log(mel + 1e-6)`, retrain `mobile_tiny`, and
   compare the predictions for a clip and for the same clip multiplied by 0.01.

### Further reading

- Stevens, Volkmann and Newman (1937); Davis and Mermelstein (1980). See
  [References](references.md).

## 1.4 Tensors and shapes

### Learning goals

- Manipulate PyTorch tensors: shape, dtype, device.
- Use `unsqueeze`, `squeeze`, `view`, `transpose` and broadcasting with confidence.
- Follow the shape of the data from the audio file to the loss.

### Theory

A **tensor** is an n-dimensional array with three attributes that explain most runtime
errors:

| Attribute | Example         | Typical error when wrong                         |
|-----------|-----------------|--------------------------------------------------|
| `shape`   | `(32, 1, 128, 32)` | `size mismatch`, silent wrong broadcasting    |
| `dtype`   | `torch.float32` | `expected Float but got Long`                    |
| `device`  | `cuda:0`        | `expected all tensors to be on the same device`  |

#### Shape conventions

- 2-D convolutions expect `(batch, channels, height, width)`, written **NCHW**.
- For a spectrogram, height is frequency and width is time; there is one channel.

#### Reshaping without copying

```python
x = torch.zeros(16000)        # (16000,)
x.unsqueeze(0).shape          # (1, 16000)      add a dimension of size 1
x.view(4, 4000).shape         # (4, 4000)       same data, new shape
x.view(-1, 1).shape           # (16000, 1)      -1 = "whatever fits"
```

`transpose(1, 2)` swaps two dimensions; `flatten(1)` merges every dimension after the
first.

#### Broadcasting

When two shapes differ, PyTorch aligns them from the right and stretches dimensions of
size 1. `(8, 40, 101) - (8, 1, 1)` subtracts one number per clip from a whole
spectrogram. This is what `keepdim=True` prepares:

```python
log_mel.mean(dim=(1, 2), keepdim=True).shape   # (8, 1, 1), ready to broadcast
```

Broadcasting is convenient and dangerous: `(32,)` against `(32, 1)` silently produces a
`(32, 32)` matrix.

### In this project

#### `audio_cnn`

| Step                               | Shape               | dtype     |
|------------------------------------|---------------------|-----------|
| `dataset.npz`, `x_train`           | `(800, 16000)`      | float32   |
| `AudioDataset.__getitem__` output  | `(1, 128, 32)`      | float32   |
| batch from the `DataLoader`        | `(32, 1, 128, 32)`  | float32   |
| labels from the `DataLoader`       | `(32,)`             | float32   |
| labels after `.view(-1, 1)`        | `(32, 1)`           | float32   |
| logits                             | `(32, 1)`           | float32   |

#### Mobile models

| Step                               | Shape               |
|------------------------------------|---------------------|
| batch from the `DataLoader`        | `(32, 16000)`       |
| after `torch.stft`                 | `(32, 257, 101)` complex |
| after the mel filter bank          | `(32, 40, 101)`     |
| after `unsqueeze(1)`               | `(32, 1, 40, 101)`  |
| logits                             | `(32, 1)`           |

Two lines of `src/train.py` exist only because of shapes and dtypes:

```python
labels = labels.to(device, non_blocking=True).view(-1, 1)
```

The loss compares logits `(32, 1)` with labels: without `.view(-1, 1)` the labels would
be `(32,)` and the loss would raise an error. And in the dataset the label is created as
`float32`, because `BCEWithLogitsLoss` needs float targets even though they are 0 or 1.

### Pitfalls

- `view` requires contiguous memory; after a `transpose`, use `reshape`.
- `x.numpy()` and `torch.from_numpy(a)` share memory with the original: modifying one
  modifies the other.
- The last batch of an epoch can be smaller than `batch_size`. Never hard-code the
  batch dimension: use `-1` or `x.size(0)`.

### Exercises

1. What is the shape of `torch.zeros(8, 40, 101).mean(dim=(1, 2))`? And with
   `keepdim=True`?
2. In a Python shell, build `AudioDataset` on ten random waveforms and print the shape
   of one item for `features=True` and `features=False`.
3. Predict, then check: `(torch.zeros(32) - torch.zeros(32, 1)).shape`.

### Further reading

- PyTorch documentation, "Broadcasting semantics" and "Tensor views".

## 1.5 Datasets and data loaders

### Learning goals

- Implement the `Dataset` protocol and understand what a `DataLoader` adds.
- Explain mini-batches, shuffling and epochs.
- Decide what belongs in preprocessing, in the dataset, and in the model.

### Theory

#### Mini-batch stochastic gradient descent

The loss we want to minimise is an average over the whole training set. Computing it
exactly at every step is wasteful, so we estimate it on a random **mini-batch** of $B$
examples. The estimate is noisy but unbiased, and the noise itself helps to escape poor
solutions. One pass over all the data is an **epoch**. With 800 training clips and
$B = 32$, an epoch is 25 gradient steps.

#### The two classes

A **`Dataset`** answers two questions:

```python
def __len__(self):            # how many examples?
def __getitem__(self, idx):   # give me example number idx
```

A **`DataLoader`** wraps a dataset and takes care of the rest: drawing indices
(`shuffle=True`), stacking examples into a batch, optionally loading in parallel
processes (`num_workers`) and preparing fast transfers to the GPU (`pin_memory`).

Shuffle the training set at every epoch, so that the batches differ from one epoch to
the next. Never shuffle for evaluation: it changes nothing and makes results harder to
compare.

### In this project

The work is split in three stages.

**1. Preprocessing, once** (`src/data/preprocess.py`). Decoding and resampling 1000
files takes a few seconds; doing it at every epoch would dominate the training time. The
result is saved as one compressed file, `data/processed/dataset.npz`, with the split
already made.

Note how the file is written: to a temporary file first, then `os.replace`. A crash in
the middle can never leave a half-written dataset.

**2. Dataset, at every access** (`src/data/data_manager.py`). `AudioDataset` keeps the
waveforms in memory and transforms one at a time:

```python
spec = self.to_db(self.mel(waveform))  # (1, n_mels, frames)
if self.train:
    spec = self.time_mask(self.freq_mask(spec))
```

Anything *random* must happen here and not in preprocessing: the augmentation has to
differ at every epoch (see [section 3.5](03-training.md#35-regularisation-and-augmentation)).

**3. Model.** The mobile models compute the spectrogram themselves, so
`AudioDataset(features=False)` returns the raw waveform. The reason is deployment: the
phone then needs no signal-processing code.

The loaders are created in `src/train.py`:

```python
generator = torch.Generator().manual_seed(RANDOM_SEED)
train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
```

The seeded `generator` makes the shuffling order reproducible.

### Pitfalls

- **`num_workers` on Windows.** Worker processes are started with *spawn*: the script is
  imported again in each worker, so the entry point must be protected by
  `if __name__ == "__main__":`. This project defaults to `num_workers=0`, which is
  fast enough for 800 short clips held in memory.
- **Random augmentation in workers.** Each worker has its own random state. PyTorch
  seeds them differently for you, but NumPy's global state used to be duplicated across
  workers; prefer `torch` random functions inside a dataset.
- **Transforms on the validation set.** `train=False` disables augmentation. Evaluating
  on augmented data would measure the wrong thing.

### Exercises

1. Iterate once over `train_loader` and count the batches. What is the size of the
   last one if the training set has 810 examples?
2. Time one epoch of data loading alone (no model) with `num_workers=0` and
   `num_workers=2`. Which one is faster here, and why?
3. Write a minimal `Dataset` that returns `(idx, idx ** 2)` and wrap it in a
   `DataLoader` with `batch_size=4`.

### Further reading

- PyTorch tutorial, "Datasets & DataLoaders".
