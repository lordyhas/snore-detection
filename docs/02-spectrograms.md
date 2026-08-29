# 2. Spectrograms

## Learning goals

- Describe the short-time Fourier transform (STFT) and its three parameters.
- Compute the shape of a spectrogram from the signal length.
- Explain the trade-off between time resolution and frequency resolution.

## Theory

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

### Shapes

For a real signal only the first $N/2 + 1$ frequency bins are distinct. With centred
frames (PyTorch pads $N/2$ samples on each side, `center=True`), a signal of $L$ samples
gives

$$\text{bins} = \frac{N}{2} + 1, \qquad \text{frames} = 1 + \left\lfloor \frac{L}{H} \right\rfloor .$$

### The resolution trade-off

Bin $k$ corresponds to the frequency $k \, f_s / N$, so the frequency resolution is
$f_s / N$ hertz, and a frame lasts $N / f_s$ seconds. A longer frame separates close
frequencies better but blurs events in time. The product of the two resolutions is
constant: this is a form of the uncertainty principle, and no choice of parameters
escapes it.

## In this project

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

## Pitfalls

- A 1-second clip with `hop_length=512` has only 32 frames. Four max-pooling layers
  reduce 32 to 2: there is little room left. Short clips call for a small hop.
- The window must live on the same device as the signal. Registering it as a *buffer*
  (`register_buffer`) makes it follow the model in `.to(device)`.

## Exercises

1. Verify the "Bins" and "Frames" columns of the table with the formulas above.
2. Which `n_fft` gives a 25 ms frame at 16 kHz? What is the frequency resolution?
3. In the notebook `notebooks/data_visualisation.ipynb`, plot the same clip with
   `n_fft=256` and `n_fft=2048`. Describe what changes.

## Further reading

- Müller, *Fundamentals of Music Processing*, chapter 2.
- Next: [3. Mel and log scales](03-mel-and-log-scales.md).
