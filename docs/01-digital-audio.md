# 1. Digital audio

## Learning goals

- Explain what sampling rate and bit depth mean.
- State the Nyquist limit and why this project works at 16 kHz.
- Read the preprocessing code and predict the shape of its output.

## Theory

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

## In this project

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

## Pitfalls

- **Mismatched sampling rates.** A model trained at 16 kHz and fed 44.1 kHz audio sees
  every frequency shifted down by a factor 2.76. Nothing crashes; the predictions are
  just wrong. The checkpoint therefore stores `sample_rate`, and `src/train.py` refuses
  to train if the requested rate differs from the one of the preprocessed data.
- **Padding.** The first version of this project padded every 1-second clip to 3
  seconds, so two thirds of each input was silence. Always check the real duration of
  your data before choosing a window length.
- **Integer versus float.** PCM-16 values passed without the division by 32768 are
  32 768 times too large for a model trained on floats.

## Exercises

1. How many samples does a 2.5 s clip contain at 16 kHz? At 44.1 kHz?
2. A pure tone at 10 kHz is sampled at 16 kHz without a low-pass filter. At which
   frequency does it appear? (Hint: $f_s - f$.)
3. Open a file from `data/raw/1/` with `librosa.load(path, sr=None)` and print its native
   sampling rate, its length and the minimum and maximum sample values.

## Further reading

- Oppenheim and Schafer, *Discrete-Time Signal Processing*, chapter 4 (sampling).
- Next: [2. Spectrograms](02-spectrograms.md).
