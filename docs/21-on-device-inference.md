# 21. On-device inference

## Learning goals

- Turn a clip classifier into a detector that runs on a continuous stream.
- Reason about latency, memory and energy on a phone.
- Know the main compression techniques and when they are worth using.

## From clips to streams

The models classify exactly one second of audio. A night is eight hours. The standard
bridge is the **sliding window**: cut the stream into overlapping windows, classify
each, then combine the decisions.

`_windows` in `src/inference.py` does it for a file, with a hop of half a window:

```python
starts = list(range(0, waveform.size - num_samples + 1, hop_samples))
```

Overlap matters because a snore that straddles two non-overlapping windows may be
recognised in neither. A hop of 0.5 s doubles the computation; a hop of 0.25 s quadruples
it.

### Post-processing

Window decisions are noisy. Simple temporal rules improve precision considerably:

- **smoothing**: average the probabilities of neighbouring windows;
- **hysteresis**: start an event above a high threshold, end it below a lower one;
- **minimum duration and rhythm**: snoring repeats with breathing, every few seconds.

The SleepLab app already implements the last idea with signal-processing rules: bursts
of low-frequency energy that repeat at breathing cadence. A learned classifier and such
rules are complementary. A sensible first integration is a **cascade**: the cheap rules
run all the time, and the network is only asked for a second opinion on the bursts they
select.

## Constraints of a phone

| Constraint | Why it matters for an all-night recording                   |
|------------|-------------------------------------------------------------|
| Energy     | the CPU must sleep most of the time, for eight hours        |
| Latency    | each window must be processed faster than it is recorded    |
| Memory     | the model stays loaded all night                            |
| App size   | the model and its runtime are downloaded by every user      |

### Orders of magnitude for this project

- Model file: 161 KB (`mobile_tiny`) or 258 KB (`mobile_small`).
- One window takes about 2.5 ms with ONNX Runtime on the development PC. This was not
  measured on a phone; expect it to be slower by a factor that only a measurement on
  the device can give.
- The runtime library weighs far more than the models: several megabytes per processor
  architecture.

With a cascade, the network runs only on candidate bursts, a small fraction of the
night, and its energy cost becomes marginal.

## Making models smaller and faster

**Architecture** comes first: `mobile_tiny` has 120 times fewer parameters than
`audio_cnn` ([chapter 9](09-depthwise-separable-convolutions.md)).

**Quantisation** stores weights, and possibly activations, as 8-bit integers instead of
32-bit floats: the file is about four times smaller and integer arithmetic is faster on
many processors.

- *Post-training quantisation* converts a trained model, using a few hundred
  representative inputs to calibrate the ranges of the activations.
- *Quantisation-aware training* simulates the rounding during training, and recovers
  most of the accuracy that the first method may lose.

**Pruning** removes weights or whole channels that contribute little.

**Knowledge distillation** trains a small *student* to reproduce the outputs of a large
*teacher*; here `audio_cnn` could teach `mobile_tiny`.

None of these is applied in this project. The models are already a few hundred
kilobytes, of which a fixed part is the mel filter bank, so quantisation would save
little. The honest order of priorities is: first make the model *right* on real phone
recordings ([chapter 18](18-generalization-and-data-leakage.md)), then make it small.

## The interface is part of the model

Most deployment bugs are mismatches between training and inference: a different sampling
rate, integers instead of floats, a different window length, a spectrogram computed by
another library with slightly different conventions. This project removes that class of
bugs by construction:

- the spectrogram is computed inside the exported graph;
- the expected format is stored in the file's metadata;
- the input is normalised per clip, so microphone gain does not matter.

See `models/README.md` for the exact interface and a Kotlin example.

## Pitfalls

- The dataset clips are clean and centred; the windows of a stream cut events anywhere.
  The random time shift used in training
  ([chapter 15](15-regularization-and-augmentation.md)) addresses this only in part.
- Measure latency on the oldest device you intend to support, not on the newest.
- Create the inference session once and reuse it. Creating it per window costs far more
  than the inference itself.

## Exercises

1. Run `python -m src.inference` on the files of `data/inference/` with both models and
   compare the per-file verdicts.
2. Modify `predict_file` to accept the hop as an argument, and study how the verdict of
   a long file changes with hops of 1 s, 0.5 s and 0.25 s.
3. Implement hysteresis on a list of window probabilities: start at 0.7, stop at 0.3.
4. Measure the mean inference time of each ONNX model over 1000 windows on your
   machine.
5. Apply `onnxruntime.quantization.quantize_dynamic` to an exported model. Compare file
   size, speed and validation accuracy.

## Further reading

- Zhang et al. (2017), "Hello Edge: Keyword Spotting on Microcontrollers".
- Jacob et al. (2018), "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference".
- Hinton, Vinyals and Dean (2015), "Distilling the Knowledge in a Neural Network".
- Next: [22. Debugging neural networks](22-debugging-neural-networks.md).
