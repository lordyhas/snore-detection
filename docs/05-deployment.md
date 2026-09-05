# Part V - Deployment

A trained model is a research result. A model that runs on a phone all night is a
product. This part covers the step between the two: turning the PyTorch model into a
portable file, checking that the file computes the same thing, and running it on a
continuous audio stream under the constraints of a mobile device.

**Contents**

- [5.1 Exporting to ONNX](#51-exporting-to-onnx)
- [5.2 On-device inference](#52-on-device-inference)
- [Key points](#key-points)

## 5.1 Exporting to ONNX

### Learning goals

- Explain what an exported model is and why the Python code is not shipped.
- Read `src/export.py`.
- Learn from three real bugs found while exporting the models of this project.

### Theory

#### Why export

A PyTorch model is Python code plus a `state_dict`. A phone does not run Python. An
**exported model** is a self-contained description of the computation, a graph of
standard operations with the weights embedded, that a small runtime can execute in any
language.

**ONNX** (Open Neural Network Exchange) is such a format. The set of available
operations is versioned by an **opset** number. **ONNX Runtime** executes the files on
servers, desktops, Android and iOS.

#### How the graph is obtained

The exporter runs the model once on an example input and records the operations, then
translates each one into ONNX operations. Consequences:

- Python control flow that depends on *values* is frozen to the branch taken by the
  example.
- The mode matters: in training mode, dropout and augmentation would be recorded as
  part of the model.
- Dimensions are fixed to those of the example unless declared **dynamic**.

### In this project

```python
wrapper = _ProbabilityModel(model).eval()
torch.onnx.export(
    wrapper,
    (torch.zeros(2, num_samples),),
    onnx_path,
    input_names=["waveform"],
    output_names=["probability"],
    dynamic_shapes={"waveform": {0: torch.export.Dim("batch")}},
    opset_version=OPSET_VERSION,
    dynamo=True,
    external_data=False,
)
```

- `.eval()` removes the SpecAugment masks (`if self.training:` in
  `SnoreMobileNet.forward`) and freezes batch normalisation.
- The wrapper adds the sigmoid: the file outputs a probability.
- The batch dimension is dynamic; the 16 000 samples are fixed.
- `external_data=False` keeps the weights inside the single `.onnx` file.
- Metadata (`sample_rate`, `num_samples`) is written into the file, so the consumer does
  not have to guess the expected audio format. `src/inference.py` reads it back.

Because the spectrogram is computed inside the model, the exported graph contains an
`STFT` operation, available from opset 17. The app passes raw audio and nothing else.

### Verify, always

An export that succeeds is not an export that is correct. `export_model` runs both
models on the whole validation set and compares:

```python
max_difference = float(np.abs(onnx_probabilities - torch_probabilities).max())
...
if not max_difference <= 1e-2:  # also catches NaN
    raise RuntimeError("The exported model does not match the PyTorch model.")
```

This check caught three bugs before any model was published.

#### Bug 1: a check that could not fail

The first comparison used a randomly initialised model on random noise, and found
differences below $10^{-4}$. But an untrained model outputs about 0.5 for any input, so the
test would have passed even with a broken front-end. **A test is only worth the cases
that could make it fail.** The comparison now uses trained weights and real data.

#### Bug 2: a floor that amplified rounding noise

With the trained model, PyTorch and ONNX Runtime disagreed by up to 0.85 in probability
on some clips. The spectrograms differed only by rounding, about $3 \times 10^{-7}$
relative to the largest value. The cause was the original line

```python
log_mel = torch.log(mel + 1e-6)
```

In bands where the true power is zero, the result was the logarithm of rounding noise.
For instance $\log(10^{-6})$ in one runtime against $\log(10^{-6} + 10^{-4})$ in the
other is a difference of 4.6 after the log, from an input difference of $10^{-4}$. The
fix places the floor relative to the peak of the clip, well above the noise
([section 1.3](01-audio-and-tensors.md#13-mel-and-log-scales)).

The lesson reaches beyond export: a model sensitive to the last bits of its input would
have been just as sensitive to the microphone of a phone.

#### Bug 3: a constant that disappeared, and a comparison blind to NaN

To protect the logarithm against an all-zero clip, the code first added a tiny constant,
`+ 1e-10`. In the exported graph that addition was no longer effective, the logarithm of
zero gave $-\infty$, and the normalisation turned it into `NaN` for silent clips.
Replacing the addition by an explicit `torch.clamp(..., min=1e-10)` solved it.

The check did not notice at first, because it was written `if max_difference > 1e-3`.
Any comparison with `NaN` is false, so `NaN > 1e-3` is false and the check passed. The
condition is now written `if not max_difference <= 1e-2`, which fails on `NaN`.

After the fixes, the largest difference in probability over the validation set is about
$2 \times 10^{-3}$ for `mobile_small` and $9 \times 10^{-4}$ for `mobile_tiny`.

### Pitfalls

- An operation may be unsupported by the exporter or by the mobile runtime. Check the
  list of operations of the exported graph against the runtime you target.
- Exporters and runtimes evolve: record the versions used (`requirements.txt`) and the
  opset.
- Constant folding and graph optimisations can change numerics. Prefer explicit,
  well-conditioned formulas over tiny epsilons.

### Exercises

1. Open an exported file with `onnx.load` and print the set of operation types in
   `model.graph.node`.
2. Export `mobile_tiny` *without* calling `.eval()`. What does the verification report,
   and why?
3. Reintroduce `torch.log(mel + 1e-6)`, retrain, export, and reproduce bug 2.
4. Evaluate `float("nan") > 1`, `float("nan") <= 1` and `not float("nan") <= 1` in
   Python.
5. Write a test that feeds an all-zero clip to both models and asserts a finite output.

### Further reading

- ONNX documentation, "Operators" and "Versioning".
- PyTorch documentation, `torch.onnx`.

## 5.2 On-device inference

### Learning goals

- Turn a clip classifier into a detector that runs on a continuous stream.
- Reason about latency, memory and energy on a phone.
- Know the main compression techniques and when they are worth using.

### From clips to streams

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

#### Post-processing

Window decisions are noisy. Simple temporal rules improve precision considerably:

- **smoothing**: average the probabilities of neighbouring windows;
- **hysteresis**: start an event above a high threshold, end it below a lower one;
- **minimum duration and rhythm**: snoring repeats with breathing, every few seconds.

The SleepLab app already implements the last idea with signal-processing rules: bursts
of low-frequency energy that repeat at breathing cadence. A learned classifier and such
rules are complementary. A sensible first integration is a **cascade**: the cheap rules
run all the time, and the network is only asked for a second opinion on the bursts they
select.

### Constraints of a phone

| Constraint | Why it matters for an all-night recording                   |
|------------|-------------------------------------------------------------|
| Energy     | the CPU must sleep most of the time, for eight hours        |
| Latency    | each window must be processed faster than it is recorded    |
| Memory     | the model stays loaded all night                            |
| App size   | the model and its runtime are downloaded by every user      |

#### Orders of magnitude for this project

- Model file: 161 KB (`mobile_tiny`) or 258 KB (`mobile_small`).
- One window takes about 2.5 ms with ONNX Runtime on the development PC. This was not
  measured on a phone; expect it to be slower by a factor that only a measurement on
  the device can give.
- The runtime library weighs far more than the models: several megabytes per processor
  architecture.

With a cascade, the network runs only on candidate bursts, a small fraction of the
night, and its energy cost becomes marginal.

### Making models smaller and faster

**Architecture** comes first: `mobile_tiny` has 120 times fewer parameters than
`audio_cnn` ([section 2.4](02-convolutional-networks.md#24-depthwise-separable-convolutions)).

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
recordings ([section 4.2](04-evaluation.md#42-generalisation-and-data-leakage)), then make it small.

### The interface is part of the model

Most deployment bugs are mismatches between training and inference: a different sampling
rate, integers instead of floats, a different window length, a spectrogram computed by
another library with slightly different conventions. This project removes that class of
bugs by construction:

- the spectrogram is computed inside the exported graph;
- the expected format is stored in the file's metadata;
- the input is normalised per clip, so microphone gain does not matter.

See `models/README.md` for the exact interface and a Kotlin example.

### Pitfalls

- The dataset clips are clean and centred; the windows of a stream cut events anywhere.
  The random time shift used in training
  ([section 3.5](03-training.md#35-regularisation-and-augmentation)) addresses this only in part.
- Measure latency on the oldest device you intend to support, not on the newest.
- Create the inference session once and reuse it. Creating it per window costs far more
  than the inference itself.

### Exercises

1. Run `python -m src.inference` on the files of `data/inference/` with both models and
   compare the per-file verdicts.
2. Modify `predict_file` to accept the hop as an argument, and study how the verdict of
   a long file changes with hops of 1 s, 0.5 s and 0.25 s.
3. Implement hysteresis on a list of window probabilities: start at 0.7, stop at 0.3.
4. Measure the mean inference time of each ONNX model over 1000 windows on your
   machine.
5. Apply `onnxruntime.quantization.quantize_dynamic` to an exported model. Compare file
   size, speed and validation accuracy.

### Further reading

- Zhang et al. (2017), "Hello Edge: Keyword Spotting on Microcontrollers".
- Jacob et al. (2018), "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference".
- Hinton, Vinyals and Dean (2015), "Distilling the Knowledge in a Neural Network".

## Key points

- An exported model is a graph of standard operations with the weights inside; it is recorded from one run of the model in evaluation mode.
- A successful export proves nothing. Compare the exported model with the original on real data and trained weights, and make sure the comparison can fail.
- Tiny epsilons and logarithms of near-zero values amplify rounding differences between runtimes; prefer well-conditioned formulas.
- Putting the preprocessing inside the model, and the expected format in the file's metadata, removes the most common class of deployment bugs.
- On a stream, classify overlapping windows and post-process the decisions; a cheap detector can select what the network examines.
- Make the model right on target data before making it smaller.

---

Previous: [Part IV - Evaluation and scientific method](04-evaluation.md) | [Contents](README.md) | Next: [Part VI - Practice](06-practice.md)
