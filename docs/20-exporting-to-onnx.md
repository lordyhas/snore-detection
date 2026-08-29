# 20. Exporting to ONNX

## Learning goals

- Explain what an exported model is and why the Python code is not shipped.
- Read `src/export.py`.
- Learn from three real bugs found while exporting the models of this project.

## Theory

### Why export

A PyTorch model is Python code plus a `state_dict`. A phone does not run Python. An
**exported model** is a self-contained description of the computation, a graph of
standard operations with the weights embedded, that a small runtime can execute in any
language.

**ONNX** (Open Neural Network Exchange) is such a format. The set of available
operations is versioned by an **opset** number. **ONNX Runtime** executes the files on
servers, desktops, Android and iOS.

### How the graph is obtained

The exporter runs the model once on an example input and records the operations, then
translates each one into ONNX operations. Consequences:

- Python control flow that depends on *values* is frozen to the branch taken by the
  example.
- The mode matters: in training mode, dropout and augmentation would be recorded as
  part of the model.
- Dimensions are fixed to those of the example unless declared **dynamic**.

## In this project

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

## Verify, always

An export that succeeds is not an export that is correct. `export_model` runs both
models on the whole validation set and compares:

```python
max_difference = float(np.abs(onnx_probabilities - torch_probabilities).max())
...
if not max_difference <= 1e-2:  # also catches NaN
    raise RuntimeError("The exported model does not match the PyTorch model.")
```

This check caught three bugs before any model was published.

### Bug 1: a check that could not fail

The first comparison used a randomly initialised model on random noise, and found
differences below $10^{-4}$. But an untrained model outputs about 0.5 for any input, so the
test would have passed even with a broken front-end. **A test is only worth the cases
that could make it fail.** The comparison now uses trained weights and real data.

### Bug 2: a floor that amplified rounding noise

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
([chapter 3](03-mel-and-log-scales.md)).

The lesson reaches beyond export: a model sensitive to the last bits of its input would
have been just as sensitive to the microphone of a phone.

### Bug 3: a constant that disappeared, and a comparison blind to NaN

To protect the logarithm against an all-zero clip, the code first added a tiny constant,
`+ 1e-10`. In the exported graph that addition was no longer effective, the logarithm of
zero gave $-\infty$, and the normalisation turned it into `NaN` for silent clips.
Replacing the addition by an explicit `torch.clamp(..., min=1e-10)` solved it.

The check did not notice at first, because it was written `if max_difference > 1e-3`.
Any comparison with `NaN` is false, so `NaN > 1e-3` is false and the check passed. The
condition is now written `if not max_difference <= 1e-2`, which fails on `NaN`.

After the fixes, the largest difference in probability over the validation set is about
$2 \times 10^{-3}$ for `mobile_small` and $9 \times 10^{-4}$ for `mobile_tiny`.

## Pitfalls

- An operation may be unsupported by the exporter or by the mobile runtime. Check the
  list of operations of the exported graph against the runtime you target.
- Exporters and runtimes evolve: record the versions used (`requirements.txt`) and the
  opset.
- Constant folding and graph optimisations can change numerics. Prefer explicit,
  well-conditioned formulas over tiny epsilons.

## Exercises

1. Open an exported file with `onnx.load` and print the set of operation types in
   `model.graph.node`.
2. Export `mobile_tiny` *without* calling `.eval()`. What does the verification report,
   and why?
3. Reintroduce `torch.log(mel + 1e-6)`, retrain, export, and reproduce bug 2.
4. Evaluate `float("nan") > 1`, `float("nan") <= 1` and `not float("nan") <= 1` in
   Python.
5. Write a test that feeds an all-zero clip to both models and asserts a finite output.

## Further reading

- ONNX documentation, "Operators" and "Versioning".
- PyTorch documentation, `torch.onnx`.
- Next: [21. On-device inference](21-on-device-inference.md).
