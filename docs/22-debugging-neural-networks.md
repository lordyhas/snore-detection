# 22. Debugging neural networks

## Learning goals

- Adopt a systematic procedure instead of changing things at random.
- Recognise the signature of the most common bugs.
- Use the sanity checks that catch them early.

## Why it is hard

Ordinary programs fail loudly. Neural networks fail silently: a model with a bug in its
data pipeline still trains, still produces a decreasing loss, and still reaches a
plausible score, only a worse one than it should, or a good one for a bad reason. The
only defence is to verify each component independently before trusting the whole.

## A procedure

### 1. Look at the data

Before any model: plot spectrograms, listen to clips of each class, print shapes, value
ranges and class counts. In this project a glance at the durations would have revealed
that the clips last one second while the code assumed three.

### 2. Check the loss at initialisation

A freshly initialised binary classifier should output probabilities near 0.5, hence a
loss near $\log 2 \approx 0.693$ ([chapter 11](11-loss-functions.md)). A very different
value points to a problem in the loss, the labels or the initialisation.

### 3. Overfit a single batch

Train on one batch of 8 to 32 examples, with regularisation and augmentation turned
off. The loss must go to nearly zero and the accuracy to 100 %. If it does not, the
model cannot even memorise: the bug is in the model, the loss or the optimisation, and
no amount of data will fix it. This is the single most useful test.

### 4. Train on the full data and read the curves

| Observation                                   | Usual meaning                              |
|-----------------------------------------------|--------------------------------------------|
| Loss flat at 0.693                            | not learning: learning rate, labels, a detached graph |
| Loss becomes `nan`                            | learning rate too high, `log(0)`, division by zero |
| Training loss falls, validation loss rises    | overfitting                                |
| Both losses high and close                    | underfitting: model too small or too regularised |
| Validation loss lower than training loss      | normal with dropout and augmentation ([chapter 16](16-training-loop.md)) |
| Validation loss very noisy                    | learning rate too high, validation set too small |
| Validation accuracy near 100 % at once        | suspect leakage or a shortcut ([chapter 18](18-generalization-and-data-leakage.md)) |

### 5. Examine the errors

List the misclassified examples, sorted by confidence, and listen to them. Confident
errors are the most instructive: they are either mislabelled data or a systematic
weakness of the model.

### 6. Test the invariances you expect

If the output should not depend on something, check that it does not. The checks made on
the mobile models:

- same clip, 100 times quieter → same probability;
- silence → probability close to 0;
- same clip, alone or inside a batch → same probability;
- PyTorch and ONNX Runtime → same probability.

The last three failed at first, and each failure exposed a real defect
([chapter 20](20-exporting-to-onnx.md)).

## Bugs that do not raise errors

- **Forgetting `model.eval()`.** Predictions vary with the batch; dropout stays active.
- **Wrong broadcasting.** Logits `(B, 1)` against labels `(B,)` in a hand-written loss
  give a `(B, B)` matrix and a loss that still decreases.
- **Labels and inputs shuffled separately.** The model learns nothing; loss stays at
  0.693.
- **Preprocessing mismatch** between training and inference: sampling rate, scaling,
  spectrogram parameters.
- **Augmenting the validation set**, or normalising with statistics that include it.
- **Evaluating on training data** because of a path mistake: a perfect score.
- **A stale file.** The preprocessed dataset or the checkpoint on disk comes from an
  older version of the code. Here the checkpoint stores the audio format it was trained
  with, and training refuses a mismatch.

## Tools

```python
print(x.shape, x.dtype, x.device, x.min().item(), x.max().item())   # the basic probe
torch.isnan(x).any()                                                # find NaN early
torch.autograd.set_detect_anomaly(True)                             # locate a NaN gradient
sum(p.numel() for p in model.parameters())                          # parameter count
```

Forward hooks (`module.register_forward_hook`) let you inspect the activations of any
layer without modifying the model.

## Habits

- Change one thing at a time, and keep a record of what was tried.
- Start from the simplest model that could work, then add complexity.
- Be more suspicious of a result that is too good than of one that is too bad.
- When a test passes, ask whether it could have failed.

## Exercises

1. Compute the loss of an untrained `mobile_tiny` on one validation batch. Is it close
   to 0.693?
2. Write a script that overfits a single batch of 16 clips with `audio_cnn` and reports
   the number of steps needed to reach 100 % accuracy.
3. Introduce one of the silent bugs above on purpose (for example, shuffle the labels)
   and observe the training curves.
4. Write the four invariance checks as automated tests with `pytest`.
5. Sort the validation clips by the confidence of `mobile_small` in its wrong answers
   and listen to the top five.

## Further reading

- Karpathy, "A Recipe for Training Neural Networks" (2019).
- Next: [23. Experiments to try](23-experiments.md).
