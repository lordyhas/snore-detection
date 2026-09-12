# Part VI - Practice

The previous parts explain how things work. This one is about what to do when they do
not, and what to do next: a systematic way to debug a model that trains badly, then
a list of experiments, from an evening to a few weeks each, to deepen your
understanding and improve the project.

**Contents**

- [6.1 Debugging neural networks](#61-debugging-neural-networks)
- [6.2 Experiments to try](#62-experiments-to-try)
- [Key points](#key-points)

## 6.1 Debugging neural networks

### Learning goals

- Adopt a systematic procedure instead of changing things at random.
- Recognise the signature of the most common bugs.
- Use the sanity checks that catch them early.

### Why it is hard

Ordinary programs fail loudly. Neural networks fail silently: a model with a bug in its
data pipeline still trains, still produces a decreasing loss, and still reaches a
plausible score, only a worse one than it should, or a good one for a bad reason. The
only defence is to verify each component independently before trusting the whole.

### A procedure

#### 1. Look at the data

Before any model: plot spectrograms, listen to clips of each class, print shapes, value
ranges and class counts. In this project a glance at the durations would have revealed
that the clips last one second while the code assumed three.

#### 2. Check the loss at initialisation

A freshly initialised binary classifier should output probabilities near 0.5, hence a
loss near $\log 2 \approx 0.693$ ([section 3.1](03-training.md#31-loss-functions)). A very different
value points to a problem in the loss, the labels or the initialisation.

#### 3. Overfit a single batch

Train on one batch of 8 to 32 examples, with regularisation and augmentation turned
off. The loss must go to nearly zero and the accuracy to 100 %. If it does not, the
model cannot even memorise: the bug is in the model, the loss or the optimisation, and
no amount of data will fix it. This is the single most useful test.

#### 4. Train on the full data and read the curves

| Observation                                   | Usual meaning                              |
|-----------------------------------------------|--------------------------------------------|
| Loss flat at 0.693                            | not learning: learning rate, labels, a detached graph |
| Loss becomes `nan`                            | learning rate too high, `log(0)`, division by zero |
| Training loss falls, validation loss rises    | overfitting                                |
| Both losses high and close                    | underfitting: model too small or too regularised |
| Validation loss lower than training loss      | normal with dropout and augmentation ([section 3.6](03-training.md#36-anatomy-of-the-training-loop)) |
| Validation loss very noisy                    | learning rate too high, validation set too small |
| Validation accuracy near 100 % at once        | suspect leakage or a shortcut ([section 4.2](04-evaluation.md#42-generalisation-and-data-leakage)) |

#### 5. Examine the errors

List the misclassified examples, sorted by confidence, and listen to them. Confident
errors are the most instructive: they are either mislabelled data or a systematic
weakness of the model.

#### 6. Test the invariances you expect

If the output should not depend on something, check that it does not. The checks made on
the mobile models:

- same clip, 100 times quieter → same probability;
- silence → probability close to 0;
- same clip, alone or inside a batch → same probability;
- PyTorch and ONNX Runtime → same probability.

The last three failed at first, and each failure exposed a real defect
([section 5.1](05-deployment.md#51-exporting-to-onnx)).

### Automated sanity checks

Steps 2, 3 and 6 of the procedure do not need a full training, so they can be run in a
few seconds on every model. This is what `src/sanity_check.py` does:

```bash
python -m src.sanity_check
```

Each check tests one hypothesis about a model built by `build_model`. A hypothesis that
does not hold points to a specific kind of bug.

| Check               | Hypothesis                                         | If it fails, suspect                         |
|---------------------|----------------------------------------------------|----------------------------------------------|
| output shape        | one finite logit per example, for any input length | a layer tied to a fixed size, a `NaN`        |
| initial loss        | untrained loss close to $\log 2$                   | the initialisation or the scale of the output |
| gradient flow       | every parameter gets a non-zero gradient           | a `detach`, a frozen or unused layer         |
| numerical gradient  | backpropagation matches finite differences         | a non-differentiable or hand-written operation |
| overfit small batch | 16 examples can be memorised                       | the model, the loss or the optimisation      |
| eval mode           | deterministic, independent of the batch            | a layer that ignores `model.eval()`          |
| gain invariance     | waveform models ignore the recording level         | the normalisation of the front-end           |
| silence             | all-zero audio gives finite loss and gradients     | a logarithm or a division without floor      |

The first six apply to every model; the last two only to the models that take a raw
waveform, since they test the spectrogram computed inside the model.

### Bugs that do not raise errors

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

### Tools

```python
print(x.shape, x.dtype, x.device, x.min().item(), x.max().item())   # the basic probe
torch.isnan(x).any()                                                # find NaN early
torch.autograd.set_detect_anomaly(True)                             # locate a NaN gradient
sum(p.numel() for p in model.parameters())                          # parameter count
```

Forward hooks (`module.register_forward_hook`) let you inspect the activations of any
layer without modifying the model.

### Habits

- Change one thing at a time, and keep a record of what was tried.
- Start from the simplest model that could work, then add complexity.
- Be more suspicious of a result that is too good than of one that is too bad.
- When a test passes, ask whether it could have failed.

### Exercises

1. Compute the loss of an untrained `mobile_tiny` on one validation batch. Is it close
   to 0.693?
2. Write a script that overfits a single batch of 16 clips with `audio_cnn` and reports
   the number of steps needed to reach 100 % accuracy.
3. Introduce one of the silent bugs above on purpose (for example, shuffle the labels)
   and observe the training curves.
4. Write the four invariance checks as automated tests with `pytest`.
5. Sort the validation clips by the confidence of `mobile_small` in its wrong answers
   and listen to the top five.

### Further reading

- Karpathy, "A Recipe for Training Neural Networks" (2019).

## 6.2 Experiments to try

The exercises of the previous sections are short. The projects below take from an
evening to a few weeks and are ordered by increasing difficulty. Each states a question,
a method and what to report, because an experiment without a question teaches little.

### How to run an experiment

1. **Write the hypothesis first.** "A grouped split will lower the validation accuracy."
2. **Change one thing.** Keep the seed, the data and everything else fixed.
3. **Repeat.** Several seeds; report mean and standard deviation
   ([section 4.3](04-evaluation.md#43-reproducibility)).
4. **Keep a log**: date, commit, command, result, one line of conclusion. Negative
   results belong in it too.

### Level 1: understand what exists

#### Experiment 1.1: Learning curves

*Question:* does the model overfit, and when?
*Method:* record training and validation loss per epoch and plot them with matplotlib,
for `audio_cnn` and `mobile_tiny`.
*Report:* the two figures and the epoch where the curves separate.

#### Experiment 1.2: Ablation of the regularisers

*Question:* which regulariser matters on this dataset?
*Method:* remove one at a time: dropout, SpecAugment, weight decay.
*Report:* a table with the best validation loss for each variant, over three seeds.

#### Experiment 1.3: Sensitivity to the learning rate

*Question:* how wide is the range of learning rates that work?
*Method:* train with $10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}$.
*Report:* best validation loss as a function of the learning rate, on a log axis.

### Level 2: improve the methodology

#### Experiment 2.1: A real test set

*Question:* how optimistic is the validation score?
*Method:* split 70 / 15 / 15, select on validation, evaluate once on test.
*Report:* validation and test accuracy with confidence intervals
([section 4.1](04-evaluation.md#41-evaluation-metrics)).

#### Experiment 2.2: Cross-validation

*Question:* how much does the score depend on the split?
*Method:* 5-fold stratified cross-validation with
`sklearn.model_selection.StratifiedKFold`.
*Report:* accuracy per fold, mean and standard deviation.

#### Experiment 2.3: Threshold selection

*Question:* which threshold suits a sleep tracker, where false positives are costly?
*Method:* plot the precision-recall curve; find the threshold giving at most one false
positive on validation.
*Report:* the curve, the threshold and the recall at that threshold.

#### Experiment 2.4: Error analysis

*Question:* what do the errors have in common?
*Method:* listen to every misclassified validation clip and to the files of
`data/inference/` that fail.
*Report:* a short typology of the errors.

### Level 3: change the model

#### Experiment 3.1: How small can it go?

*Question:* where does accuracy start to fall when the model shrinks?
*Method:* `SnoreMobileNet` with `width` 4, 8, 16, 32.
*Report:* accuracy against parameter count.

#### Experiment 3.2: Residual network

*Question:* do skip connections help at this depth?
*Method:* build a model from `ResidualBlock`
([section 2.5](02-convolutional-networks.md#25-residual-and-dense-blocks)) with a comparable parameter count.
*Report:* comparison with `audio_cnn` over several seeds.

#### Experiment 3.3: Front-end choices

*Question:* how much frequency detail is needed?
*Method:* vary `n_mels` (20, 40, 64) and the upper frequency of the filter bank
(2 kHz, 4 kHz, 8 kHz) in `LogMelFrontend`.
*Report:* a table. Relate the result to the spectrum of snoring.

#### Experiment 3.4: Knowledge distillation

*Question:* can `audio_cnn` teach `mobile_tiny`?
*Method:* train the student on a mix of the true labels and the teacher's
probabilities.
*Report:* student accuracy with and without the teacher.

### Level 4: face the real problem

#### Experiment 4.1: Realistic negatives

*Question:* does the model confuse snoring with ordinary bedroom sounds?
*Method:* gather recordings of breathing, bedding, fans, traffic, speech; cut them into
1-second clips; evaluate without retraining, then retrain with them.
*Report:* false positive rate per category, before and after.

#### Experiment 4.2: Noise robustness

*Question:* how does accuracy degrade with background noise?
*Method:* mix validation clips with noise at signal-to-noise ratios from 20 dB down to
0 dB.
*Report:* accuracy against signal-to-noise ratio; then repeat after training with noise
augmentation.

#### Experiment 4.3: Target-domain data

*Question:* how does the model behave on audio from the phone?
*Method:* export clips recorded by the SleepLab app, label a few hundred by ear,
evaluate, then fine-tune.
*Report:* accuracy on phone data before and after fine-tuning, with the split made by
night and not by clip ([section 4.2](04-evaluation.md#42-generalisation-and-data-leakage)).

#### Experiment 4.4: Event-level evaluation

*Question:* how many snores per night are found, missed or invented?
*Method:* annotate the start and end of every snore in one hour of real recording; run
the sliding window with post-processing ([section 5.2](05-deployment.md#52-on-device-inference));
match detected events to annotated ones.
*Report:* event-level precision and recall, which is what the user of the app
experiences.

#### Experiment 4.5: Quantisation

*Question:* what does 8-bit quantisation cost and save?
*Method:* post-training quantisation of the ONNX model.
*Report:* file size, latency on a phone and accuracy, before and after.

### A suggested order

Experiment 2.1 first, because every later comparison needs a trustworthy measure. Then
experiments 2.4 and 4.1,
which tell you what is actually wrong. Model changes (level 3) come last: they are the
most enjoyable and usually the least useful until the data problem is addressed.

## Key points

- Neural networks fail silently; verify each component on its own before trusting the whole.
- Three checks catch most bugs: look at the data, check the initial loss, overfit a single batch.
- Test the invariances you expect, and be more suspicious of a result that is too good than of one that is too bad.
- An experiment starts with a written hypothesis, changes one thing, and is repeated over several seeds.
- Fix the measurement (a real test set, target-domain data) before changing the model.

---

Previous: [Part V - Deployment](05-deployment.md) | [Contents](README.md)
