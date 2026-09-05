# Part IV - Evaluation and scientific method

Training produces a number, such as "99 % accuracy". This part is about what that
number means and how far it can be trusted. It is the least technical part and the
one that matters most in practice: most disappointing machine learning projects fail
here, with a correct model evaluated the wrong way.

**Contents**

- [4.1 Evaluation metrics](#41-evaluation-metrics)
- [4.2 Generalisation and data leakage](#42-generalisation-and-data-leakage)
- [4.3 Reproducibility](#43-reproducibility)
- [Key points](#key-points)

## 4.1 Evaluation metrics

### Learning goals

- Build a confusion matrix and derive accuracy, precision, recall and F1 from it.
- Choose a metric and a decision threshold according to the cost of each error.
- Quantify the uncertainty of a score measured on a small set.

### Theory

#### The confusion matrix

For a binary classifier, every prediction falls in one of four cells:

|                     | Predicted snoring   | Predicted not snoring |
|---------------------|---------------------|-----------------------|
| **Actually snoring**    | true positive (TP)  | false negative (FN)   |
| **Actually not snoring**| false positive (FP) | true negative (TN)    |

#### Metrics

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \qquad
  \text{precision} = \frac{TP}{TP + FP} \qquad
  \text{recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

- **Precision**: when the model says "snoring", how often is it right?
- **Recall**: of all the real snores, how many did it find?
- **F1**: their harmonic mean, which is low as soon as one of the two is low.

#### Accuracy can mislead

In a real night, snoring may occupy 5 % of the time. A model that never detects anything
then has 95 % accuracy and is useless: its recall is 0. Accuracy is only meaningful on
balanced data, like the 500 / 500 dataset of this project. Remember that the balance of
the dataset is not the balance of the application.

#### The threshold

The model outputs a probability $p$; the decision is $p \geq \tau$. Raising $\tau$
makes the model more conservative: precision goes up, recall goes down. The value 0.5 is
a convention, not an optimum. Two curves summarise all thresholds at once:

- the **ROC curve** plots recall against the false positive rate $FP / (FP + TN)$; its
  area (AUC) is the probability that a random positive gets a higher score than a random
  negative;
- the **precision-recall curve**, more informative when positives are rare.

### In this project

`_run_epoch` in `src/train.py` computes the four metrics with scikit-learn, at threshold
0.5 (logit 0). Results on the 200 validation clips (100 per class):

| Model          | TP | FN | FP | TN  | Accuracy | Precision | Recall | F1    |
|----------------|----|----|----|-----|----------|-----------|--------|-------|
| `audio_cnn`    | 98 | 2  | 0  | 100 | 0.990    | 1.000     | 0.980  | 0.990 |
| `mobile_small` | 98 | 2  | 3  | 97  | 0.975    | 0.970     | 0.980  | 0.975 |

For `mobile_small`: precision $= 98 / 101 = 0.970$, recall $= 98 / 100 = 0.98$.

#### How sure are we?

An accuracy measured on $n$ examples is an estimate. Its standard error is approximately

$$\text{SE} = \sqrt{\frac{a \, (1 - a)}{n}} .$$

For $a = 0.975$ and $n = 200$: $\text{SE} \approx 0.011$, so a 95 % confidence interval
is roughly $0.975 \pm 0.022$, that is $[0.953, 0.997]$. The interval of `audio_cnn`
overlaps it widely. **The validation set cannot rank these models.** Reporting 0.99
versus 0.975 as a difference would be reading noise.

The 11 recordings of `data/inference/` are an even smaller sample: 10 correct out of 11
is compatible with a true accuracy anywhere between roughly 60 % and 100 %.

#### Which error matters in the application?

For a sleep tracker, a false positive records a snore that did not happen and inflates
the score of the night; a false negative misses one. Over a night with thousands of
windows, even a 1 % false positive rate produces dozens of phantom events. Precision at
a low false positive rate matters more than accuracy, and a rule such as "several
consecutive positive windows" can trade a little recall for much better precision.

### Pitfalls

- `zero_division=0` in the scikit-learn calls: when the model predicts no positive at
  all, precision is undefined and reported as 0. This happens in the first epochs.
- Metrics computed per batch and then averaged are wrong for precision, recall and F1.
  This project accumulates all the predictions of the epoch first.
- Tuning the threshold on the validation set and reporting the result on the same set
  is another form of optimism.

### Exercises

1. A model yields TP = 40, FN = 10, FP = 20, TN = 930. Compute the four metrics. Is
   accuracy a good summary?
2. Compute the confusion matrix of `mobile_tiny` on the validation set with
   `sklearn.metrics.confusion_matrix`.
3. Collect the validation probabilities and plot precision and recall as functions of
   the threshold. Which threshold gives no false positive?
4. How many validation clips would be needed for the 95 % interval of a 0.98 accuracy to
   be $\pm 0.005$?

### Further reading

- Fawcett, "An introduction to ROC analysis" (2006).

## 4.2 Generalisation and data leakage

### Learning goals

- Distinguish training, validation and test sets and the role of each.
- Recognise data leakage and shortcut learning.
- Assess honestly what the results of this project do and do not show.

### Theory

#### What we really want

The goal is never a low error on the data we have. It is a low error on the data the
model will meet later: **generalisation**. All evaluation methodology exists to estimate
that future error without fooling ourselves.

#### Three sets

| Set        | Used for                                             | Seen how often |
|------------|------------------------------------------------------|----------------|
| Training   | fitting the weights                                  | every epoch    |
| Validation | choosing hyperparameters, early stopping, model selection | every epoch |
| Test       | the final, unbiased estimate                         | once           |

Every decision taken by looking at a set leaks information about it into the model.
Once the test set has guided a decision, it has become a validation set.

#### The i.i.d. assumption

A random split estimates future performance only if future data resemble the dataset:
*independent and identically distributed* samples from the same source. Two things break
this assumption in practice.

**Leakage between splits.** If near-duplicates of training examples sit in the
validation set, the model is rewarded for memorising. Typical causes: several clips cut
from one recording, the same speaker or patient on both sides, augmented copies split
after augmentation. The remedy is a **grouped split**: all clips from one source go to
the same side.

**Distribution shift.** The deployment data differ from the collected data: another
microphone, another room, other background noises. No split of the original dataset can
measure this; only data from the target conditions can.

#### Shortcut learning

A network minimises the loss by any available means. If some accidental feature of the
dataset predicts the label, the network will use it, because it is easier to learn than
the real concept (Geirhos et al., 2020). The classic example is a classifier that
recognises cows by the grass in the background.

### In this project

#### What is in place

`train_test_split(..., stratify=labels, random_state=RANDOM_SEED)` in
`src/data/preprocess.py` keeps the 50 / 50 class balance in both splits and makes the
split reproducible.

#### What is missing

- **No test set.** The validation split is used for scheduling, stopping and selecting
  the model. The reported 0.98 to 0.99 are optimistic.
- **No grouping.** The dataset is a flat list of 1-second files. If several of them were
  cut from the same recording, which is plausible but not documented by the source,
  related clips are on both sides of the split.

#### A shortcut found in the data

Ten clips of the dataset are *exactly* zero, digital silence, and all ten are labelled
non-snoring. This is harmless in itself, since silence is indeed not snoring, but it
revealed a problem. With the first version of the mobile front-end, the output for those
clips depended on rounding errors of the order of $10^{-7}$
([section 5.1](05-deployment.md#51-exporting-to-onnx)). More generally, files produced by different
tools carry different fingerprints (noise floor, filtering, clipping), and a network can
learn "which tool made this file" instead of "is this a snore".

#### Evidence about generalisation

The only data from outside the dataset are the 11 files of `data/inference/`:

| Model          | Validation accuracy | Extra recordings |
|----------------|---------------------|------------------|
| `mobile_tiny`  | 0.98                | 10 / 11          |
| `mobile_small` | 0.975               | 8 / 11           |

Two models that are equivalent on validation differ on outside data. Eleven files prove
nothing ([section 4.1](#41-evaluation-metrics)), but the direction is the expected one:
performance drops outside the dataset.

#### What the results show

They show that a small CNN separates the two classes *of this dataset* almost perfectly.
They do not show that it detects snoring on a phone at night. The non-snoring class
consists of sounds such as a baby crying or a door opening, whereas the real negatives
in a bedroom are breathing, bedding noise, traffic, a fan, speech. That gap is the main
open problem of the project.

### How to do better

1. **Hold out a test set** before any further experiment, and evaluate on it once.
2. **Collect target-domain data**: real nights recorded by the app, labelled by ear. A
   few hundred clips from the phone are worth more than thousands from elsewhere.
3. **Split by source**: by night and by person, never by clip.
4. **Look at the errors.** Listening to the misclassified clips is the fastest way to
   understand what the model has really learned.

### Exercises

1. Modify `preprocess_data` to produce three splits, 70 / 15 / 15, stratified.
2. Train five times with five values of `RANDOM_SEED` for the split. Report the mean and
   standard deviation of the validation accuracy. How does the spread compare with the
   difference between models?
3. Remove the ten all-zero clips from the dataset and retrain. Does the behaviour on
   silence change?
4. Use `sklearn.model_selection.GroupShuffleSplit` on an imaginary `recording_id` array
   and explain what it guarantees.

### Further reading

- Geirhos et al. (2020), "Shortcut learning in deep neural networks".
- Kapoor and Narayanan (2023), "Leakage and the reproducibility crisis in
  machine-learning-based science".

## 4.3 Reproducibility

### Learning goals

- List the sources of randomness in a training run.
- Seed them, and know what seeding does not guarantee.
- Report results in a way that others, and your future self, can trust.

### Theory

#### Where randomness enters

| Source                               | In this project                              |
|--------------------------------------|----------------------------------------------|
| Train / validation split             | `train_test_split(random_state=...)`         |
| Weight initialisation                | when the model is built                      |
| Order of the batches                 | `DataLoader(shuffle=True)`                   |
| Dropout masks                        | at every training step                       |
| Augmentation (masks, time shift)     | at every access to a training example        |
| Non-deterministic GPU algorithms     | some cuDNN convolution implementations       |

Two runs of the same code therefore give two different models. The difference is not a
bug; it is a property of the method that must be measured and controlled.

#### Pseudo-random generators and seeds

A pseudo-random generator is a deterministic function of its internal state. Setting the
**seed** sets the state, hence the entire sequence that follows. Python, NumPy and
PyTorch each have their own generator, and PyTorch has one per device.

### In this project

```python
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
...
generator = torch.Generator().manual_seed(RANDOM_SEED)
train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, ...)
```

`RANDOM_SEED = 42` is defined once, in `src/config.py`, and reused for the split. A
single constant avoids the situation where two parts of the code are seeded differently.

The seeds are set *before* the model is built, because building the model consumes
random numbers for the initial weights.

### What seeding does not guarantee

- **Across hardware.** CPU and GPU do not perform floating-point operations in the same
  order, and the results differ in the last bits. Those differences are then amplified
  by training.
- **On a GPU.** Some cuDNN algorithms are not deterministic. Full determinism requires

  ```python
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.benchmark = False
  ```

  at some cost in speed. This project does not set them, so two GPU runs can differ
  slightly.
- **Across versions.** A new release of PyTorch may change an initialisation or a
  kernel. This is why `requirements.txt` records minimum versions, and why a strict
  reproduction needs exact ones (`pip freeze`).
- **With several workers.** The order in which workers deliver batches and their
  individual random states must be controlled too (`worker_init_fn`).

### Floating point is part of the story

Floating-point addition is not associative: $(a + b) + c$ may differ from $a + (b + c)$
in the last bit. Two correct implementations of the same formula can therefore disagree
slightly. Normally this is invisible. It becomes visible when a computation *amplifies*
small differences, which happened in this project between PyTorch and ONNX Runtime
([section 5.1](05-deployment.md#51-exporting-to-onnx)). A numerically robust model is one whose output
does not depend on the last bits.

### Reporting results

A single run with a single seed is an anecdote. Good practice:

1. Fix the data split and vary the training seed over several runs (five is a common
   minimum). Report mean and standard deviation.
2. State what was selected on which set
   ([section 4.2](#42-generalisation-and-data-leakage)).
3. Record the exact command, the commit and the package versions with every result.
4. Do not choose the seed that gives the best number. The seed is not a hyperparameter.

The results quoted in this documentation come from single runs. They are indications,
and the tables say so.

### Exercises

1. Run `python -m src.train --model mobile_tiny` twice on the CPU. Are the logs
   identical? Then twice on the GPU.
2. Add a `--seed` option to `src/train.py` that only affects training, not the split.
   Run five seeds and report mean and standard deviation of the validation accuracy.
3. Evaluate `0.1 + 0.2 == 0.3` and
   `(0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3)` in Python. Explain.
4. Move `torch.manual_seed` *after* `build_model`. What changes between two runs?

### Further reading

- PyTorch documentation, "Reproducibility".
- Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks" (2021).
