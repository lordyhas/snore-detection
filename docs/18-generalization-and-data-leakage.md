# 18. Generalisation and data leakage

## Learning goals

- Distinguish training, validation and test sets and the role of each.
- Recognise data leakage and shortcut learning.
- Assess honestly what the results of this project do and do not show.

## Theory

### What we really want

The goal is never a low error on the data we have. It is a low error on the data the
model will meet later: **generalisation**. All evaluation methodology exists to estimate
that future error without fooling ourselves.

### Three sets

| Set        | Used for                                             | Seen how often |
|------------|------------------------------------------------------|----------------|
| Training   | fitting the weights                                  | every epoch    |
| Validation | choosing hyperparameters, early stopping, model selection | every epoch |
| Test       | the final, unbiased estimate                         | once           |

Every decision taken by looking at a set leaks information about it into the model.
Once the test set has guided a decision, it has become a validation set.

### The i.i.d. assumption

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

### Shortcut learning

A network minimises the loss by any available means. If some accidental feature of the
dataset predicts the label, the network will use it, because it is easier to learn than
the real concept (Geirhos et al., 2020). The classic example is a classifier that
recognises cows by the grass in the background.

## In this project

### What is in place

`train_test_split(..., stratify=labels, random_state=RANDOM_SEED)` in
`src/data/preprocess.py` keeps the 50 / 50 class balance in both splits and makes the
split reproducible.

### What is missing

- **No test set.** The validation split is used for scheduling, stopping and selecting
  the model. The reported 0.98 to 0.99 are optimistic.
- **No grouping.** The dataset is a flat list of 1-second files. If several of them were
  cut from the same recording, which is plausible but not documented by the source,
  related clips are on both sides of the split.

### A shortcut found in the data

Ten clips of the dataset are *exactly* zero, digital silence, and all ten are labelled
non-snoring. This is harmless in itself, since silence is indeed not snoring, but it
revealed a problem. With the first version of the mobile front-end, the output for those
clips depended on rounding errors of the order of $10^{-7}$
([chapter 20](20-exporting-to-onnx.md)). More generally, files produced by different
tools carry different fingerprints (noise floor, filtering, clipping), and a network can
learn "which tool made this file" instead of "is this a snore".

### Evidence about generalisation

The only data from outside the dataset are the 11 files of `data/inference/`:

| Model          | Validation accuracy | Extra recordings |
|----------------|---------------------|------------------|
| `mobile_tiny`  | 0.98                | 10 / 11          |
| `mobile_small` | 0.975               | 8 / 11           |

Two models that are equivalent on validation differ on outside data. Eleven files prove
nothing ([chapter 17](17-evaluation-metrics.md)), but the direction is the expected one:
performance drops outside the dataset.

### What the results show

They show that a small CNN separates the two classes *of this dataset* almost perfectly.
They do not show that it detects snoring on a phone at night. The non-snoring class
consists of sounds such as a baby crying or a door opening, whereas the real negatives
in a bedroom are breathing, bedding noise, traffic, a fan, speech. That gap is the main
open problem of the project.

## How to do better

1. **Hold out a test set** before any further experiment, and evaluate on it once.
2. **Collect target-domain data**: real nights recorded by the app, labelled by ear. A
   few hundred clips from the phone are worth more than thousands from elsewhere.
3. **Split by source**: by night and by person, never by clip.
4. **Look at the errors.** Listening to the misclassified clips is the fastest way to
   understand what the model has really learned.

## Exercises

1. Modify `preprocess_data` to produce three splits, 70 / 15 / 15, stratified.
2. Train five times with five values of `RANDOM_SEED` for the split. Report the mean and
   standard deviation of the validation accuracy. How does the spread compare with the
   difference between models?
3. Remove the ten all-zero clips from the dataset and retrain. Does the behaviour on
   silence change?
4. Use `sklearn.model_selection.GroupShuffleSplit` on an imaginary `recording_id` array
   and explain what it guarantees.

## Further reading

- Geirhos et al. (2020), "Shortcut learning in deep neural networks".
- Kapoor and Narayanan (2023), "Leakage and the reproducibility crisis in
  machine-learning-based science".
- Next: [19. Reproducibility](19-reproducibility.md).
