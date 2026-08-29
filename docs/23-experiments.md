# 23. Experiments to try

The exercises of the previous chapters are short. The projects below take from an
evening to a few weeks and are ordered by increasing difficulty. Each states a question,
a method and what to report, because an experiment without a question teaches little.

## How to run an experiment

1. **Write the hypothesis first.** "A grouped split will lower the validation accuracy."
2. **Change one thing.** Keep the seed, the data and everything else fixed.
3. **Repeat.** Several seeds; report mean and standard deviation
   ([chapter 19](19-reproducibility.md)).
4. **Keep a log**: date, commit, command, result, one line of conclusion. Negative
   results belong in it too.

## Level 1: understand what exists

### 1.1 Learning curves

*Question:* does the model overfit, and when?
*Method:* record training and validation loss per epoch and plot them with matplotlib,
for `audio_cnn` and `mobile_tiny`.
*Report:* the two figures and the epoch where the curves separate.

### 1.2 Ablation of the regularisers

*Question:* which regulariser matters on this dataset?
*Method:* remove one at a time: dropout, SpecAugment, weight decay.
*Report:* a table with the best validation loss for each variant, over three seeds.

### 1.3 Sensitivity to the learning rate

*Question:* how wide is the range of learning rates that work?
*Method:* train with $10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}$.
*Report:* best validation loss as a function of the learning rate, on a log axis.

## Level 2: improve the methodology

### 2.1 A real test set

*Question:* how optimistic is the validation score?
*Method:* split 70 / 15 / 15, select on validation, evaluate once on test.
*Report:* validation and test accuracy with confidence intervals
([chapter 17](17-evaluation-metrics.md)).

### 2.2 Cross-validation

*Question:* how much does the score depend on the split?
*Method:* 5-fold stratified cross-validation with
`sklearn.model_selection.StratifiedKFold`.
*Report:* accuracy per fold, mean and standard deviation.

### 2.3 Threshold selection

*Question:* which threshold suits a sleep tracker, where false positives are costly?
*Method:* plot the precision-recall curve; find the threshold giving at most one false
positive on validation.
*Report:* the curve, the threshold and the recall at that threshold.

### 2.4 Error analysis

*Question:* what do the errors have in common?
*Method:* listen to every misclassified validation clip and to the files of
`data/inference/` that fail.
*Report:* a short typology of the errors.

## Level 3: change the model

### 3.1 How small can it go?

*Question:* where does accuracy start to fall when the model shrinks?
*Method:* `SnoreMobileNet` with `width` 4, 8, 16, 32.
*Report:* accuracy against parameter count.

### 3.2 Residual network

*Question:* do skip connections help at this depth?
*Method:* build a model from `ResidualBlock`
([chapter 10](10-residual-and-dense-blocks.md)) with a comparable parameter count.
*Report:* comparison with `audio_cnn` over several seeds.

### 3.3 Front-end choices

*Question:* how much frequency detail is needed?
*Method:* vary `n_mels` (20, 40, 64) and the upper frequency of the filter bank
(2 kHz, 4 kHz, 8 kHz) in `LogMelFrontend`.
*Report:* a table. Relate the result to the spectrum of snoring.

### 3.4 Knowledge distillation

*Question:* can `audio_cnn` teach `mobile_tiny`?
*Method:* train the student on a mix of the true labels and the teacher's
probabilities.
*Report:* student accuracy with and without the teacher.

## Level 4: face the real problem

### 4.1 Realistic negatives

*Question:* does the model confuse snoring with ordinary bedroom sounds?
*Method:* gather recordings of breathing, bedding, fans, traffic, speech; cut them into
1-second clips; evaluate without retraining, then retrain with them.
*Report:* false positive rate per category, before and after.

### 4.2 Noise robustness

*Question:* how does accuracy degrade with background noise?
*Method:* mix validation clips with noise at signal-to-noise ratios from 20 dB down to
0 dB.
*Report:* accuracy against signal-to-noise ratio; then repeat after training with noise
augmentation.

### 4.3 Target-domain data

*Question:* how does the model behave on audio from the phone?
*Method:* export clips recorded by the SleepLab app, label a few hundred by ear,
evaluate, then fine-tune.
*Report:* accuracy on phone data before and after fine-tuning, with the split made by
night and not by clip ([chapter 18](18-generalization-and-data-leakage.md)).

### 4.4 Event-level evaluation

*Question:* how many snores per night are found, missed or invented?
*Method:* annotate the start and end of every snore in one hour of real recording; run
the sliding window with post-processing ([chapter 21](21-on-device-inference.md));
match detected events to annotated ones.
*Report:* event-level precision and recall, which is what the user of the app
experiences.

### 4.5 Quantisation

*Question:* what does 8-bit quantisation cost and save?
*Method:* post-training quantisation of the ONNX model.
*Report:* file size, latency on a phone and accuracy, before and after.

## A suggested order

2.1 first, because every later comparison needs a trustworthy measure. Then 2.4 and 4.1,
which tell you what is actually wrong. Model changes (level 3) come last: they are the
most enjoyable and usually the least useful until the data problem is addressed.
