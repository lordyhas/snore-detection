# 17. Evaluation metrics

## Learning goals

- Build a confusion matrix and derive accuracy, precision, recall and F1 from it.
- Choose a metric and a decision threshold according to the cost of each error.
- Quantify the uncertainty of a score measured on a small set.

## Theory

### The confusion matrix

For a binary classifier, every prediction falls in one of four cells:

|                     | Predicted snoring   | Predicted not snoring |
|---------------------|---------------------|-----------------------|
| **Actually snoring**    | true positive (TP)  | false negative (FN)   |
| **Actually not snoring**| false positive (FP) | true negative (TN)    |

### Metrics

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \qquad
  \text{precision} = \frac{TP}{TP + FP} \qquad
  \text{recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

- **Precision**: when the model says "snoring", how often is it right?
- **Recall**: of all the real snores, how many did it find?
- **F1**: their harmonic mean, which is low as soon as one of the two is low.

### Accuracy can mislead

In a real night, snoring may occupy 5 % of the time. A model that never detects anything
then has 95 % accuracy and is useless: its recall is 0. Accuracy is only meaningful on
balanced data, like the 500 / 500 dataset of this project. Remember that the balance of
the dataset is not the balance of the application.

### The threshold

The model outputs a probability $p$; the decision is $p \geq \tau$. Raising $\tau$
makes the model more conservative: precision goes up, recall goes down. The value 0.5 is
a convention, not an optimum. Two curves summarise all thresholds at once:

- the **ROC curve** plots recall against the false positive rate $FP / (FP + TN)$; its
  area (AUC) is the probability that a random positive gets a higher score than a random
  negative;
- the **precision-recall curve**, more informative when positives are rare.

## In this project

`_run_epoch` in `src/train.py` computes the four metrics with scikit-learn, at threshold
0.5 (logit 0). Results on the 200 validation clips (100 per class):

| Model          | TP | FN | FP | TN  | Accuracy | Precision | Recall | F1    |
|----------------|----|----|----|-----|----------|-----------|--------|-------|
| `audio_cnn`    | 98 | 2  | 0  | 100 | 0.990    | 1.000     | 0.980  | 0.990 |
| `mobile_small` | 98 | 2  | 3  | 97  | 0.975    | 0.970     | 0.980  | 0.975 |

For `mobile_small`: precision $= 98 / 101 = 0.970$, recall $= 98 / 100 = 0.98$.

### How sure are we?

An accuracy measured on $n$ examples is an estimate. Its standard error is approximately

$$\text{SE} = \sqrt{\frac{a \, (1 - a)}{n}} .$$

For $a = 0.975$ and $n = 200$: $\text{SE} \approx 0.011$, so a 95 % confidence interval
is roughly $0.975 \pm 0.022$, that is $[0.953, 0.997]$. The interval of `audio_cnn`
overlaps it widely. **The validation set cannot rank these models.** Reporting 0.99
versus 0.975 as a difference would be reading noise.

The 11 recordings of `data/inference/` are an even smaller sample: 10 correct out of 11
is compatible with a true accuracy anywhere between roughly 60 % and 100 %.

### Which error matters in the application?

For a sleep tracker, a false positive records a snore that did not happen and inflates
the score of the night; a false negative misses one. Over a night with thousands of
windows, even a 1 % false positive rate produces dozens of phantom events. Precision at
a low false positive rate matters more than accuracy, and a rule such as "several
consecutive positive windows" can trade a little recall for much better precision.

## Pitfalls

- `zero_division=0` in the scikit-learn calls: when the model predicts no positive at
  all, precision is undefined and reported as 0. This happens in the first epochs.
- Metrics computed per batch and then averaged are wrong for precision, recall and F1.
  This project accumulates all the predictions of the epoch first.
- Tuning the threshold on the validation set and reporting the result on the same set
  is another form of optimism.

## Exercises

1. A model yields TP = 40, FN = 10, FP = 20, TN = 930. Compute the four metrics. Is
   accuracy a good summary?
2. Compute the confusion matrix of `mobile_tiny` on the validation set with
   `sklearn.metrics.confusion_matrix`.
3. Collect the validation probabilities and plot precision and recall as functions of
   the threshold. Which threshold gives no false positive?
4. How many validation clips would be needed for the 95 % interval of a 0.98 accuracy to
   be $\pm 0.005$?

## Further reading

- Fawcett, "An introduction to ROC analysis" (2006).
- Next: [18. Generalisation and data leakage](18-generalization-and-data-leakage.md).
