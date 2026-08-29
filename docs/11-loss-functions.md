# 11. Loss functions

## Learning goals

- Derive binary cross-entropy from maximum likelihood.
- Explain what a logit is and why the model does not end with a sigmoid.
- Understand the numerical trick inside `BCEWithLogitsLoss`.

## Theory

### From score to probability

The network outputs a real number $z$, the **logit**. The sigmoid maps it to a
probability:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}} .$$

$z = 0$ gives $p = 0.5$; large positive $z$ gives $p \to 1$. Conversely
$z = \log \frac{p}{1 - p}$, the log-odds, hence the name.

### Maximum likelihood

We model the label $y \in \{0, 1\}$ as a Bernoulli variable with parameter $p$:
$P(y \mid x) = p^y (1 - p)^{1 - y}$. Maximising the likelihood of the training set is the
same as minimising the average negative log-likelihood, which is the **binary
cross-entropy**:

$$\mathcal{L}(z, y) = -\big[\, y \log p + (1 - y) \log(1 - p) \,\big] .$$

A confident and correct prediction costs almost nothing; a confident and wrong one costs
a lot ($p = 0.01$ with $y = 1$ costs $-\log 0.01 = 4.6$).

A useful reference value: a model that always answers $p = 0.5$ has a loss of
$\log 2 \approx 0.693$. A loss stuck around 0.69 means the model has learned nothing.

### The gradient

Differentiating with respect to the logit gives a remarkably simple result:

$$\frac{\partial \mathcal{L}}{\partial z} = \sigma(z) - y .$$

The gradient is the prediction error. It does not vanish when the model is confidently
wrong, which is exactly why cross-entropy is preferred to the squared error for
classification.

### Numerical stability

Computing `log(sigmoid(z))` literally fails for large $|z|$: in `float32`,
`sigmoid(40)` rounds to exactly 1, so $\log(1 - p) = \log 0 = -\infty$. The loss can be
rewritten in a form that never overflows:

$$\mathcal{L}(z, y) = \max(z, 0) - z\,y + \log\!\big(1 + e^{-|z|}\big) .$$

This is what `torch.nn.BCEWithLogitsLoss` computes. It is the reason why the model
returns logits and the sigmoid lives in the loss.

## In this project

```python
criterion = torch.nn.BCEWithLogitsLoss()          # src/train.py
...
logits = model(specs)                             # (batch, 1)
loss = criterion(logits, labels)                  # labels: (batch, 1), float
predictions = logits.detach().view(-1) >= 0       # same as sigmoid(logit) >= 0.5
```

The threshold on the logit is 0 because $\sigma(0) = 0.5$: no sigmoid is needed to
decide. For deployment, `src/export.py` wraps the model so that the ONNX file outputs a
probability directly:

```python
def forward(self, waveform):
    return torch.sigmoid(self.model(waveform))
```

The untrained checkpoint that used to be in this repository had a validation loss of
0.677, right next to $\log 2$: the number alone revealed that it had not learned.

## Related losses

- **Multi-class** problems use `nn.CrossEntropyLoss`, the same idea with a softmax over
  several logits.
- **Class imbalance**: `BCEWithLogitsLoss(pos_weight=...)` multiplies the loss of
  positive examples. Not needed here, as the classes are balanced (500 / 500), but real
  nights contain far more non-snoring than snoring.

## Pitfalls

- `nn.BCELoss` expects probabilities, `nn.BCEWithLogitsLoss` expects logits. Applying a
  sigmoid *and* using the second one trains, badly, with a squashed signal.
- Targets must be `float` and have the same shape as the logits.

## Exercises

1. Compute by hand the loss for $z = 2$, $y = 1$ and for $z = 2$, $y = 0$.
2. Prove that $\partial \mathcal{L} / \partial z = \sigma(z) - y$, using
   $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
3. In PyTorch, compare `torch.log(torch.sigmoid(torch.tensor(-200.)))` with
   `torch.nn.functional.logsigmoid(torch.tensor(-200.))`.
4. Check numerically that the stable formula equals the definition for $z = 3$, $y = 0$.

## Further reading

- Goodfellow, Bengio and Courville, *Deep Learning*, sections 5.5 and 6.2.
- Next: [12. Autograd and backpropagation](12-autograd-and-backpropagation.md).
