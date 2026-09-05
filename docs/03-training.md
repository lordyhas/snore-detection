# Part III - Training

A model is a function with adjustable parameters; training is the search for good
values. This part builds that search piece by piece: what is minimised, how its
gradient is obtained, how the parameters are moved, when to slow down and when to
stop, and how to keep the model from memorising. The last section puts the pieces
back together by reading `src/train.py` from top to bottom.

**Contents**

- [3.1 Loss functions](#31-loss-functions)
- [3.2 Autograd and backpropagation](#32-autograd-and-backpropagation)
- [3.3 Optimisers](#33-optimisers)
- [3.4 Learning rate schedules and early stopping](#34-learning-rate-schedules-and-early-stopping)
- [3.5 Regularisation and augmentation](#35-regularisation-and-augmentation)
- [3.6 Anatomy of the training loop](#36-anatomy-of-the-training-loop)
- [Key points](#key-points)

## 3.1 Loss functions

### Learning goals

- Derive binary cross-entropy from maximum likelihood.
- Explain what a logit is and why the model does not end with a sigmoid.
- Understand the numerical trick inside `BCEWithLogitsLoss`.

### Theory

#### From score to probability

The network outputs a real number $z$, the **logit**. The sigmoid maps it to a
probability:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}} .$$

$z = 0$ gives $p = 0.5$; large positive $z$ gives $p \to 1$. Conversely
$z = \log \frac{p}{1 - p}$, the log-odds, hence the name.

#### Maximum likelihood

We model the label $y \in \{0, 1\}$ as a Bernoulli variable with parameter $p$:
$P(y \mid x) = p^y (1 - p)^{1 - y}$. Maximising the likelihood of the training set is the
same as minimising the average negative log-likelihood, which is the **binary
cross-entropy**:

$$\mathcal{L}(z, y) = -\big[\, y \log p + (1 - y) \log(1 - p) \,\big] .$$

A confident and correct prediction costs almost nothing; a confident and wrong one costs
a lot ($p = 0.01$ with $y = 1$ costs $-\log 0.01 = 4.6$).

A useful reference value: a model that always answers $p = 0.5$ has a loss of
$\log 2 \approx 0.693$. A loss stuck around 0.69 means the model has learned nothing.

#### The gradient

Differentiating with respect to the logit gives a remarkably simple result:

$$\frac{\partial \mathcal{L}}{\partial z} = \sigma(z) - y .$$

The gradient is the prediction error. It does not vanish when the model is confidently
wrong, which is exactly why cross-entropy is preferred to the squared error for
classification.

#### Numerical stability

Computing `log(sigmoid(z))` literally fails for large $|z|$: in `float32`,
`sigmoid(40)` rounds to exactly 1, so $\log(1 - p) = \log 0 = -\infty$. The loss can be
rewritten in a form that never overflows:

$$\mathcal{L}(z, y) = \max(z, 0) - z\,y + \log\!\big(1 + e^{-|z|}\big) .$$

This is what `torch.nn.BCEWithLogitsLoss` computes. It is the reason why the model
returns logits and the sigmoid lives in the loss.

### In this project

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

### Related losses

- **Multi-class** problems use `nn.CrossEntropyLoss`, the same idea with a softmax over
  several logits.
- **Class imbalance**: `BCEWithLogitsLoss(pos_weight=...)` multiplies the loss of
  positive examples. Not needed here, as the classes are balanced (500 / 500), but real
  nights contain far more non-snoring than snoring.

### Pitfalls

- `nn.BCELoss` expects probabilities, `nn.BCEWithLogitsLoss` expects logits. Applying a
  sigmoid *and* using the second one trains, badly, with a squashed signal.
- Targets must be `float` and have the same shape as the logits.

### Exercises

1. Compute by hand the loss for $z = 2$, $y = 1$ and for $z = 2$, $y = 0$.
2. Prove that $\partial \mathcal{L} / \partial z = \sigma(z) - y$, using
   $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
3. In PyTorch, compare `torch.log(torch.sigmoid(torch.tensor(-200.)))` with
   `torch.nn.functional.logsigmoid(torch.tensor(-200.))`.
4. Check numerically that the stable formula equals the definition for $z = 3$, $y = 0$.

### Further reading

- Goodfellow, Bengio and Courville, *Deep Learning*, sections 5.5 and 6.2.

## 3.2 Autograd and backpropagation

### Learning goals

- State what backpropagation computes and why it is efficient.
- Use `requires_grad`, `backward`, `detach` and `no_grad` correctly.
- Explain why gradients must be reset at every step.

### Theory

Training means adjusting every parameter $\theta$ in the direction that lowers the loss.
We need $\partial \mathcal{L} / \partial \theta$ for more than a million parameters.

A network is a composition of functions, $\mathcal{L} = f_n(\dots f_2(f_1(x)))$, so the
derivative is given by the **chain rule**: a product of local derivatives. The key
observation of **backpropagation** is the order of the products. Starting from the
output, which is a scalar, and moving backwards, every intermediate quantity is a vector
of the size of one layer. The gradient with respect to *all* parameters costs about as
much as two forward passes, whatever their number. This is reverse-mode automatic
differentiation.

For one layer $h = f(u)$ inside the chain:

$$\frac{\partial \mathcal{L}}{\partial u}
  = \frac{\partial \mathcal{L}}{\partial h} \cdot \frac{\partial h}{\partial u} .$$

Each operation only needs to know how to turn the gradient of its output into the
gradient of its inputs.

### Autograd in PyTorch

PyTorch records the operations applied to tensors in a **computation graph**, built on
the fly during the forward pass. Calling `.backward()` on a scalar walks the graph in
reverse and fills the `.grad` attribute of every leaf tensor that has
`requires_grad=True`. The parameters of an `nn.Module` have it by default.

```python
w = torch.tensor(3.0, requires_grad=True)
loss = (2 * w - 1) ** 2
loss.backward()
w.grad          # d/dw (2w - 1)^2 = 4 (2w - 1) = 20
```

Three tools control the graph:

| Tool                      | Effect                                              |
|---------------------------|-----------------------------------------------------|
| `tensor.detach()`         | a tensor with the same values, cut from the graph   |
| `torch.no_grad()`         | no graph is built inside the block                  |
| `torch.set_grad_enabled(flag)` | the same, with a boolean                       |

Disabling the graph during evaluation saves memory and time, since the intermediate
values no longer have to be kept.

#### Gradients accumulate

`backward()` *adds* to `.grad`. This is deliberate (it allows accumulating over several
batches), and it means the gradients must be cleared before each step.

### In this project

The whole mechanism appears in `_run_epoch` (`src/train.py`):

```python
with torch.set_grad_enabled(is_training):
    for specs, labels in loader:
        ...
        if is_training:
            optimizer.zero_grad(set_to_none=True)   # 1. clear old gradients

        logits = model(specs)                       # 2. forward pass, graph is built
        loss = criterion(logits, labels)
        if is_training:
            loss.backward()                         # 3. gradients of every parameter
            optimizer.step()                        # 4. update the parameters

        total_loss += loss.item() * labels.size(0)
        predictions.extend((logits.detach().view(-1) >= 0).cpu().int().tolist())
```

- `set_to_none=True` frees the gradient tensors instead of filling them with zeros,
  which is slightly faster.
- `loss.item()` converts the loss to a Python float. Accumulating `loss` itself would
  keep the graph of every batch alive and fill the memory.
- `logits.detach()`: the predictions are only used for metrics and must not hold the
  graph.
- The same function serves for validation: with `is_training=False`, no graph is built
  and steps 1, 3 and 4 are skipped.

### Pitfalls

- **In-place operations** on a tensor needed for the backward pass raise an error
  ("modified by an inplace operation").
- **Non-differentiable steps.** `argmax`, thresholds and conversions to NumPy cut the
  gradient. The metrics of this project are computed from thresholded predictions and
  are therefore not used as a loss.
- **Square root at zero.** $\frac{d}{dx}\sqrt{x}$ is infinite at 0. The standard
  deviation in `LogMelFrontend` adds `1e-4` under the root, which also keeps the
  gradient finite for a silent clip.

### Exercises

1. Compute by hand the gradient of $\mathcal{L} = (wx + b - y)^2$ with respect to $w$
   and $b$ for $x = 2$, $y = 1$, $w = 0.5$, $b = 0$. Check with autograd.
2. Call `loss.backward()` twice in a row on the small example above (use
   `retain_graph=True` the first time). What happens to `w.grad`?
3. Remove `optimizer.zero_grad` from the training loop and observe the training loss
   during a few epochs. Explain.
4. Print `p.grad.norm()` for the first and the last convolution of `AudioCNN` after one
   backward pass.

### Further reading

- Goodfellow, Bengio and Courville, *Deep Learning*, section 6.5.
- PyTorch documentation, "Autograd mechanics".

## 3.3 Optimisers

### Learning goals

- Write the update rules of SGD, momentum, Adam and AdamW.
- Explain the difference between L2 regularisation and decoupled weight decay.
- Know which hyperparameters matter most.

### Theory

Let $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ be the gradient on the current
mini-batch and $\eta$ the **learning rate**.

#### Stochastic gradient descent

$$\theta_{t+1} = \theta_t - \eta \, g_t$$

Too small a learning rate and training crawls; too large and the loss oscillates or
diverges. The learning rate is the most important hyperparameter of all.

#### Momentum

Gradients of successive batches are noisy and often point in inconsistent directions.
Momentum keeps a running average and moves along it:

$$v_{t+1} = \mu \, v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta \, v_{t+1}$$

with $\mu$ typically 0.9. Consistent directions are amplified, oscillations cancel out.

#### Adam

Adam (Kingma and Ba, 2015) adds a *per-parameter* step size. It tracks the average
gradient $m$ and the average squared gradient $v$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \, g_t, \qquad
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) \, g_t^2$$

$$\hat m_t = \frac{m_t}{1 - \beta_1^t}, \quad
  \hat v_t = \frac{v_t}{1 - \beta_2^t}, \qquad
  \theta_{t+1} = \theta_t - \eta \, \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}$$

The division by $\sqrt{\hat v_t}$ normalises the step: parameters with small gradients
take larger steps, and the update has roughly the magnitude $\eta$ whatever the scale of
the loss. The hats correct the bias of averages that start at zero. Defaults:
$\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$. Adam works well with little
tuning, which explains its popularity; $\eta = 10^{-3}$ is the customary starting point.

#### Weight decay and AdamW

Penalising large weights limits overfitting. The classical form adds
$\frac{\lambda}{2} \lVert \theta \rVert^2$ to the loss, which adds $\lambda \theta$ to
the gradient. With plain SGD this is the same as shrinking the weights at each step.
With Adam it is not: the penalty term gets divided by $\sqrt{\hat v_t}$ like the rest, so
parameters with large gradients are regularised *less*.

AdamW (Loshchilov and Hutter, 2019) **decouples** the decay from the adaptive step:

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}
  + \lambda \, \theta_t \right)$$

Every weight shrinks by the same factor, as intended.

### In this project

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
```

with `learning_rate = 0.001` and `weight_decay = 0.01` by default (`src/train.py`),
both adjustable from the command line with `--learning-rate` and `--weight-decay`.

`model.parameters()` hands over every learnable tensor. The optimiser keeps its own
state ($m$ and $v$ for each parameter), which is why resuming a training run properly
requires saving `optimizer.state_dict()` in addition to the weights. This project saves
only the weights: the checkpoints are meant for inference, not for resuming.

### Practical guidance

| Symptom                                   | Likely cause           | Try                    |
|-------------------------------------------|------------------------|------------------------|
| Loss becomes `nan` or explodes            | learning rate too high | divide it by 10        |
| Loss decreases very slowly                | learning rate too low  | multiply it by 3 to 10 |
| Training loss low, validation loss rising | overfitting            | more decay, augmentation, fewer epochs |
| Validation loss jumps between epochs      | step too large for the end of training | a schedule ([section 3.4](#34-learning-rate-schedules-and-early-stopping)) |

The last symptom is visible in this project: in the first training run of `audio_cnn`,
the validation loss went 0.13 → 1.71 → 0.15 between epochs 3 and 5, before the learning
rate was reduced.

### Exercises

1. Minimise $f(\theta) = \theta^2$ by hand with SGD from $\theta_0 = 1$, for
   $\eta = 0.1$, $\eta = 0.9$ and $\eta = 1.1$. Describe the three behaviours.
2. Train `mobile_tiny` with `--learning-rate 0.1`, `0.001` and `0.00001`. Plot or
   compare the validation losses.
3. Replace `AdamW` by `torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)` and
   compare the number of epochs needed.
4. Why is the bias correction $1 / (1 - \beta_1^t)$ important at $t = 1$ and negligible
   at $t = 1000$?

### Further reading

- Kingma and Ba (2015); Loshchilov and Hutter (2019). See
  [References](references.md).

## 3.4 Learning rate schedules and early stopping

### Learning goals

- Explain why the learning rate should decrease during training.
- Describe `ReduceLROnPlateau` and early stopping, and how they interact.
- Understand why the *best* weights are saved, not the last ones.

### Theory

#### Why decrease the learning rate

Mini-batch gradients are noisy. With a constant step $\eta$, the parameters never
settle: they bounce around a minimum at a distance that grows with $\eta$. A large step
is useful at the beginning, to travel fast, and harmful at the end, when precision is
needed. A **schedule** reduces $\eta$ over time.

Common schedules:

| Schedule            | Rule                                                        |
|---------------------|-------------------------------------------------------------|
| Step decay          | multiply by a factor every $k$ epochs                       |
| Cosine annealing    | follow half a cosine from $\eta_0$ down to 0                |
| Warm-up             | start small and increase during the first steps             |
| Reduce on plateau   | reduce when a monitored metric stops improving              |

The first three are fixed in advance. The last one reacts to what actually happens,
which makes it a good default when you do not know how long training should last.

#### Early stopping

As training goes on, the training loss keeps decreasing while the validation loss
eventually rises: the model starts memorising the training set. **Early stopping** ends
training when the validation loss has not improved for a number of epochs, the
*patience*, and keeps the weights of the best epoch. It is a form of regularisation that
costs nothing and removes the need to choose the number of epochs.

### In this project

Both mechanisms monitor the validation loss (`src/train.py`):

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
...
scheduler.step(val_loss)
```

`ReduceLROnPlateau` multiplies the learning rate by 0.1 (its default `factor`) when the
loss has not improved for more than 3 epochs.

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_epoch = epoch
    best_val_metrics = val_metrics
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1
    if epochs_without_improvement >= early_stopping_patience:
        break
```

Two details matter.

**`.clone()`.** `model.state_dict()` returns references to the live parameters. Without
a copy, `best_state` would follow the model and end up containing the *last* weights.

**The patience values are ordered.** Early stopping waits 8 epochs, the scheduler 3.
After a plateau, the learning rate is reduced and the model gets several epochs to
improve at the lower rate before training is abandoned. With the opposite order the
scheduler would never act.

The reference run of `audio_cnn` stopped after 24 epochs and kept epoch 16: exactly 8
epochs without improvement.

### A caveat on the validation set

The validation loss is used three times: to reduce the learning rate, to stop, and to
select the best epoch. The model is therefore, indirectly, fitted to the validation set,
and the validation score is an optimistic estimate. Selecting the best of 24 noisy
measurements favours a lucky one. An unbiased estimate requires a third set that is
touched only once, at the very end ([section 4.2](04-evaluation.md#42-generalisation-and-data-leakage)).

### Pitfalls

- `scheduler.step(val_loss)` needs the metric; most other schedulers take no argument.
- Monitoring accuracy instead of loss is tempting but coarse: with 200 validation clips
  accuracy moves in steps of 0.005 and many epochs tie.
- With a patience that is too short, a single noisy epoch ends training early. Compare
  `mobile_small` (kept epoch 5, stopped at 13) with `mobile_tiny` (kept epoch 37).

### Exercises

1. Log `optimizer.param_groups[0]["lr"]` at every epoch and find when the learning rate
   was reduced.
2. Train with `--early-stopping-patience 2`, then `20`. Compare the best epoch and the
   best validation loss.
3. Replace the scheduler by
   `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)` (call
   `scheduler.step()` without argument) and compare.
4. Remove `.clone()` from `best_state` and show, with a test, that the saved weights are
   then those of the last epoch.

### Further reading

- Prechelt, "Early Stopping - But When?" (1998).
- Loshchilov and Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017).

## 3.5 Regularisation and augmentation

### Learning goals

- Define overfitting in terms of the gap between training and validation error.
- Explain dropout, weight decay and data augmentation.
- Read the augmentations of this project and reason about their strength.

### Theory

#### Overfitting

A model with enough capacity can memorise its training set, noise included. The symptom
is a low training loss with a higher validation loss. **Regularisation** is anything that
reduces this gap, usually by making memorisation harder than learning the general rule.

This project uses five regularisers; three were covered earlier (weight decay in
[section 3.3](#33-optimisers), early stopping in
[section 3.4](#34-learning-rate-schedules-and-early-stopping), batch normalisation in
[section 2.2](02-convolutional-networks.md#22-normalisation-activation-and-pooling)). The two others follow.

#### Dropout

During training, dropout (Srivastava et al., 2014) sets each activation to zero with
probability $p$ and multiplies the survivors by $1 / (1 - p)$, so that the expected value
is unchanged:

$$\tilde h_i = \frac{m_i \, h_i}{1 - p}, \qquad m_i \sim \text{Bernoulli}(1 - p) .$$

No unit can rely on the presence of another one, which discourages fragile
co-adaptations. One interpretation is that each step trains a different sub-network and
the final model averages them. At evaluation time dropout does nothing; like batch
normalisation, it is switched by `model.train()` and `model.eval()`.

#### Data augmentation

The most effective regulariser is more data. Augmentation manufactures it by applying
transformations that change the input but not its label. The choice of transformations
encodes knowledge about the problem: a snore remains a snore if it starts 200 ms later.

**SpecAugment** (Park et al., 2019) operates on the spectrogram:

- *frequency masking*: a band of consecutive mel channels is set to zero;
- *time masking*: a span of consecutive frames is set to zero.

The model must then recognise a sound from incomplete evidence, and cannot depend on one
particular band or instant.

### In this project

| Technique          | Where                         | Setting                              |
|--------------------|-------------------------------|--------------------------------------|
| Dropout            | `AudioCNN` classifier         | $p = 0.3$, twice                     |
| Dropout            | `SnoreMobileNet` classifier   | $p = 0.2$                            |
| Weight decay       | `AdamW` in `src/train.py`     | $\lambda = 0.01$                     |
| Frequency masking  | `AudioDataset` (128 bands)    | up to 15 bands                       |
| Time masking       | `AudioDataset` (32 frames)    | up to 6 frames                       |
| Frequency masking  | `SnoreMobileNet` (40 bands)   | up to 6 bands                        |
| Time masking       | `SnoreMobileNet` (101 frames) | up to 15 frames                      |
| Random time shift  | `AudioDataset(features=False)`| circular, any offset                 |

The masks are applied in the dataset for `audio_cnn` and inside the model for the mobile
models, guarded by `if self.training:` so that they disappear at evaluation and in the
exported file.

#### Augmentation strength is a hyperparameter

When the clip duration was corrected from 3 seconds to 1, the time mask kept its old
maximum of 30 frames while clips now had 32 frames: a single mask could erase almost the
entire clip, leaving a label with no evidence. The fix scales the mask to the clip:

```python
num_frames = self.num_samples // hop_length + 1
self.time_mask = torchaudio.transforms.TimeMasking(
    time_mask_param=max(1, min(30, num_frames // 5))
)
```

An augmentation that destroys the information defining the label teaches the model
noise. Always look at augmented examples before training.

#### The time shift

```python
shift = int(torch.randint(0, self.num_samples, (1,)))
waveform = torch.roll(waveform, shift)
```

`torch.roll` is circular: what leaves on the right re-enters on the left. This creates
an artificial discontinuity at the junction. It is a simple approximation; shifting with
padding or cropping from a longer recording would be more faithful.

### Augmentations not used here

- **Additive noise** at a random signal-to-noise ratio, ideally real bedroom noise.
- **Mixup**: train on convex combinations of two clips and of their labels.
- **Random gain**: useless for the mobile models, whose per-clip normalisation already
  removes gain ([section 1.3](01-audio-and-tensors.md#13-mel-and-log-scales)).
- **Pitch shifting and time stretching**: to use with care, since they may change what
  makes a snore a snore.

### Pitfalls

- Never augment the validation set.
- torchaudio's masking transforms draw one mask for the whole tensor they receive. In
  the dataset that is one mask per clip; inside the mobile models it is one mask per
  *batch*.

### Exercises

1. Plot one training spectrogram five times with `train=True`. Do the masks look
   reasonable?
2. Train `audio_cnn` with `dropout=0.0` and without masking. Compare the gap between the
   training and validation losses.
3. Implement additive Gaussian noise at a random level in `AudioDataset` for
   `features=False`, and evaluate its effect on the recordings of `data/inference/`.
4. Why is dropout multiplied by $1 / (1 - p)$ during training rather than scaling the
   activations at evaluation time?

### Further reading

- Srivastava et al. (2014); Park et al. (2019). See [References](references.md).

## 3.6 Anatomy of the training loop

### Learning goals

- Assemble the previous chapters into a complete training procedure.
- Know the role of every line of `src/train.py`.
- Understand what a checkpoint contains and why.

### The algorithm

```text
load the data, build the model, the loss, the optimiser, the scheduler
repeat for each epoch:
    training pass     for each batch: forward, loss, backward, update
    validation pass   for each batch: forward, loss            (no gradient)
    adjust the learning rate
    keep a copy of the weights if the validation loss improved
    stop if it has not improved for a while
save the best weights
```

Every deep learning training script is a variation on these lines.

### Reading `train_model`

**1. Guard clauses.** The function first checks that the preprocessed file exists, that
it contains the expected arrays, and that the arguments are valid. Failing early with a
clear message is cheaper than failing after ten minutes of training.

**2. Seeds** ([section 4.3](04-evaluation.md#43-reproducibility)).

**3. Data.** Two `AudioDataset` objects with `train=True` and `train=False`, wrapped in
`DataLoader`s ([section 1.5](01-audio-and-tensors.md#15-datasets-and-data-loaders)).

**4. Device.**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(model_name).to(device)
```

The model is moved once. Each batch is moved inside the loop, because the data stays on
the CPU until it is needed:

```python
specs = specs.to(device, non_blocking=True)
```

`non_blocking=True`, combined with `pin_memory=True` in the loader, lets the copy to the
GPU happen asynchronously.

**5. The epoch function.** `_run_epoch` serves both passes; the presence of an optimiser
decides which:

```python
is_training = optimizer is not None
model.train(is_training)
```

One function instead of two guarantees that the loss and the metrics are computed in
exactly the same way for training and validation.

**6. Averaging the loss correctly.**

```python
total_loss += loss.item() * labels.size(0)
...
return total_loss / len(targets)
```

The loss of a batch is a mean over that batch. Multiplying by the batch size before
summing gives the exact mean over the dataset even when the last batch is smaller.

**7. The checkpoint.**

```python
torch.save(
    {
        "model_state_dict": best_state,
        "model_name": model_name,
        "sample_rate": sample_rate,
        "duration": duration,
        ...
        "best_epoch": best_epoch,
        "validation_loss": best_val_loss,
        "validation_metrics": best_val_metrics,
    },
    model_path,
)
```

A `state_dict` is an ordered dictionary from parameter names to tensors: the weights
without the code. Saving it, rather than the whole model object, keeps the file loadable
after the code is refactored. The other fields answer the questions you will have in
three months: which model, on which audio format, how good.

Loading mirrors saving:

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model = build_model(checkpoint["model_name"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

`weights_only=True` refuses to execute arbitrary pickled code, which matters whenever a
checkpoint comes from someone else. `map_location="cpu"` allows loading on a machine
without a GPU.

### Training and validation metrics are not comparable as such

The training loss of an epoch is measured in training mode: with dropout, with masked
spectrograms, with batch statistics, and while the weights are still changing. The
validation loss is measured afterwards, on clean inputs. It is therefore normal to see a
validation loss *below* the training loss, especially in the first epochs. To compare
them fairly, evaluate the training set in `eval()` mode as well.

### What a production loop would add

- Mixed precision (`torch.autocast`) and gradient clipping.
- Periodic checkpoints that include the optimiser state, to resume after a crash.
- Experiment tracking (TensorBoard, Weights & Biases) instead of log lines.
- A learning-rate warm-up.

None is needed for a model that trains in under a minute, and each adds complexity. Add
them when a problem calls for them.

### Exercises

1. Draw the call graph of `python -m src.main`: which function calls which, in which
   file.
2. Add the training accuracy to the log line of each epoch.
3. Add a second evaluation of the training set in `eval()` mode and compare its loss
   with the one measured during the training pass.
4. Save `optimizer.state_dict()` in the checkpoint and write a `--resume` option.
5. Use `torch.nn.utils.clip_grad_norm_` before `optimizer.step()` and log the gradient
   norm.

### Further reading

- PyTorch tutorial, "Optimizing Model Parameters" and "Save and Load the Model".
