# 13. Optimisers

## Learning goals

- Write the update rules of SGD, momentum, Adam and AdamW.
- Explain the difference between L2 regularisation and decoupled weight decay.
- Know which hyperparameters matter most.

## Theory

Let $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ be the gradient on the current
mini-batch and $\eta$ the **learning rate**.

### Stochastic gradient descent

$$\theta_{t+1} = \theta_t - \eta \, g_t$$

Too small a learning rate and training crawls; too large and the loss oscillates or
diverges. The learning rate is the most important hyperparameter of all.

### Momentum

Gradients of successive batches are noisy and often point in inconsistent directions.
Momentum keeps a running average and moves along it:

$$v_{t+1} = \mu \, v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta \, v_{t+1}$$

with $\mu$ typically 0.9. Consistent directions are amplified, oscillations cancel out.

### Adam

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

### Weight decay and AdamW

Penalising large weights limits overfitting. The classical form adds
$\frac{\lambda}{2} \lVert \theta \rVert^2$ to the loss, which adds $\lambda \theta$ to
the gradient. With plain SGD this is the same as shrinking the weights at each step.
With Adam it is not: the penalty term gets divided by $\sqrt{\hat v_t}$ like the rest, so
parameters with large gradients are regularised *less*.

AdamW (Loshchilov and Hutter, 2019) **decouples** the decay from the adaptive step:

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}
  + \lambda \, \theta_t \right)$$

Every weight shrinks by the same factor, as intended.

## In this project

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

## Practical guidance

| Symptom                                   | Likely cause           | Try                    |
|-------------------------------------------|------------------------|------------------------|
| Loss becomes `nan` or explodes            | learning rate too high | divide it by 10        |
| Loss decreases very slowly                | learning rate too low  | multiply it by 3 to 10 |
| Training loss low, validation loss rising | overfitting            | more decay, augmentation, fewer epochs |
| Validation loss jumps between epochs      | step too large for the end of training | a schedule ([chapter 14](14-learning-rate-and-early-stopping.md)) |

The last symptom is visible in this project: in the first training run of `audio_cnn`,
the validation loss went 0.13 → 1.71 → 0.15 between epochs 3 and 5, before the learning
rate was reduced.

## Exercises

1. Minimise $f(\theta) = \theta^2$ by hand with SGD from $\theta_0 = 1$, for
   $\eta = 0.1$, $\eta = 0.9$ and $\eta = 1.1$. Describe the three behaviours.
2. Train `mobile_tiny` with `--learning-rate 0.1`, `0.001` and `0.00001`. Plot or
   compare the validation losses.
3. Replace `AdamW` by `torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)` and
   compare the number of epochs needed.
4. Why is the bias correction $1 / (1 - \beta_1^t)$ important at $t = 1$ and negligible
   at $t = 1000$?

## Further reading

- Kingma and Ba (2015); Loshchilov and Hutter (2019). See
  [References](25-references.md).
- Next: [14. Learning rate schedules and early stopping](14-learning-rate-and-early-stopping.md).
