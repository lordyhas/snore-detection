# 12. Autograd and backpropagation

## Learning goals

- State what backpropagation computes and why it is efficient.
- Use `requires_grad`, `backward`, `detach` and `no_grad` correctly.
- Explain why gradients must be reset at every step.

## Theory

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

## Autograd in PyTorch

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

### Gradients accumulate

`backward()` *adds* to `.grad`. This is deliberate (it allows accumulating over several
batches), and it means the gradients must be cleared before each step.

## In this project

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

## Pitfalls

- **In-place operations** on a tensor needed for the backward pass raise an error
  ("modified by an inplace operation").
- **Non-differentiable steps.** `argmax`, thresholds and conversions to NumPy cut the
  gradient. The metrics of this project are computed from thresholded predictions and
  are therefore not used as a loss.
- **Square root at zero.** $\frac{d}{dx}\sqrt{x}$ is infinite at 0. The standard
  deviation in `LogMelFrontend` adds `1e-4` under the root, which also keeps the
  gradient finite for a silent clip.

## Exercises

1. Compute by hand the gradient of $\mathcal{L} = (wx + b - y)^2$ with respect to $w$
   and $b$ for $x = 2$, $y = 1$, $w = 0.5$, $b = 0$. Check with autograd.
2. Call `loss.backward()` twice in a row on the small example above (use
   `retain_graph=True` the first time). What happens to `w.grad`?
3. Remove `optimizer.zero_grad` from the training loop and observe the training loss
   during a few epochs. Explain.
4. Print `p.grad.norm()` for the first and the last convolution of `AudioCNN` after one
   backward pass.

## Further reading

- Goodfellow, Bengio and Courville, *Deep Learning*, section 6.5.
- PyTorch documentation, "Autograd mechanics".
- Next: [13. Optimisers](13-optimizers.md).
