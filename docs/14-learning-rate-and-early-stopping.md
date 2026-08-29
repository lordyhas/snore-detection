# 14. Learning rate schedules and early stopping

## Learning goals

- Explain why the learning rate should decrease during training.
- Describe `ReduceLROnPlateau` and early stopping, and how they interact.
- Understand why the *best* weights are saved, not the last ones.

## Theory

### Why decrease the learning rate

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

### Early stopping

As training goes on, the training loss keeps decreasing while the validation loss
eventually rises: the model starts memorising the training set. **Early stopping** ends
training when the validation loss has not improved for a number of epochs, the
*patience*, and keeps the weights of the best epoch. It is a form of regularisation that
costs nothing and removes the need to choose the number of epochs.

## In this project

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

## A caveat on the validation set

The validation loss is used three times: to reduce the learning rate, to stop, and to
select the best epoch. The model is therefore, indirectly, fitted to the validation set,
and the validation score is an optimistic estimate. Selecting the best of 24 noisy
measurements favours a lucky one. An unbiased estimate requires a third set that is
touched only once, at the very end ([chapter 18](18-generalization-and-data-leakage.md)).

## Pitfalls

- `scheduler.step(val_loss)` needs the metric; most other schedulers take no argument.
- Monitoring accuracy instead of loss is tempting but coarse: with 200 validation clips
  accuracy moves in steps of 0.005 and many epochs tie.
- With a patience that is too short, a single noisy epoch ends training early. Compare
  `mobile_small` (kept epoch 5, stopped at 13) with `mobile_tiny` (kept epoch 37).

## Exercises

1. Log `optimizer.param_groups[0]["lr"]` at every epoch and find when the learning rate
   was reduced.
2. Train with `--early-stopping-patience 2`, then `20`. Compare the best epoch and the
   best validation loss.
3. Replace the scheduler by
   `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)` (call
   `scheduler.step()` without argument) and compare.
4. Remove `.clone()` from `best_state` and show, with a test, that the saved weights are
   then those of the last epoch.

## Further reading

- Prechelt, "Early Stopping - But When?" (1998).
- Loshchilov and Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017).
- Next: [15. Regularisation and augmentation](15-regularization-and-augmentation.md).
