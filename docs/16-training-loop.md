# 16. Anatomy of the training loop

## Learning goals

- Assemble the previous chapters into a complete training procedure.
- Know the role of every line of `src/train.py`.
- Understand what a checkpoint contains and why.

## The algorithm

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

## Reading `train_model`

**1. Guard clauses.** The function first checks that the preprocessed file exists, that
it contains the expected arrays, and that the arguments are valid. Failing early with a
clear message is cheaper than failing after ten minutes of training.

**2. Seeds** ([chapter 19](19-reproducibility.md)).

**3. Data.** Two `AudioDataset` objects with `train=True` and `train=False`, wrapped in
`DataLoader`s ([chapter 5](05-datasets-and-dataloaders.md)).

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

## Training and validation metrics are not comparable as such

The training loss of an epoch is measured in training mode: with dropout, with masked
spectrograms, with batch statistics, and while the weights are still changing. The
validation loss is measured afterwards, on clean inputs. It is therefore normal to see a
validation loss *below* the training loss, especially in the first epochs. To compare
them fairly, evaluate the training set in `eval()` mode as well.

## What a production loop would add

- Mixed precision (`torch.autocast`) and gradient clipping.
- Periodic checkpoints that include the optimiser state, to resume after a crash.
- Experiment tracking (TensorBoard, Weights & Biases) instead of log lines.
- A learning-rate warm-up.

None is needed for a model that trains in under a minute, and each adds complexity. Add
them when a problem calls for them.

## Exercises

1. Draw the call graph of `python -m src.main`: which function calls which, in which
   file.
2. Add the training accuracy to the log line of each epoch.
3. Add a second evaluation of the training set in `eval()` mode and compare its loss
   with the one measured during the training pass.
4. Save `optimizer.state_dict()` in the checkpoint and write a `--resume` option.
5. Use `torch.nn.utils.clip_grad_norm_` before `optimizer.step()` and log the gradient
   norm.

## Further reading

- PyTorch tutorial, "Optimizing Model Parameters" and "Save and Load the Model".
- Next: [17. Evaluation metrics](17-evaluation-metrics.md).
