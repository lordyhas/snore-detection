# 4. Tensors and shapes

## Learning goals

- Manipulate PyTorch tensors: shape, dtype, device.
- Use `unsqueeze`, `squeeze`, `view`, `transpose` and broadcasting with confidence.
- Follow the shape of the data from the audio file to the loss.

## Theory

A **tensor** is an n-dimensional array with three attributes that explain most runtime
errors:

| Attribute | Example         | Typical error when wrong                         |
|-----------|-----------------|--------------------------------------------------|
| `shape`   | `(32, 1, 128, 32)` | `size mismatch`, silent wrong broadcasting    |
| `dtype`   | `torch.float32` | `expected Float but got Long`                    |
| `device`  | `cuda:0`        | `expected all tensors to be on the same device`  |

### Shape conventions

- 2-D convolutions expect `(batch, channels, height, width)`, written **NCHW**.
- For a spectrogram, height is frequency and width is time; there is one channel.

### Reshaping without copying

```python
x = torch.zeros(16000)        # (16000,)
x.unsqueeze(0).shape          # (1, 16000)      add a dimension of size 1
x.view(4, 4000).shape         # (4, 4000)       same data, new shape
x.view(-1, 1).shape           # (16000, 1)      -1 = "whatever fits"
```

`transpose(1, 2)` swaps two dimensions; `flatten(1)` merges every dimension after the
first.

### Broadcasting

When two shapes differ, PyTorch aligns them from the right and stretches dimensions of
size 1. `(8, 40, 101) - (8, 1, 1)` subtracts one number per clip from a whole
spectrogram. This is what `keepdim=True` prepares:

```python
log_mel.mean(dim=(1, 2), keepdim=True).shape   # (8, 1, 1), ready to broadcast
```

Broadcasting is convenient and dangerous: `(32,)` against `(32, 1)` silently produces a
`(32, 32)` matrix.

## In this project

### `audio_cnn`

| Step                               | Shape               | dtype     |
|------------------------------------|---------------------|-----------|
| `dataset.npz`, `x_train`           | `(800, 16000)`      | float32   |
| `AudioDataset.__getitem__` output  | `(1, 128, 32)`      | float32   |
| batch from the `DataLoader`        | `(32, 1, 128, 32)`  | float32   |
| labels from the `DataLoader`       | `(32,)`             | float32   |
| labels after `.view(-1, 1)`        | `(32, 1)`           | float32   |
| logits                             | `(32, 1)`           | float32   |

### Mobile models

| Step                               | Shape               |
|------------------------------------|---------------------|
| batch from the `DataLoader`        | `(32, 16000)`       |
| after `torch.stft`                 | `(32, 257, 101)` complex |
| after the mel filter bank          | `(32, 40, 101)`     |
| after `unsqueeze(1)`               | `(32, 1, 40, 101)`  |
| logits                             | `(32, 1)`           |

Two lines of `src/train.py` exist only because of shapes and dtypes:

```python
labels = labels.to(device, non_blocking=True).view(-1, 1)
```

The loss compares logits `(32, 1)` with labels: without `.view(-1, 1)` the labels would
be `(32,)` and the loss would raise an error. And in the dataset the label is created as
`float32`, because `BCEWithLogitsLoss` needs float targets even though they are 0 or 1.

## Pitfalls

- `view` requires contiguous memory; after a `transpose`, use `reshape`.
- `x.numpy()` and `torch.from_numpy(a)` share memory with the original: modifying one
  modifies the other.
- The last batch of an epoch can be smaller than `batch_size`. Never hard-code the
  batch dimension: use `-1` or `x.size(0)`.

## Exercises

1. What is the shape of `torch.zeros(8, 40, 101).mean(dim=(1, 2))`? And with
   `keepdim=True`?
2. In a Python shell, build `AudioDataset` on ten random waveforms and print the shape
   of one item for `features=True` and `features=False`.
3. Predict, then check: `(torch.zeros(32) - torch.zeros(32, 1)).shape`.

## Further reading

- PyTorch documentation, "Broadcasting semantics" and "Tensor views".
- Next: [5. Datasets and data loaders](05-datasets-and-dataloaders.md).
