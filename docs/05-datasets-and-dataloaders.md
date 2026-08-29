# 5. Datasets and data loaders

## Learning goals

- Implement the `Dataset` protocol and understand what a `DataLoader` adds.
- Explain mini-batches, shuffling and epochs.
- Decide what belongs in preprocessing, in the dataset, and in the model.

## Theory

### Mini-batch stochastic gradient descent

The loss we want to minimise is an average over the whole training set. Computing it
exactly at every step is wasteful, so we estimate it on a random **mini-batch** of $B$
examples. The estimate is noisy but unbiased, and the noise itself helps to escape poor
solutions. One pass over all the data is an **epoch**. With 800 training clips and
$B = 32$, an epoch is 25 gradient steps.

### The two classes

A **`Dataset`** answers two questions:

```python
def __len__(self):            # how many examples?
def __getitem__(self, idx):   # give me example number idx
```

A **`DataLoader`** wraps a dataset and takes care of the rest: drawing indices
(`shuffle=True`), stacking examples into a batch, optionally loading in parallel
processes (`num_workers`) and preparing fast transfers to the GPU (`pin_memory`).

Shuffle the training set at every epoch, so that the batches differ from one epoch to
the next. Never shuffle for evaluation: it changes nothing and makes results harder to
compare.

## In this project

The work is split in three stages.

**1. Preprocessing, once** (`src/data/preprocess.py`). Decoding and resampling 1000
files takes a few seconds; doing it at every epoch would dominate the training time. The
result is saved as one compressed file, `data/processed/dataset.npz`, with the split
already made.

Note how the file is written: to a temporary file first, then `os.replace`. A crash in
the middle can never leave a half-written dataset.

**2. Dataset, at every access** (`src/data/data_manager.py`). `AudioDataset` keeps the
waveforms in memory and transforms one at a time:

```python
spec = self.to_db(self.mel(waveform))  # (1, n_mels, frames)
if self.train:
    spec = self.time_mask(self.freq_mask(spec))
```

Anything *random* must happen here and not in preprocessing: the augmentation has to
differ at every epoch (see [chapter 15](15-regularization-and-augmentation.md)).

**3. Model.** The mobile models compute the spectrogram themselves, so
`AudioDataset(features=False)` returns the raw waveform. The reason is deployment: the
phone then needs no signal-processing code.

The loaders are created in `src/train.py`:

```python
generator = torch.Generator().manual_seed(RANDOM_SEED)
train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
```

The seeded `generator` makes the shuffling order reproducible.

## Pitfalls

- **`num_workers` on Windows.** Worker processes are started with *spawn*: the script is
  imported again in each worker, so the entry point must be protected by
  `if __name__ == "__main__":`. This project defaults to `num_workers=0`, which is
  fast enough for 800 short clips held in memory.
- **Random augmentation in workers.** Each worker has its own random state. PyTorch
  seeds them differently for you, but NumPy's global state used to be duplicated across
  workers; prefer `torch` random functions inside a dataset.
- **Transforms on the validation set.** `train=False` disables augmentation. Evaluating
  on augmented data would measure the wrong thing.

## Exercises

1. Iterate once over `train_loader` and count the batches. What is the size of the
   last one if the training set has 810 examples?
2. Time one epoch of data loading alone (no model) with `num_workers=0` and
   `num_workers=2`. Which one is faster here, and why?
3. Write a minimal `Dataset` that returns `(idx, idx ** 2)` and wrap it in a
   `DataLoader` with `batch_size=4`.

## Further reading

- PyTorch tutorial, "Datasets & DataLoaders".
- Next: [6. Convolution](06-convolution.md).
