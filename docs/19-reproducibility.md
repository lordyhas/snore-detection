# 19. Reproducibility

## Learning goals

- List the sources of randomness in a training run.
- Seed them, and know what seeding does not guarantee.
- Report results in a way that others, and your future self, can trust.

## Theory

### Where randomness enters

| Source                               | In this project                              |
|--------------------------------------|----------------------------------------------|
| Train / validation split             | `train_test_split(random_state=...)`         |
| Weight initialisation                | when the model is built                      |
| Order of the batches                 | `DataLoader(shuffle=True)`                   |
| Dropout masks                        | at every training step                       |
| Augmentation (masks, time shift)     | at every access to a training example        |
| Non-deterministic GPU algorithms     | some cuDNN convolution implementations       |

Two runs of the same code therefore give two different models. The difference is not a
bug; it is a property of the method that must be measured and controlled.

### Pseudo-random generators and seeds

A pseudo-random generator is a deterministic function of its internal state. Setting the
**seed** sets the state, hence the entire sequence that follows. Python, NumPy and
PyTorch each have their own generator, and PyTorch has one per device.

## In this project

```python
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
...
generator = torch.Generator().manual_seed(RANDOM_SEED)
train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, ...)
```

`RANDOM_SEED = 42` is defined once, in `src/config.py`, and reused for the split. A
single constant avoids the situation where two parts of the code are seeded differently.

The seeds are set *before* the model is built, because building the model consumes
random numbers for the initial weights.

## What seeding does not guarantee

- **Across hardware.** CPU and GPU do not perform floating-point operations in the same
  order, and the results differ in the last bits. Those differences are then amplified
  by training.
- **On a GPU.** Some cuDNN algorithms are not deterministic. Full determinism requires

  ```python
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.benchmark = False
  ```

  at some cost in speed. This project does not set them, so two GPU runs can differ
  slightly.
- **Across versions.** A new release of PyTorch may change an initialisation or a
  kernel. This is why `requirements.txt` records minimum versions, and why a strict
  reproduction needs exact ones (`pip freeze`).
- **With several workers.** The order in which workers deliver batches and their
  individual random states must be controlled too (`worker_init_fn`).

## Floating point is part of the story

Floating-point addition is not associative: $(a + b) + c$ may differ from $a + (b + c)$
in the last bit. Two correct implementations of the same formula can therefore disagree
slightly. Normally this is invisible. It becomes visible when a computation *amplifies*
small differences, which happened in this project between PyTorch and ONNX Runtime
([chapter 20](20-exporting-to-onnx.md)). A numerically robust model is one whose output
does not depend on the last bits.

## Reporting results

A single run with a single seed is an anecdote. Good practice:

1. Fix the data split and vary the training seed over several runs (five is a common
   minimum). Report mean and standard deviation.
2. State what was selected on which set
   ([chapter 18](18-generalization-and-data-leakage.md)).
3. Record the exact command, the commit and the package versions with every result.
4. Do not choose the seed that gives the best number. The seed is not a hyperparameter.

The results quoted in this documentation come from single runs. They are indications,
and the tables say so.

## Exercises

1. Run `python -m src.train --model mobile_tiny` twice on the CPU. Are the logs
   identical? Then twice on the GPU.
2. Add a `--seed` option to `src/train.py` that only affects training, not the split.
   Run five seeds and report mean and standard deviation of the validation accuracy.
3. Evaluate `0.1 + 0.2 == 0.3` and
   `(0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3)` in Python. Explain.
4. Move `torch.manual_seed` *after* `build_model`. What changes between two runs?

## Further reading

- PyTorch documentation, "Reproducibility".
- Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks" (2021).
- Next: [20. Exporting to ONNX](20-exporting-to-onnx.md).
