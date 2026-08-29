# 8. Walk-through of the audio CNN

## Learning goals

- Read an `nn.Module` and predict the shape of every intermediate tensor.
- Account for every parameter of a real model.
- Understand the design pattern "feature extractor, pooling, classifier".

## How a module is written

A PyTorch model is a class deriving from `nn.Module` with two methods:

- `__init__` creates the layers. Assigning a module to `self` registers its parameters;
- `forward` describes the computation. Calling `model(x)` runs `forward` (plus hooks),
  so never call `forward` directly.

`nn.Sequential` chains modules whose output feeds the next one.

## The architecture

`AudioCNN` in `src/models/audio_cnn.py` follows the VGG recipe: stack simple blocks,
halve the resolution and double the channels at each stage. Spatial detail is traded for
semantic detail.

Shapes for a 1-second clip (128 mel bands, 32 frames):

| Stage                     | Output shape       | Parameters |
|---------------------------|--------------------|------------|
| Input                     | `(1, 128, 32)`     |            |
| `ConvBlock(1, 32)`        | `(32, 64, 16)`     | 9 632      |
| `ConvBlock(32, 64)`       | `(64, 32, 8)`      | 55 552     |
| `ConvBlock(64, 128)`      | `(128, 16, 4)`     | 221 696    |
| `ConvBlock(128, 256)`     | `(256, 8, 2)`      | 885 760    |
| `AdaptiveAvgPool2d(1)`    | `(256, 1, 1)`      | 0          |
| `Flatten` and `Dropout`   | `(256,)`           | 0          |
| `Linear(256, 64)`, ReLU   | `(64,)`            | 16 448     |
| `Dropout`, `Linear(64, 1)`| `(1,)`             | 65         |
| **Total**                 |                    | **1 189 153** |

Three observations.

**Where the parameters are.** The last block holds 74 % of them. The cost of a
convolution grows with the *product* of its input and output channels, so the wide, late
layers dominate. This is the starting point of
[chapter 9](09-depthwise-separable-convolutions.md).

**Where the computation is.** The opposite holds for the number of operations: early
layers work on large maps. The first block applies its kernels at 128 × 32 positions,
the last one at 16 × 4.

**One output.** The model returns a single number, a *logit*, not a probability.
The sigmoid is applied by the loss during training
([chapter 11](11-loss-functions.md)) and by the caller at inference.

## The classifier head

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(dropout),
    nn.Linear(256, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout),
    nn.Linear(64, num_outputs),
)
```

After global pooling, each clip is summarised by 256 numbers. A small two-layer
perceptron maps them to the decision. Dropout is placed here, on the dense layers, where
it is most useful ([chapter 15](15-regularization-and-augmentation.md)).

## Is the model the right size?

There are 800 training clips and 1.19 million parameters: about 1500 parameters per
example. Classical statistics would predict heavy overfitting. In practice the
validation accuracy reaches 0.99, thanks to weight sharing, batch normalisation, dropout,
augmentation and early stopping. But `mobile_tiny`, with 9 889 parameters, reaches 0.98
on the same split. On this dataset the large model is far from necessary, which is a
hint that the task, as posed by this dataset, is easy
([chapter 18](18-generalization-and-data-leakage.md)).

## Exercises

1. Run the following and compare with the table:

   ```python
   from src.models.audio_cnn import AudioCNN
   model = AudioCNN()
   for name, module in model.features.named_children():
       print(name, sum(p.numel() for p in module.parameters()))
   ```

2. Pass `torch.zeros(1, 1, 128, 32)` through `model.features` and print the shape after
   each block.
3. The model accepts a 3-second clip (94 frames) without modification. Which layer makes
   this possible?
4. Replace the two-layer head by a single `Linear(256, 1)`. How many parameters are
   saved? Does the validation accuracy change?

## Further reading

- Simonyan and Zisserman, "Very Deep Convolutional Networks for Large-Scale Image
  Recognition" (2015).
- Next: [9. Depthwise separable convolutions](09-depthwise-separable-convolutions.md).
