# 24. Glossary

Terms are listed alphabetically, with the chapter where they are explained.

**Activation function.** Non-linear function applied after a layer, such as ReLU.
[Chapter 7](07-normalization-activation-pooling.md).

**Adam, AdamW.** Optimisers with a per-parameter adaptive step; AdamW applies weight
decay separately from the adaptive step. [Chapter 13](13-optimizers.md).

**Aliasing.** Frequencies above the Nyquist limit folded onto lower ones when sampling.
[Chapter 1](01-digital-audio.md).

**Augmentation.** Random label-preserving transformation of training examples.
[Chapter 15](15-regularization-and-augmentation.md).

**Autograd.** PyTorch's automatic differentiation engine.
[Chapter 12](12-autograd-and-backpropagation.md).

**Backpropagation.** Computation of all gradients by applying the chain rule from the
output backwards. [Chapter 12](12-autograd-and-backpropagation.md).

**Batch (mini-batch).** Group of examples processed together for one gradient step.
[Chapter 5](05-datasets-and-dataloaders.md).

**Batch normalisation.** Layer that standardises activations per channel, with batch
statistics in training and running statistics in evaluation.
[Chapter 7](07-normalization-activation-pooling.md).

**Binary cross-entropy.** Loss for two-class problems; the negative log-likelihood of a
Bernoulli model. [Chapter 11](11-loss-functions.md).

**Broadcasting.** Automatic expansion of dimensions of size 1 in element-wise
operations. [Chapter 4](04-tensors-and-shapes.md).

**Buffer.** Tensor that belongs to a module without being a learned parameter, such as
running statistics or a mel filter bank. [Chapter 2](02-spectrograms.md).

**Checkpoint.** File holding the weights of a model and the information needed to use
them. [Chapter 16](16-training-loop.md).

**Confusion matrix.** Table of true and false positives and negatives.
[Chapter 17](17-evaluation-metrics.md).

**Convolution.** Layer that applies the same small kernel at every position of its
input. [Chapter 6](06-convolution.md).

**Data leakage.** Information from the evaluation data reaching the training process.
[Chapter 18](18-generalization-and-data-leakage.md).

**Decibel (dB).** Logarithmic unit of power ratio: $10 \log_{10}$ of the ratio.
[Chapter 3](03-mel-and-log-scales.md).

**Depthwise separable convolution.** A per-channel spatial convolution followed by a
1×1 convolution. [Chapter 9](09-depthwise-separable-convolutions.md).

**Distribution shift.** Difference between the data a model was trained on and the data
it meets in use. [Chapter 18](18-generalization-and-data-leakage.md).

**Dropout.** Random zeroing of activations during training.
[Chapter 15](15-regularization-and-augmentation.md).

**Early stopping.** Ending training when the validation loss stops improving and keeping
the best weights. [Chapter 14](14-learning-rate-and-early-stopping.md).

**Epoch.** One pass over the training set. [Chapter 5](05-datasets-and-dataloaders.md).

**F1 score.** Harmonic mean of precision and recall.
[Chapter 17](17-evaluation-metrics.md).

**Generalisation.** Performance on data not used for training.
[Chapter 18](18-generalization-and-data-leakage.md).

**Global average pooling.** Averaging each channel over all positions.
[Chapter 7](07-normalization-activation-pooling.md).

**Gradient.** Vector of partial derivatives of the loss with respect to the parameters.
[Chapter 12](12-autograd-and-backpropagation.md).

**Hop length.** Number of samples between two consecutive STFT frames.
[Chapter 2](02-spectrograms.md).

**Hyperparameter.** Setting chosen by the practitioner rather than learned: learning
rate, batch size, number of layers. [Chapter 13](13-optimizers.md).

**Kernel.** The small array of weights of a convolution. [Chapter 6](06-convolution.md).

**Learning rate.** Size of the step taken along the gradient.
[Chapter 13](13-optimizers.md).

**Logit.** Raw output of the model before the sigmoid; the log-odds.
[Chapter 11](11-loss-functions.md).

**Loss.** Function measuring the error of the model, minimised by training.
[Chapter 11](11-loss-functions.md).

**Mel scale.** Perceptual frequency scale, fine at low frequencies and coarse at high
ones. [Chapter 3](03-mel-and-log-scales.md).

**Momentum.** Running average of past gradients used as the update direction.
[Chapter 13](13-optimizers.md).

**Nyquist frequency.** Half the sampling rate; the highest representable frequency.
[Chapter 1](01-digital-audio.md).

**ONNX.** Portable file format for neural networks.
[Chapter 20](20-exporting-to-onnx.md).

**Overfitting.** Fitting the training data, noise included, at the expense of
generalisation. [Chapter 15](15-regularization-and-augmentation.md).

**Parameter.** Learned number of the model; a weight or a bias.
[Chapter 6](06-convolution.md).

**Pooling.** Reduction of spatial resolution by taking the maximum or the mean over
windows. [Chapter 7](07-normalization-activation-pooling.md).

**Precision.** Fraction of positive predictions that are correct.
[Chapter 17](17-evaluation-metrics.md).

**Quantisation.** Representing weights and activations with fewer bits, typically 8.
[Chapter 21](21-on-device-inference.md).

**Recall.** Fraction of actual positives that are found.
[Chapter 17](17-evaluation-metrics.md).

**Receptive field.** Region of the input that can influence a given unit.
[Chapter 6](06-convolution.md).

**Regularisation.** Any technique that reduces the gap between training and validation
error. [Chapter 15](15-regularization-and-augmentation.md).

**Residual connection.** Addition of a block's input to its output.
[Chapter 10](10-residual-and-dense-blocks.md).

**Sampling rate.** Number of samples per second of audio.
[Chapter 1](01-digital-audio.md).

**Seed.** Initial state of a pseudo-random generator.
[Chapter 19](19-reproducibility.md).

**Shortcut learning.** Reliance on an accidental feature of the dataset instead of the
intended concept. [Chapter 18](18-generalization-and-data-leakage.md).

**Sigmoid.** Function $1 / (1 + e^{-z})$ mapping a logit to a probability.
[Chapter 11](11-loss-functions.md).

**SpecAugment.** Augmentation that masks frequency bands and time spans of a
spectrogram. [Chapter 15](15-regularization-and-augmentation.md).

**Spectrogram.** Time-frequency image of a signal, obtained with the STFT.
[Chapter 2](02-spectrograms.md).

**State dict.** Dictionary of a module's parameters and buffers.
[Chapter 16](16-training-loop.md).

**STFT.** Short-time Fourier transform: Fourier transforms of successive windowed
frames. [Chapter 2](02-spectrograms.md).

**Stride.** Step with which a kernel moves over its input.
[Chapter 6](06-convolution.md).

**Tensor.** n-dimensional array with a shape, a data type and a device.
[Chapter 4](04-tensors-and-shapes.md).

**Test set.** Data used once, for the final estimate of performance.
[Chapter 18](18-generalization-and-data-leakage.md).

**Validation set.** Data used to make choices during development.
[Chapter 18](18-generalization-and-data-leakage.md).

**Weight decay.** Shrinking of the weights at each step, to limit overfitting.
[Chapter 13](13-optimizers.md).
