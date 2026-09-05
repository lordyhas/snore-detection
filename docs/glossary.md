# Glossary

Terms are listed alphabetically, with the section where they are explained.

**Activation function.** Non-linear function applied after a layer, such as ReLU.
[Section 2.2](02-convolutional-networks.md#22-normalisation-activation-and-pooling).

**Adam, AdamW.** Optimisers with a per-parameter adaptive step; AdamW applies weight
decay separately from the adaptive step. [Section 3.3](03-training.md#33-optimisers).

**Aliasing.** Frequencies above the Nyquist limit folded onto lower ones when sampling.
[Section 1.1](01-audio-and-tensors.md#11-digital-audio).

**Augmentation.** Random label-preserving transformation of training examples.
[Section 3.5](03-training.md#35-regularisation-and-augmentation).

**Autograd.** PyTorch's automatic differentiation engine.
[Section 3.2](03-training.md#32-autograd-and-backpropagation).

**Backpropagation.** Computation of all gradients by applying the chain rule from the
output backwards. [Section 3.2](03-training.md#32-autograd-and-backpropagation).

**Batch (mini-batch).** Group of examples processed together for one gradient step.
[Section 1.5](01-audio-and-tensors.md#15-datasets-and-data-loaders).

**Batch normalisation.** Layer that standardises activations per channel, with batch
statistics in training and running statistics in evaluation.
[Section 2.2](02-convolutional-networks.md#22-normalisation-activation-and-pooling).

**Binary cross-entropy.** Loss for two-class problems; the negative log-likelihood of a
Bernoulli model. [Section 3.1](03-training.md#31-loss-functions).

**Broadcasting.** Automatic expansion of dimensions of size 1 in element-wise
operations. [Section 1.4](01-audio-and-tensors.md#14-tensors-and-shapes).

**Buffer.** Tensor that belongs to a module without being a learned parameter, such as
running statistics or a mel filter bank. [Section 1.2](01-audio-and-tensors.md#12-spectrograms).

**Checkpoint.** File holding the weights of a model and the information needed to use
them. [Section 3.6](03-training.md#36-anatomy-of-the-training-loop).

**Confusion matrix.** Table of true and false positives and negatives.
[Section 4.1](04-evaluation.md#41-evaluation-metrics).

**Convolution.** Layer that applies the same small kernel at every position of its
input. [Section 2.1](02-convolutional-networks.md#21-convolution).

**Data leakage.** Information from the evaluation data reaching the training process.
[Section 4.2](04-evaluation.md#42-generalisation-and-data-leakage).

**Decibel (dB).** Logarithmic unit of power ratio: $10 \log_{10}$ of the ratio.
[Section 1.3](01-audio-and-tensors.md#13-mel-and-log-scales).

**Depthwise separable convolution.** A per-channel spatial convolution followed by a
1×1 convolution. [Section 2.4](02-convolutional-networks.md#24-depthwise-separable-convolutions).

**Distribution shift.** Difference between the data a model was trained on and the data
it meets in use. [Section 4.2](04-evaluation.md#42-generalisation-and-data-leakage).

**Dropout.** Random zeroing of activations during training.
[Section 3.5](03-training.md#35-regularisation-and-augmentation).

**Early stopping.** Ending training when the validation loss stops improving and keeping
the best weights. [Section 3.4](03-training.md#34-learning-rate-schedules-and-early-stopping).

**Epoch.** One pass over the training set. [Section 1.5](01-audio-and-tensors.md#15-datasets-and-data-loaders).

**F1 score.** Harmonic mean of precision and recall.
[Section 4.1](04-evaluation.md#41-evaluation-metrics).

**Generalisation.** Performance on data not used for training.
[Section 4.2](04-evaluation.md#42-generalisation-and-data-leakage).

**Global average pooling.** Averaging each channel over all positions.
[Section 2.2](02-convolutional-networks.md#22-normalisation-activation-and-pooling).

**Gradient.** Vector of partial derivatives of the loss with respect to the parameters.
[Section 3.2](03-training.md#32-autograd-and-backpropagation).

**Hop length.** Number of samples between two consecutive STFT frames.
[Section 1.2](01-audio-and-tensors.md#12-spectrograms).

**Hyperparameter.** Setting chosen by the practitioner rather than learned: learning
rate, batch size, number of layers. [Section 3.3](03-training.md#33-optimisers).

**Kernel.** The small array of weights of a convolution. [Section 2.1](02-convolutional-networks.md#21-convolution).

**Learning rate.** Size of the step taken along the gradient.
[Section 3.3](03-training.md#33-optimisers).

**Logit.** Raw output of the model before the sigmoid; the log-odds.
[Section 3.1](03-training.md#31-loss-functions).

**Loss.** Function measuring the error of the model, minimised by training.
[Section 3.1](03-training.md#31-loss-functions).

**Mel scale.** Perceptual frequency scale, fine at low frequencies and coarse at high
ones. [Section 1.3](01-audio-and-tensors.md#13-mel-and-log-scales).

**Momentum.** Running average of past gradients used as the update direction.
[Section 3.3](03-training.md#33-optimisers).

**Nyquist frequency.** Half the sampling rate; the highest representable frequency.
[Section 1.1](01-audio-and-tensors.md#11-digital-audio).

**ONNX.** Portable file format for neural networks.
[Section 5.1](05-deployment.md#51-exporting-to-onnx).

**Overfitting.** Fitting the training data, noise included, at the expense of
generalisation. [Section 3.5](03-training.md#35-regularisation-and-augmentation).

**Parameter.** Learned number of the model; a weight or a bias.
[Section 2.1](02-convolutional-networks.md#21-convolution).

**Pooling.** Reduction of spatial resolution by taking the maximum or the mean over
windows. [Section 2.2](02-convolutional-networks.md#22-normalisation-activation-and-pooling).

**Precision.** Fraction of positive predictions that are correct.
[Section 4.1](04-evaluation.md#41-evaluation-metrics).

**Quantisation.** Representing weights and activations with fewer bits, typically 8.
[Section 5.2](05-deployment.md#52-on-device-inference).

**Recall.** Fraction of actual positives that are found.
[Section 4.1](04-evaluation.md#41-evaluation-metrics).

**Receptive field.** Region of the input that can influence a given unit.
[Section 2.1](02-convolutional-networks.md#21-convolution).

**Regularisation.** Any technique that reduces the gap between training and validation
error. [Section 3.5](03-training.md#35-regularisation-and-augmentation).

**Residual connection.** Addition of a block's input to its output.
[Section 2.5](02-convolutional-networks.md#25-residual-and-dense-blocks).

**Sampling rate.** Number of samples per second of audio.
[Section 1.1](01-audio-and-tensors.md#11-digital-audio).

**Seed.** Initial state of a pseudo-random generator.
[Section 4.3](04-evaluation.md#43-reproducibility).

**Shortcut learning.** Reliance on an accidental feature of the dataset instead of the
intended concept. [Section 4.2](04-evaluation.md#42-generalisation-and-data-leakage).

**Sigmoid.** Function $1 / (1 + e^{-z})$ mapping a logit to a probability.
[Section 3.1](03-training.md#31-loss-functions).

**SpecAugment.** Augmentation that masks frequency bands and time spans of a
spectrogram. [Section 3.5](03-training.md#35-regularisation-and-augmentation).

**Spectrogram.** Time-frequency image of a signal, obtained with the STFT.
[Section 1.2](01-audio-and-tensors.md#12-spectrograms).

**State dict.** Dictionary of a module's parameters and buffers.
[Section 3.6](03-training.md#36-anatomy-of-the-training-loop).

**STFT.** Short-time Fourier transform: Fourier transforms of successive windowed
frames. [Section 1.2](01-audio-and-tensors.md#12-spectrograms).

**Stride.** Step with which a kernel moves over its input.
[Section 2.1](02-convolutional-networks.md#21-convolution).

**Tensor.** n-dimensional array with a shape, a data type and a device.
[Section 1.4](01-audio-and-tensors.md#14-tensors-and-shapes).

**Test set.** Data used once, for the final estimate of performance.
[Section 4.2](04-evaluation.md#42-generalisation-and-data-leakage).

**Validation set.** Data used to make choices during development.
[Section 4.2](04-evaluation.md#42-generalisation-and-data-leakage).

**Weight decay.** Shrinking of the weights at each step, to limit overfitting.
[Section 3.3](03-training.md#33-optimisers).

---

[Contents](README.md)
