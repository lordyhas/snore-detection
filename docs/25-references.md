# 25. References

## Textbooks

- Goodfellow, I., Bengio, Y. and Courville, A. (2016). *Deep Learning*. MIT Press.
  The standard reference; freely readable at deeplearningbook.org. Chapters 6 to 9 cover
  most of this documentation.
- Zhang, A., Lipton, Z., Li, M. and Smola, A. *Dive into Deep Learning*. An interactive
  book with runnable PyTorch code, at d2l.ai.
- Bishop, C. (2006). *Pattern Recognition and Machine Learning*. Springer. For the
  statistical foundations.
- Oppenheim, A. and Schafer, R. (2009). *Discrete-Time Signal Processing*, 3rd edition.
  Pearson. For sampling and the Fourier transform.
- Müller, M. (2015). *Fundamentals of Music Processing*. Springer. An accessible
  treatment of the STFT and of audio features.

## Papers, by chapter

### Audio features (chapters 1 to 3)

- Stevens, S., Volkmann, J. and Newman, E. (1937). A scale for the measurement of the
  psychological magnitude pitch. *Journal of the Acoustical Society of America*, 8(3).
- Davis, S. and Mermelstein, P. (1980). Comparison of parametric representations for
  monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on
  Acoustics, Speech, and Signal Processing*, 28(4).

### Architectures (chapters 6 to 10)

- LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P. (1998). Gradient-based learning
  applied to document recognition. *Proceedings of the IEEE*, 86(11).
- Simonyan, K. and Zisserman, A. (2015). Very deep convolutional networks for
  large-scale image recognition. *ICLR*.
- Ioffe, S. and Szegedy, C. (2015). Batch normalization: accelerating deep network
  training by reducing internal covariate shift. *ICML*.
- He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for image
  recognition. *CVPR*.
- Huang, G., Liu, Z., van der Maaten, L. and Weinberger, K. (2017). Densely connected
  convolutional networks. *CVPR*.
- Howard, A. et al. (2017). MobileNets: efficient convolutional neural networks for
  mobile vision applications. arXiv:1704.04861.
- Dumoulin, V. and Visin, F. (2016). A guide to convolution arithmetic for deep
  learning. arXiv:1603.07285.

### Audio classification with CNNs

- Hershey, S. et al. (2017). CNN architectures for large-scale audio classification.
  *ICASSP*.
- Zhang, Y., Suda, N., Lai, L. and Chandra, V. (2017). Hello Edge: keyword spotting on
  microcontrollers. arXiv:1711.07128.

### Optimisation (chapters 11 to 14)

- Rumelhart, D., Hinton, G. and Williams, R. (1986). Learning representations by
  back-propagating errors. *Nature*, 323.
- Kingma, D. and Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR*.
- Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
- Loshchilov, I. and Hutter, F. (2017). SGDR: stochastic gradient descent with warm
  restarts. *ICLR*.
- Prechelt, L. (1998). Early stopping - but when? In *Neural Networks: Tricks of the
  Trade*. Springer.

### Regularisation (chapter 15)

- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R.
  (2014). Dropout: a simple way to prevent neural networks from overfitting. *Journal of
  Machine Learning Research*, 15.
- Park, D. et al. (2019). SpecAugment: a simple data augmentation method for automatic
  speech recognition. *Interspeech*.
- Zhang, H., Cisse, M., Dauphin, Y. and Lopez-Paz, D. (2018). mixup: beyond empirical
  risk minimization. *ICLR*.

### Evaluation and methodology (chapters 17 to 19)

- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*,
  27(8).
- Geirhos, R. et al. (2020). Shortcut learning in deep neural networks. *Nature Machine
  Intelligence*, 2.
- Kapoor, S. and Narayanan, A. (2023). Leakage and the reproducibility crisis in
  machine-learning-based science. *Patterns*, 4(9).
- Bouthillier, X. et al. (2021). Accounting for variance in machine learning benchmarks.
  *MLSys*.

### Deployment (chapters 20 and 21)

- Jacob, B. et al. (2018). Quantization and training of neural networks for efficient
  integer-arithmetic-only inference. *CVPR*.
- Hinton, G., Vinyals, O. and Dean, J. (2015). Distilling the knowledge in a neural
  network. arXiv:1503.02531.

### Practice (chapter 22)

- Karpathy, A. (2019). A recipe for training neural networks. Blog post,
  karpathy.github.io.

## Software documentation

- PyTorch: pytorch.org/docs, in particular the notes "Autograd mechanics",
  "Broadcasting semantics" and "Reproducibility", and the tutorials "Learn the Basics".
- torchaudio: pytorch.org/audio, transforms `MelSpectrogram`, `AmplitudeToDB`,
  `FrequencyMasking`, `TimeMasking`.
- librosa: librosa.org/doc.
- scikit-learn: scikit-learn.org, user guide sections "Model evaluation" and
  "Cross-validation".
- ONNX: onnx.ai; ONNX Runtime: onnxruntime.ai.

## Dataset

- Snoring dataset, 500 snoring and 500 non-snoring one-second clips, obtained from
  Kaggle (see the link in the main `README.md`).
