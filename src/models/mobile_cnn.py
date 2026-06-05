"""
Snore Detection - lightweight waveform-in classifiers for smartphones.

Author: Hassan Kajila
"""

import torch
import torch.nn as nn
import torchaudio


class LogMelFrontend(nn.Module):
    """Waveform -> normalised log-mel spectrogram, kept inside the model.

    The exported model therefore takes raw audio, and the app needs no signal processing.
    Each clip is normalised on its own, so the result does not depend on microphone gain.
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, n_mels=40):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        fbanks = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=50.0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.register_buffer("fbanks", fbanks, persistent=False)  # (n_freqs, n_mels)

    def forward(self, waveform):
        # waveform: (batch, samples)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        power = torch.view_as_real(stft).pow(2).sum(-1)  # (batch, n_freqs, frames)
        mel = torch.matmul(power.transpose(1, 2), self.fbanks).transpose(1, 2)
        log_mel = torch.log(mel + 1e-6)
        mean = log_mel.mean(dim=(1, 2), keepdim=True)
        std = log_mel.std(dim=(1, 2), keepdim=True)
        return ((log_mel - mean) / (std + 1e-5)).unsqueeze(1)  # (batch, 1, n_mels, frames)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                groups=in_channels, bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SnoreMobileNet(nn.Module):
    '''
    Input (samples,) waveform at 16 kHz, values in [-1, 1]
      -> LogMelFrontend (1, 40, T)
      -> Conv 3x3 stride 2
      -> 4x depthwise separable blocks (two of them stride 2)
      -> AdaptiveAvgPool (global)
      -> Dropout -> Linear 1
    Output: 1 logit
    '''

    def __init__(self, width=32, dropout=0.2, sample_rate=16000):
        super().__init__()
        self.frontend = LogMelFrontend(sample_rate=sample_rate)
        # SpecAugment, only applied in training mode
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=6)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=15)
        self.features = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(width, width * 2, stride=2),
            DepthwiseSeparableBlock(width * 2, width * 2),
            DepthwiseSeparableBlock(width * 2, width * 4, stride=2),
            DepthwiseSeparableBlock(width * 4, width * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, 1),
        )

    def forward(self, waveform):
        # waveform: (batch, samples)
        x = self.frontend(waveform)
        if self.training:
            x = self.time_mask(self.freq_mask(x))
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)  # raw logits, no sigmoid
