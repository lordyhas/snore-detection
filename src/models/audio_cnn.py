"""
Snore Detection - CNN classifier for mel spectrograms.

Author: Hassan Kajila
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class AudioCNN(nn.Module):
    def __init__(self, in_channels=1, num_outputs=1, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # works with any input size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_outputs),
        )

    '''
    Input (1, 128, T)
      -> ConvBlock 32   (2x Conv-BN-ReLU + MaxPool)
      -> ConvBlock 64
      -> ConvBlock 128
      -> ConvBlock 256
      -> AdaptiveAvgPool (global)
      -> Dropout -> Linear 64 -> ReLU -> Dropout -> Linear 1
    Output: 1 logit
    '''

    def forward(self, x):
        # x: (batch, 1, n_mels, frames)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)  # raw logits, no sigmoid


