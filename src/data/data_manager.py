import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(
        self,
        waveforms,
        labels,
        sample_rate=16000,
        duration=1.0,
        n_mels=128,
        n_fft=1024,
        hop_length=512,
        train=False,
    ):
        self.waveforms = waveforms
        self.labels = labels
        self.num_samples = int(sample_rate * duration)
        self.train = train
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        # SpecAugment, only applied on the training set
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)

    def __len__(self):
        return len(self.waveforms)


    def __getitem__(self, idx):
        waveform = torch.as_tensor(self.waveforms[idx], dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)

        # fixed length: crop or zero-pad
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[1]))

        spec = self.to_db(self.mel(waveform))  # (1, n_mels, frames)
        if self.train:
            spec = self.time_mask(self.freq_mask(spec))

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return spec, label