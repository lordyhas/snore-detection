"""
Snore Detection - tests of the sanity checks: they pass on the real models and each
one fails on a model broken on purpose.

Author: Hassan Kajila
"""

import pytest
import torch
import torch.nn as nn

import src.models.registry as registry
import src.sanity_check as sanity_check
from src.models.audio_cnn import AudioCNN
from src.models.mobile_cnn import LogMelFrontend, SnoreMobileNet


@pytest.fixture(autouse=True)
def synthetic_data(monkeypatch, tmp_path):
    """Run on random noise, so the tests need no dataset and are the same everywhere."""
    monkeypatch.setattr(sanity_check, "PROCESSED_DATA_FILE", str(tmp_path / "missing.npz"))


def results_for(monkeypatch, model_name, factory=None):
    if factory is not None:
        monkeypatch.setitem(registry.MODELS, model_name, factory)
    return {result.name: result.passed for result in sanity_check.run_checks(model_name)}


class DetachedFeatures(AudioCNN):
    """The convolutional part is cut from the graph: it receives no gradient."""

    def forward(self, x):
        return self.classifier(self.pool(self.features(x).detach()))


class AlwaysTraining(AudioCNN):
    """`eval()` has no effect: dropout and batch statistics stay active."""

    def eval(self):
        return self.train()


class _HalvedGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 0.5


class WrongBackward(AudioCNN):
    """One operation has a backward pass that does not match its forward pass."""

    def forward(self, x):
        return self.classifier(self.pool(_HalvedGradient.apply(self.features(x))))


class BadlyScaledOutput(AudioCNN):
    """The last layer starts with huge weights: confident before any training."""

    def __init__(self):
        super().__init__()
        nn.init.normal_(self.classifier[5].weight, std=30.0)


class Frozen(AudioCNN):
    """Nothing useful can be learned: every real parameter is frozen."""

    def __init__(self):
        super().__init__()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return super().forward(x) + 0 * self.unused


class AbsoluteFloorFrontend(LogMelFrontend):
    """The first version of the front-end: a fixed floor under the logarithm."""

    def forward(self, waveform):
        stft = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, return_complex=True,
        )
        power = torch.view_as_real(stft).pow(2).sum(-1)
        mel = torch.matmul(power.transpose(1, 2), self.fbanks).transpose(1, 2)
        log_mel = torch.log(mel + 1e-6)
        centered = log_mel - log_mel.mean(dim=(1, 2), keepdim=True)
        return (centered / (centered.std(dim=(1, 2), keepdim=True) + 1e-5)).unsqueeze(1)


def gain_dependent_model():
    model = SnoreMobileNet(width=16)
    model.frontend = AbsoluteFloorFrontend()
    return model


@pytest.mark.parametrize("model_name", list(registry.MODELS))
def test_real_models_pass_every_check(monkeypatch, model_name):
    results = results_for(monkeypatch, model_name)
    assert all(results.values()), results


def test_waveform_models_get_the_extra_checks(monkeypatch):
    assert "gain invariance" not in results_for(monkeypatch, "audio_cnn")
    assert {"gain invariance", "silence"} <= set(results_for(monkeypatch, "mobile_tiny"))


@pytest.mark.parametrize(
    "factory, model_name, failing_check",
    [
        (DetachedFeatures, "audio_cnn", "gradient flow"),
        (DetachedFeatures, "audio_cnn", "overfit small batch"),
        (AlwaysTraining, "audio_cnn", "eval mode"),
        (WrongBackward, "audio_cnn", "numerical gradient"),
        (BadlyScaledOutput, "audio_cnn", "initial loss"),
        (Frozen, "audio_cnn", "overfit small batch"),
        (gain_dependent_model, "mobile_tiny", "gain invariance"),
    ],
)
def test_broken_model_is_caught(monkeypatch, factory, model_name, failing_check):
    results = results_for(monkeypatch, model_name, factory)
    assert results[failing_check] is False, results


def test_only_the_broken_part_is_reported(monkeypatch):
    results = results_for(monkeypatch, "audio_cnn", WrongBackward)
    assert [name for name, passed in results.items() if not passed] == ["numerical gradient"]


def test_exit_code_reports_failures(monkeypatch, capsys):
    assert sanity_check.main(["--model", "mobile_tiny"]) == 0
    monkeypatch.setitem(registry.MODELS, "mobile_tiny", gain_dependent_model)
    assert sanity_check.main(["--model", "mobile_tiny"]) == 1
    assert "[FAIL] gain invariance" in capsys.readouterr().out
