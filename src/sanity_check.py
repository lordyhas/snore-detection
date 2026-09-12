"""
Snore Detection - sanity checks to run on a model before trusting its training.

Author: Hassan Kajila
"""

import argparse
import copy
import logging
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

from src.config import PROCESSED_DATA_FILE, RANDOM_SEED
from src.data.data_manager import AudioDataset
from src.models.registry import MODELS, WAVEFORM_MODELS, build_model


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DURATION = 1.0


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def load_examples(model_name: str, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `count` class-balanced examples in the input format of the model.

    Real validation clips are used when the preprocessed dataset exists; otherwise random
    noise with random labels, which is enough for every check (a network can memorise
    noise) and lets the script run without the data.
    """
    if os.path.isfile(PROCESSED_DATA_FILE):
        with np.load(PROCESSED_DATA_FILE, allow_pickle=False) as data:
            waveforms, labels = data["x_val"], data["y_val"]
            sample_rate, duration = int(data["sample_rate"]), float(data["duration"])
        # skip silent clips: they carry no signal to check anything with
        audible = np.abs(waveforms).max(axis=1) > 0
        negatives = np.flatnonzero((labels == 0) & audible)[: count // 2]
        positives = np.flatnonzero((labels == 1) & audible)[: count - count // 2]
        indices = np.concatenate([negatives, positives])
        waveforms, labels = waveforms[indices], labels[indices]
    else:
        logger.warning("%s not found: using random noise.", PROCESSED_DATA_FILE)
        generator = np.random.default_rng(RANDOM_SEED)
        sample_rate, duration = SAMPLE_RATE, DURATION
        waveforms = generator.normal(0, 0.1, (count, int(sample_rate * duration)))
        waveforms = waveforms.astype(np.float32)
        labels = np.arange(count) % 2

    dataset = AudioDataset(
        waveforms,
        labels,
        sample_rate=sample_rate,
        duration=duration,
        train=False,
        features=model_name not in WAVEFORM_MODELS,
    )
    inputs, targets = zip(*(dataset[index] for index in range(len(dataset))))
    return torch.stack(inputs), torch.stack(targets).view(-1, 1)


def _loss(model, inputs, targets) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(model(inputs), targets)


def check_output(model_name, inputs, targets) -> CheckResult:
    """The model returns one finite logit per example, for any input length."""
    model = build_model(model_name).eval()
    with torch.no_grad():
        logits = model(inputs)
        longer = model(torch.cat([inputs, inputs], dim=-1))
    expected = (inputs.size(0), 1)
    passed = (
        tuple(logits.shape) == expected
        and tuple(longer.shape) == expected
        and bool(torch.isfinite(logits).all())
        and bool(torch.isfinite(longer).all())
    )
    return CheckResult(
        "output shape",
        passed,
        f"{tuple(inputs.shape)} -> {tuple(logits.shape)}, "
        f"twice longer input -> {tuple(longer.shape)}",
    )


def check_initial_loss(model_name, inputs, targets, tolerance=0.1) -> CheckResult:
    """An untrained binary classifier should be undecided: loss close to log(2)."""
    losses = []
    for seed in range(5):
        torch.manual_seed(RANDOM_SEED + seed)
        model = build_model(model_name).eval()
        with torch.no_grad():
            losses.append(float(_loss(model, inputs, targets)))
    loss = sum(losses) / len(losses)
    return CheckResult(
        "initial loss",
        abs(loss - math.log(2)) <= tolerance,
        f"{loss:.4f} over 5 initialisations (expected {math.log(2):.4f} +/- {tolerance})",
    )


def check_gradient_flow(model_name, inputs, targets) -> CheckResult:
    """After one backward pass, every parameter has a finite, non-zero gradient."""
    model = build_model(model_name).train()
    _loss(model, inputs, targets).backward()
    norms = {
        name: float(parameter.grad.norm()) if parameter.grad is not None else float("nan")
        for name, parameter in model.named_parameters()
    }
    broken = [name for name, norm in norms.items() if not norm > 0 or math.isinf(norm)]
    detail = (
        f"{len(norms)} tensors, gradient norms from "
        f"{min(norms.values()):.2e} to {max(norms.values()):.2e}"
    )
    if broken:
        detail = f"no usable gradient for: {', '.join(broken[:3])}"
    return CheckResult("gradient flow", not broken, detail)


def check_numerical_gradient(
    model_name, inputs, targets, samples=20, steps=(1e-5, 1e-6, 1e-7), tolerance=1e-4
) -> CheckResult:
    """Backpropagation agrees with finite differences on randomly chosen weights."""
    # double precision: in float32 the finite difference is dominated by rounding
    model = build_model(model_name).double().eval()
    inputs, targets = inputs[:4].double(), targets[:4].double()
    with torch.no_grad():
        # move away from the initialisation: with biases at exactly zero, many
        # activations sit exactly on the kink of the ReLU, where the derivative used by
        # backpropagation and the slope measured by a finite difference differ
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    _loss(model, inputs, targets).backward()

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    parameters = list(model.parameters())
    worst = 0.0
    with torch.no_grad():
        for _ in range(samples):
            parameter = parameters[int(torch.randint(len(parameters), (1,), generator=generator))]
            flat, gradient = parameter.view(-1), parameter.grad.view(-1)
            index = int(torch.randint(flat.numel(), (1,), generator=generator))
            original, analytical = float(flat[index]), float(gradient[index])
            errors = []
            for step in steps:
                flat[index] = original + step
                loss_plus = float(_loss(model, inputs, targets))
                flat[index] = original - step
                loss_minus = float(_loss(model, inputs, targets))
                flat[index] = original
                numerical = (loss_plus - loss_minus) / (2 * step)
                errors.append(
                    abs(numerical - analytical) / max(1e-6, abs(numerical) + abs(analytical))
                )
            # a large step may cross a ReLU kink and a small one suffers from rounding:
            # each spoils only some step sizes, whereas a wrong gradient is wrong for all
            worst = max(worst, min(errors))
    return CheckResult(
        "numerical gradient",
        worst <= tolerance,
        f"max relative error {worst:.2e} on {samples} weights (limit {tolerance:.0e})",
    )


def check_overfit(model_name, inputs, targets, max_steps=300, device=None) -> CheckResult:
    """The model can memorise a handful of examples: 100 % accuracy on them."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    accuracy, loss, step = 0.0, float("nan"), 0
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _loss(model, inputs, targets).backward()
        optimizer.step()
        if step % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(inputs)
                loss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, targets))
                accuracy = float(((logits >= 0) == (targets >= 0.5)).float().mean())
            if accuracy == 1.0 and loss < 0.1:
                break
    return CheckResult(
        "overfit small batch",
        accuracy == 1.0 and loss < 0.1,
        f"accuracy {accuracy:.2f}, loss {loss:.4f} on {inputs.size(0)} examples "
        f"after {step} steps",
    )


COMMON_CHECKS = [
    check_output,
    check_initial_loss,
    check_gradient_flow,
    check_numerical_gradient,
    check_overfit,
]
WAVEFORM_CHECKS = []


def run_checks(model_name: str, batch_size: int = 16) -> list[CheckResult]:
    """Run every check that applies to the model and return the results."""
    inputs, targets = load_examples(model_name, batch_size)
    checks = COMMON_CHECKS + (WAVEFORM_CHECKS if model_name in WAVEFORM_MODELS else [])
    results = []
    for check in checks:
        torch.manual_seed(RANDOM_SEED)
        try:
            results.append(check(model_name, inputs, copy.deepcopy(targets)))
        except Exception as error:  # a crash is a failed check, not a failed run
            name = check.__name__.removeprefix("check_").replace("_", " ")
            results.append(CheckResult(name, False, f"{type(error).__name__}: {error}"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run sanity checks on the models.")
    parser.add_argument(
        "--model", dest="model_name", choices=["all", *MODELS], default="all"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    names = list(MODELS) if args.model_name == "all" else [args.model_name]
    failures = 0
    for model_name in names:
        print(f"\n{model_name}")
        for result in run_checks(model_name, args.batch_size):
            failures += not result.passed
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.name:<20} {result.detail}")
    print(f"\n{failures} check(s) failed." if failures else "\nAll checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
