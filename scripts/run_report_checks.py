"""Run reproducible implementation and attack-strength checks for the report.

The primary checkpoint curves remain full-CIFAR-10, single-restart PGD-20
measurements.  This script intentionally performs a smaller, fixed stratified
diagnostic: it verifies DCT algebra and perturbation bounds, measures leakage
introduced by final image clipping, and compares the primary pixel attack with
multi-restart PGD-50 at selected checkpoints.  Results are written separately
so the primary evaluation CSVs are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.preact_resnet import PreActResNet18
from scripts.dct_pgd import (
    dct_2d,
    generate_low_frequency_dct_pgd,
    idct_2d,
    low_frequency_project,
    out_of_mask_energy_fraction,
)
from scripts.evaluate import Normalizer, generate_pgd_adversarial


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)


def device_for_checks() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fixed_stratified_subset(dataset, examples: int, seed: int) -> Subset:
    """Select an approximately class-balanced, reproducible CIFAR-10 subset."""
    if examples < 10:
        raise ValueError("examples must be at least 10 to retain all CIFAR-10 classes.")
    labels = np.asarray(dataset.targets)
    rng = np.random.default_rng(seed)
    per_class, remainder = divmod(examples, 10)
    selected = []
    for label in range(10):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        selected.extend(class_indices[:per_class + (label < remainder)].tolist())
    rng.shuffle(selected)
    return Subset(dataset, selected)


def load_model(path: Path, device: torch.device) -> PreActResNet18:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PreActResNet18(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def dct_algebra_checks() -> dict[str, float | bool]:
    torch.manual_seed(20260912)
    samples = torch.randn(8, 3, 32, 32)
    recovered = idct_2d(dct_2d(samples))
    projected = low_frequency_project(samples, cutoff=8)
    return {
        "dct_roundtrip_max_abs_error": float((samples - recovered).abs().max()),
        "projected_out_of_mask_energy_max": float(
            out_of_mask_energy_fraction(projected, cutoff=8).max()
        ),
        "roundtrip_pass": bool(torch.allclose(samples, recovered, atol=1e-5, rtol=1e-5)),
        "projection_pass": bool(
            out_of_mask_energy_fraction(projected, cutoff=8).max() < 1e-10
        ),
    }


def evaluate_pixel_attacks(
    model: torch.nn.Module,
    normalizer: Normalizer,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    epsilon: float,
    alpha: float,
    primary_steps: int,
    strong_steps: int,
    restarts: int,
) -> dict[str, float]:
    """Compare one PGD-20 start with the highest-loss PGD-50 restart candidate."""
    primary_correct = strong_correct = total = 0
    primary_loss = strong_loss = 0.0
    final_linf_max = 0.0

    for batch_index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        total += labels.numel()

        torch.manual_seed(seed + batch_index)
        with torch.enable_grad():
            primary_adv = generate_pgd_adversarial(
                model, normalizer, images, labels, epsilon, alpha, primary_steps, device
            )
        with torch.no_grad():
            primary_outputs = model(normalizer(primary_adv))
            primary_batch_losses = F.cross_entropy(primary_outputs, labels, reduction="none")
            primary_loss += primary_batch_losses.sum().item()
            primary_correct += primary_outputs.argmax(dim=1).eq(labels).sum().item()

        strongest_losses = None
        strongest_outputs = None
        for restart in range(restarts):
            torch.manual_seed(seed + 100_000 * (restart + 1) + batch_index)
            with torch.enable_grad():
                candidate = generate_pgd_adversarial(
                    model, normalizer, images, labels, epsilon, alpha, strong_steps, device
                )
            with torch.no_grad():
                candidate_outputs = model(normalizer(candidate))
                candidate_losses = F.cross_entropy(
                    candidate_outputs, labels, reduction="none"
                )
                candidate_linf = (candidate - images).abs().amax(dim=(1, 2, 3))
                final_linf_max = max(final_linf_max, candidate_linf.max().item())
                if strongest_losses is None:
                    strongest_losses = candidate_losses
                    strongest_outputs = candidate_outputs
                else:
                    choose_candidate = candidate_losses > strongest_losses
                    strongest_losses = torch.where(
                        choose_candidate, candidate_losses, strongest_losses
                    )
                    strongest_outputs = torch.where(
                        choose_candidate[:, None], candidate_outputs, strongest_outputs
                    )

        strong_loss += strongest_losses.sum().item()
        strong_correct += strongest_outputs.argmax(dim=1).eq(labels).sum().item()

    return {
        "examples": total,
        "primary_accuracy": primary_correct / total,
        "primary_loss": primary_loss / total,
        "strong_accuracy": strong_correct / total,
        "strong_loss": strong_loss / total,
        "strong_final_linf_max": final_linf_max,
    }


def evaluate_clipping_leakage(
    model: torch.nn.Module,
    normalizer: Normalizer,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    epsilon: float,
    alpha: float,
    steps: int,
) -> dict[str, float]:
    final_leakage, preclip_leakage = [], []
    final_linf = []
    for batch_index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        torch.manual_seed(seed + batch_index)
        with torch.enable_grad():
            _, metadata = generate_low_frequency_dct_pgd(
                model,
                normalizer,
                images,
                labels,
                epsilon,
                alpha,
                steps,
                cutoff=8,
                return_metadata=True,
            )
        final_leakage.extend(metadata["final_out_of_mask_energy_fraction"].cpu().tolist())
        preclip_leakage.extend(metadata["preclip_out_of_mask_energy_fraction"].cpu().tolist())
        final_linf.extend(metadata["final_linf"].cpu().tolist())

    return {
        "examples": len(final_leakage),
        "preclip_leakage_mean": float(np.mean(preclip_leakage)),
        "preclip_leakage_max": float(np.max(preclip_leakage)),
        "final_leakage_mean": float(np.mean(final_leakage)),
        "final_leakage_p95": float(np.quantile(final_leakage, 0.95)),
        "final_leakage_max": float(np.max(final_leakage)),
        "final_linf_max": float(np.max(final_linf)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--primary-steps", type=int, default=20)
    parser.add_argument("--strong-steps", type=int, default=50)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument(
        "--pixel-checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="selected checkpoint for pixel attack comparison; may be supplied multiple times",
    )
    parser.add_argument(
        "--leakage-checkpoint",
        type=Path,
        default=Path("checkpoints/low-frequency-only/seed-42/epoch_200.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("report/checks")
    )
    args = parser.parse_args()

    if not args.pixel_checkpoint:
        args.pixel_checkpoint = [
            "pixel-seed42-peak=checkpoints/pixel-only/seed-42/epoch_105.pt",
            "pixel-seed42-final=checkpoints/pixel-only/seed-42/epoch_200.pt",
        ]
    checkpoint_specs = []
    for spec in args.pixel_checkpoint:
        if "=" not in spec:
            parser.error("--pixel-checkpoint must have LABEL=PATH form")
        label, raw_path = spec.split("=", maxsplit=1)
        checkpoint_specs.append((label, Path(raw_path)))

    for _, checkpoint_path in checkpoint_specs:
        if not checkpoint_path.is_file():
            parser.error(f"checkpoint not found: {checkpoint_path}")
    if not args.leakage_checkpoint.is_file():
        parser.error(f"leakage checkpoint not found: {args.leakage_checkpoint}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_for_checks()
    dataset = datasets.CIFAR10(root="data", train=False, download=False, transform=transforms.ToTensor())
    subset = fixed_stratified_subset(dataset, args.examples, args.seed)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    normalizer = Normalizer(MEAN, STD).to(device)
    epsilon, alpha = 8.0 / 255.0, 2.0 / 255.0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    implementation = dct_algebra_checks()
    implementation.update({
        "device": str(device),
        "torch_version": torch.__version__,
        "examples": args.examples,
        "subset_seed": args.seed,
    })

    rows = []
    for check_index, (label, checkpoint_path) in enumerate(checkpoint_specs):
        model = load_model(checkpoint_path, device)
        start = perf_counter()
        metrics = evaluate_pixel_attacks(
            model, normalizer, loader, device, args.seed + check_index,
            epsilon, alpha, args.primary_steps, args.strong_steps, args.restarts,
        )
        metrics.update({
            "label": label,
            "checkpoint": str(checkpoint_path),
            "primary_steps": args.primary_steps,
            "strong_steps": args.strong_steps,
            "strong_restarts": args.restarts,
            "elapsed_seconds": perf_counter() - start,
        })
        rows.append(metrics)
        print(f"{label}: PGD-{args.primary_steps}={metrics['primary_accuracy']:.2%}; "
              f"PGD-{args.strong_steps} x {args.restarts}={metrics['strong_accuracy']:.2%}")

    leakage_model = load_model(args.leakage_checkpoint, device)
    leakage = evaluate_clipping_leakage(
        leakage_model, normalizer, loader, device, args.seed,
        epsilon, alpha, args.primary_steps,
    )
    leakage.update({
        "checkpoint": str(args.leakage_checkpoint),
        "attack_steps": args.primary_steps,
        "epsilon": epsilon,
        "alpha": alpha,
    })

    with (args.output_dir / "attack_strength_checks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "implementation_checks.json").open("w") as handle:
        json.dump({"dct_algebra": implementation, "clipping_leakage": leakage}, handle, indent=2)

    print(f"Wrote {args.output_dir / 'attack_strength_checks.csv'}")
    print(f"Wrote {args.output_dir / 'implementation_checks.json'}")


if __name__ == "__main__":
    main()
