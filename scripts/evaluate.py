# File Name: evaluate.py
# Last Updated: July 29, 2026
# Description:
#   This script measures how well a trained model is doing at each saved
#   checkpoint. For every checkpoint, it checks accuracy on normal
#   (clean) test images, pixel-space PGD-20 images, and low-frequency
#   DCT-masked PGD-20 images. It also reports per-image union robustness:
#   a test image counts as robust only when it resists both attacks.
#   Results for every checkpoint get written out to a CSV file.
# References:
#   * Rice, L., Wong, E., and Kolter, J. Z. (2020). Overfitting in
#     adversarially robust deep learning. ICML.
#   * Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A.
#     (2018). Towards Deep Learning Models Resistant to Adversarial
#     Attacks. ICLR. (defines the PGD attack used here)

import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import ssl
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.preact_resnet import PreActResNet18
from scripts.dct_pgd import generate_low_frequency_dct_pgd

# Skips SSL certificate checks so the CIFAR-10 download doesn't fail on
# machines with outdated or misconfigured certificates.
ssl._create_default_https_context = ssl._create_unverified_context


class Normalizer(nn.Module):
    """Rescales images using CIFAR-10's known mean and standard deviation.

    Models train and evaluate better on normalized data. This is kept as
    its own step (instead of baking it into the dataset transform) so the
    raw, unnormalized image can still be used directly when generating
    the adversarial attack.
    """
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x):
        return (x - self.mean) / self.std


def generate_pgd_adversarial(model, normalizer, X, y, epsilon, alpha, num_steps, device):
    """Creates adversarial (attacked) versions of a batch of images.

    This is the PGD attack, and it's the standard method for testing
    adversarial robustness. How it works:
    1. Start from the original image plus a small random change.
    2. Run the image through the model and see how wrong it is (the loss).
    3. Change the pixels a bit further in the direction that makes the model
       even more wrong.
    4. Keep the change small and repeat this for a set number of steps.

    The result still looks like the original image but is designed to
    fool the model.
    """
    was_training = model.training
    model.eval()
    try:
        # Start from a small random perturbation instead of zero. This helps
        # avoid a weaker, less realistic attack and is standard practice for PGD.
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).to(device)
        delta = torch.clamp(X + delta, min=0.0, max=1.0) - X

        for _ in range(num_steps):
            delta = delta.detach().requires_grad_(True)
            outputs = model(normalizer(X + delta))
            loss = F.cross_entropy(outputs, y)
            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]

            # Step 3: move in the direction that increases the loss (the sign
            # of the gradient).
            delta = delta.detach() + alpha * grad.sign()
            # Step 4: keep the change within the allowed budget and make sure
            # the image stays a valid pixel value.
            delta = torch.clamp(delta, min=-epsilon, max=epsilon)
            delta = torch.clamp(X + delta, min=0.0, max=1.0) - X
    finally:
        model.train(was_training)

    return (X + delta).detach()


def evaluate_checkpoint(
    model, normalizer, dataloader, device, epsilon, alpha, num_steps,
    dct_cutoff,
):
    """Evaluate clean, pixel-PGD, low-frequency-PGD, and union robustness.

    Each test batch receives both attacks. Union robustness is computed from
    their paired per-image predictions, so an image is counted only if it is
    correctly classified after both the pixel and low-frequency attacks.
    """
    model.eval()
    clean_loss = 0.0
    clean_correct = 0
    pixel_robust_loss = 0.0
    pixel_robust_correct = 0
    low_frequency_robust_loss = 0.0
    low_frequency_robust_correct = 0
    union_robust_correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Clean predictions are evaluated before generating either attack.
        with torch.no_grad():
            batch_size = X.size(0)
            total += batch_size
            clean_outputs = model(normalizer(X))
            clean_loss += F.cross_entropy(clean_outputs, y, reduction='sum').item()
            clean_correct += clean_outputs.argmax(dim=1).eq(y).sum().item()

        # Gradients are enabled only while creating adversarial images.
        with torch.enable_grad():
            pixel_adv = generate_pgd_adversarial(
                model, normalizer, X, y,
                epsilon=epsilon,
                alpha=alpha,
                num_steps=num_steps,
                device=device
            )
            low_frequency_adv = generate_low_frequency_dct_pgd(
                model, normalizer, X, y,
                epsilon=epsilon,
                alpha=alpha,
                num_steps=num_steps,
                cutoff=dct_cutoff,
            )

        with torch.no_grad():
            pixel_outputs = model(normalizer(pixel_adv))
            low_frequency_outputs = model(normalizer(low_frequency_adv))
            pixel_robust_loss += F.cross_entropy(
                pixel_outputs, y, reduction='sum'
            ).item()
            low_frequency_robust_loss += F.cross_entropy(
                low_frequency_outputs, y, reduction='sum'
            ).item()
            pixel_correct = pixel_outputs.argmax(dim=1).eq(y)
            low_frequency_correct = low_frequency_outputs.argmax(dim=1).eq(y)
            pixel_robust_correct += pixel_correct.sum().item()
            low_frequency_robust_correct += low_frequency_correct.sum().item()
            union_robust_correct += (pixel_correct & low_frequency_correct).sum().item()

    return {
        'clean_loss': clean_loss / total,
        'clean_acc': clean_correct / total,
        'pixel_robust_loss': pixel_robust_loss / total,
        'pixel_robust_acc': pixel_robust_correct / total,
        'low_frequency_robust_loss': low_frequency_robust_loss / total,
        'low_frequency_robust_acc': low_frequency_robust_correct / total,
        'union_robust_acc': union_robust_correct / total,
    }


def extract_epoch_number(filename):
    """Pulls the epoch number out of a checkpoint filename (e.g. epoch_12.pt -> 12).

    Used to sort checkpoints in the right order and to label results in
    the output CSV.
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1


def main():
    """Runs evaluation on every checkpoint in a directory and saves results to CSV.

    Overall flow:
    1. Parse settings and set up the device (GPU or CPU) and random seed.
    2. Find and sort all checkpoint files in the given directory.
    3. Load the test dataset.
    4. For each checkpoint, in order: load its weights, evaluate on clean
       and both PGD-20 attack domains, compute union robustness, print a
       summary line, and write the result to the output CSV.
    """
    parser = argparse.ArgumentParser(description='Evaluate PreActResNet-18 Checkpoints with pixel and low-frequency PGD-20')
    parser.add_argument(
        '--training-mode', default='pixel-only',
        choices=('pixel-only', 'low-frequency-only', 'mixed-domain'),
        help='training condition being evaluated; used to choose default paths',
    )
    parser.add_argument(
        '--run-name', default=None, type=str,
        help='name for this run inside its training-mode directory (default: seed-<seed>)',
    )
    parser.add_argument('--checkpoint-dir', default=None, type=str, help='Directory containing saved .pt checkpoints')
    parser.add_argument('--output-csv', default=None, type=str, help='Path to output CSV file')
    parser.add_argument('--data-dir', default='./data', type=str, help='Dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size for evaluation')
    parser.add_argument('--num-workers', default=2, type=int, help='DataLoader worker processes when using an accelerator (default: 2)')
    parser.add_argument('--epsilon', default=8.0/255.0, type=float, help='Adversarial perturbation magnitude epsilon')
    parser.add_argument('--alpha', default=2.0/255.0, type=float, help='PGD step size alpha')
    parser.add_argument('--attack-steps', default=20, type=int, help='Number of PGD attack steps (default: 20 for PGD-20)')
    parser.add_argument('--dct-cutoff', default=8, type=int, help='side length of the retained top-left DCT low-frequency mask')
    parser.add_argument('--diagnostic', action='store_true', help='Diagnostic mode: evaluate on 10%% of test set')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    args = parser.parse_args()

    if not 1 <= args.dct_cutoff <= 32:
        parser.error('--dct-cutoff must be between 1 and 32 for CIFAR-10 images.')

    # Match train.py's condition/seed layout unless a path is explicitly
    # supplied. This keeps metrics for independent runs from overwriting.
    if args.run_name is None:
        args.run_name = f'seed-{args.seed}' if args.seed is not None else 'unseeded'
    run_parts = [args.training_mode]
    if args.diagnostic:
        run_parts.append('diagnostic')
    run_parts.append(args.run_name)
    run_relative_path = os.path.join(*run_parts)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join('checkpoints', run_relative_path)
    if args.output_csv is None:
        args.output_csv = os.path.join(
            'report', run_relative_path, 'evaluation_results.csv'
        )

    # Step 1: set up device and reproducibility.
    # Fix all random seeds so the evaluation (e.g. random PGD start point)
    # is reproducible between runs.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Prefer GPU acceleration when available, falling back to CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")

    if not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory '{args.checkpoint_dir}' does not exist.")

    # Step 2: find and sort all checkpoint files.
    ckpt_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pt') or f.endswith('.pth')]
    if not ckpt_files:
        raise FileNotFoundError(f"No .pt or .pth checkpoint files found in '{args.checkpoint_dir}'.")

    # Sort by epoch number rather than filename, so results are evaluated
    # and logged in training order.
    ckpt_files.sort(key=extract_epoch_number)

    print(f"Found {len(ckpt_files)} checkpoints in '{args.checkpoint_dir}'.")
    print(
        f"Training mode: {args.training_mode} | Run: {args.run_name} | "
        f"Results: {args.output_csv}"
    )
    print(
        f"Attack Configuration: PGD-{args.attack_steps} | "
        f"Epsilon: {args.epsilon:.4f} ({args.epsilon * 255:.1f}/255) | "
        f"Alpha: {args.alpha:.4f} ({args.alpha * 255:.1f}/255) | "
        f"DCT cutoff: {args.dct_cutoff}"
    )
    if args.diagnostic:
        print("Running in Diagnostic Mode (10% test dataset).")

    # Step 3: load the test dataset.
    # No normalization here; that happens separately via the Normalizer
    # module so the raw image is still available for generating attacks.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)

    # Diagnostic mode uses a small slice of the test set for a quick sanity
    # check, instead of a full (slower) evaluation.
    if args.diagnostic:
        test_indices = list(range(len(test_dataset)))[:int(0.1 * len(test_dataset))]
        test_dataset = Subset(test_dataset, test_indices)
        print(f"Diagnostic test dataset size: {len(test_dataset)}")

    # CPU diagnostics must avoid multiprocessing in this local environment;
    # Lambda GPU runs continue to use the requested worker count.
    dataloader_workers = args.num_workers if device.type != 'cpu' else 0
    pin_memory = device.type == 'cuda'
    print(f"DataLoader workers: {dataloader_workers} | Pin memory: {pin_memory}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=dataloader_workers, pin_memory=pin_memory)

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    normalizer = Normalizer(mean=cifar10_mean, std=cifar10_std).to(device)

    # Model is created once and its weights get overwritten for each
    # checkpoint in the loop below, instead of rebuilding it every time.
    model = PreActResNet18(num_classes=10).to(device)

    fieldnames = [
        'epoch', 'clean_loss', 'clean_acc', 'pixel_robust_loss',
        'pixel_robust_acc', 'low_frequency_robust_loss',
        'low_frequency_robust_acc', 'union_robust_acc', 'eval_time_sec',
    ]
    
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_csv, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        print("\nStarting Checkpoint Evaluations:")
        print("=" * 85)
        print(
            f"{'Epoch':^8} | {'Clean Acc':^10} | {'Pixel Acc':^10} | "
            f"{'Low-Freq Acc':^12} | {'Union Acc':^10} | {'Time (s)':^8}"
        )
        print("=" * 85)

        for ckpt_name in ckpt_files:
            # Step 4: load this checkpoint's weights, then evaluate it.
            epoch = extract_epoch_number(ckpt_name)
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

            # Checkpoints may be saved either as a plain state dict or
            # wrapped in a dict with extra info (e.g. optimizer state), so
            # this handles both formats.
            checkpoint = torch.load(ckpt_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)

            start_time = time.time()

            metrics = evaluate_checkpoint(
                model=model,
                normalizer=normalizer,
                dataloader=test_loader,
                device=device,
                epsilon=args.epsilon,
                alpha=args.alpha,
                num_steps=args.attack_steps,
                dct_cutoff=args.dct_cutoff,
            )

            eval_duration = time.time() - start_time

            print(
                f"{epoch:^8d} | {metrics['clean_acc'] * 100:^9.2f}% | "
                f"{metrics['pixel_robust_acc'] * 100:^9.2f}% | "
                f"{metrics['low_frequency_robust_acc'] * 100:^11.2f}% | "
                f"{metrics['union_robust_acc'] * 100:^9.2f}% | {eval_duration:^8.2f}"
            )

            # Write results after every checkpoint (not just at the end) so
            # progress isn't lost if a later checkpoint fails or the run
            # gets interrupted.
            writer.writerow({
                'epoch': epoch,
                'clean_loss': f"{metrics['clean_loss']:.6f}",
                'clean_acc': f"{metrics['clean_acc']:.6f}",
                'pixel_robust_loss': f"{metrics['pixel_robust_loss']:.6f}",
                'pixel_robust_acc': f"{metrics['pixel_robust_acc']:.6f}",
                'low_frequency_robust_loss': f"{metrics['low_frequency_robust_loss']:.6f}",
                'low_frequency_robust_acc': f"{metrics['low_frequency_robust_acc']:.6f}",
                'union_robust_acc': f"{metrics['union_robust_acc']:.6f}",
                'eval_time_sec': f"{eval_duration:.2f}"
            })
            csv_file.flush()

        print("=" * 85)
        print(f"\nEvaluation complete. Results saved to '{args.output_csv}'.")


if __name__ == '__main__':
    main()
