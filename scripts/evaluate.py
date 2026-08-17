# File Name: evaluate.py
# Last Updated: July 29, 2026
# Description:
#   This script measures how well a trained model is doing at each saved
#   checkpoint. For every checkpoint, it checks accuracy on normal
#   (clean) test images, and it also checks accuracy after attacking
#   those images with PGD-20, a standard adversarial attack. This lets us
#   see how the model's real-world robustness changes over the course of
#   training, which is what we need to find robust overfitting. Results
#   for every checkpoint get written out to a CSV file.
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
    model.eval()
    
    # Start from a small random perturbation instead of zero. This helps
    # avoid a weaker, less realistic attack and is standard practice for PGD.
    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).to(device)
    delta = torch.clamp(X + delta, min=0.0, max=1.0) - X
    
    for _ in range(num_steps):
        delta.requires_grad = True
        perturbed_X = X + delta
        outputs = model(normalizer(perturbed_X))
        loss = F.cross_entropy(outputs, y)
        
        model.zero_grad()
        loss.backward()
        
        grad = delta.grad.detach()
        
        # Step 3: move in the direction that increases the loss (the sign
        # of the gradient).
        delta = delta.detach() + alpha * grad.sign()
        # Step 4: keep the change within the allowed budget and make sure
        # the image stays a valid pixel value.
        delta = torch.clamp(delta, min=-epsilon, max=epsilon)
        delta = torch.clamp(X + delta, min=0.0, max=1.0) - X
        
    return (X + delta).detach()


def evaluate_checkpoint(model, normalizer, dataloader, device, epsilon, alpha, num_steps):
    """Measures a model's accuracy and loss on clean and attacked images.

    Runs through the test data in two passes:
    1. Evaluate on the original ("clean") images.
    2. Generate an adversarial attack for each batch, then evaluate on that.

    Returns the average loss and accuracy for both passes, which together
    show how much the model's performance drops under attack.
    """
    model.eval()
    clean_loss = 0.0
    clean_correct = 0
    robust_loss = 0.0
    robust_correct = 0
    total = 0
    
    # Step 1: evaluate on the original, unmodified images.
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            batch_size = X.size(0)
            total += batch_size
            
            clean_outputs = model(normalizer(X))
            c_loss = F.cross_entropy(clean_outputs, y, reduction='sum')
            clean_loss += c_loss.item()
            clean_pred = clean_outputs.argmax(dim=1)
            clean_correct += clean_pred.eq(y).sum().item()
            
    # Step 2: generate an attack for each batch, then evaluate on that.
    # Gradients need to be enabled here since generating the attack requires
    # them, even though the model itself isn't being trained.
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        with torch.enable_grad():
            X_adv = generate_pgd_adversarial(
                model, normalizer, X, y,
                epsilon=epsilon,
                alpha=alpha,
                num_steps=num_steps,
                device=device
            )
            
        with torch.no_grad():
            robust_outputs = model(normalizer(X_adv))
            r_loss = F.cross_entropy(robust_outputs, y, reduction='sum')
            robust_loss += r_loss.item()
            robust_pred = robust_outputs.argmax(dim=1)
            robust_correct += robust_pred.eq(y).sum().item()
            
    return (
        clean_loss / total,
        clean_correct / total,
        robust_loss / total,
        robust_correct / total
    )


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
       and adversarial (PGD-20) images, print a summary line, and write
       the result to the output CSV.
    """
    parser = argparse.ArgumentParser(description='Evaluate PreActResNet-18 Checkpoints with PGD-20')
    parser.add_argument('--checkpoint-dir', default='checkpoints', type=str, help='Directory containing saved .pt checkpoints')
    parser.add_argument('--output-csv', default='report/evaluation_results.csv', type=str, help='Path to output CSV file')
    parser.add_argument('--data-dir', default='./data', type=str, help='Dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size for evaluation')
    parser.add_argument('--epsilon', default=8.0/255.0, type=float, help='Adversarial perturbation magnitude epsilon')
    parser.add_argument('--alpha', default=2.0/255.0, type=float, help='PGD step size alpha')
    parser.add_argument('--attack-steps', default=20, type=int, help='Number of PGD attack steps (default: 20 for PGD-20)')
    parser.add_argument('--diagnostic', action='store_true', help='Diagnostic mode: evaluate on 10% of test set')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    args = parser.parse_args()

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
    print(f"Attack Configuration: PGD-{args.attack_steps} | Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255) | Alpha: {args.alpha:.4f} ({args.alpha*255:.1f}/255)")
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

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    normalizer = Normalizer(mean=cifar10_mean, std=cifar10_std).to(device)

    # Model is created once and its weights get overwritten for each
    # checkpoint in the loop below, instead of rebuilding it every time.
    model = PreActResNet18(num_classes=10).to(device)

    fieldnames = ['epoch', 'clean_loss', 'clean_acc', 'robust_loss', 'robust_acc', 'eval_time_sec']
    
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_csv, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        print("\nStarting Checkpoint Evaluations:")
        print("=" * 85)
        print(f"{'Epoch':^8} | {'Clean Acc':^12} | {'Robust Acc':^12} | {'Clean Loss':^12} | {'Robust Loss':^12} | {'Time (s)':^8}")
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

            clean_loss, clean_acc, robust_loss, robust_acc = evaluate_checkpoint(
                model=model,
                normalizer=normalizer,
                dataloader=test_loader,
                device=device,
                epsilon=args.epsilon,
                alpha=args.alpha,
                num_steps=args.attack_steps
            )

            eval_duration = time.time() - start_time

            print(f"{epoch:^8d} | {clean_acc * 100:^11.2f}% | {robust_acc * 100:^11.2f}% | {clean_loss:^12.4f} | {robust_loss:^12.4f} | {eval_duration:^8.2f}")

            # Write results after every checkpoint (not just at the end) so
            # progress isn't lost if a later checkpoint fails or the run
            # gets interrupted.
            writer.writerow({
                'epoch': epoch,
                'clean_loss': f"{clean_loss:.6f}",
                'clean_acc': f"{clean_acc:.6f}",
                'robust_loss': f"{robust_loss:.6f}",
                'robust_acc': f"{robust_acc:.6f}",
                'eval_time_sec': f"{eval_duration:.2f}"
            })
            csv_file.flush()

        print("=" * 85)
        print(f"\nEvaluation complete. Results saved to '{args.output_csv}'.")


if __name__ == '__main__':
    main()