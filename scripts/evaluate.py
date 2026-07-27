'''
Title: PreActResNet-18 Checkpoint Evaluation Script (PGD-20)
Purpose:
    This script iterates through saved model checkpoints from the adversarial 
    training run, evaluating each checkpoint on the CIFAR-10 test set under 
    both clean (unperturbed) conditions and PGD-20 adversarial attacks. 
    Metrics are saved to a CSV file for tracking robust overfitting curves.
References:
    - Rice, Wong & Kolter (2020) "Overfitting in adversarially robust deep learning"
    - Goodfellow et al. (2014) "Explaining and Harnessing Adversarial Examples"
'''

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

from Models.preact_resnet import PreActResNet18

# Ignore SSL errors for CIFAR-10 downloading if needed
ssl._create_default_https_context = ssl._create_unverified_context


class Normalizer(nn.Module):
    """
    Channel-wise normalization layer operating inside the PyTorch forward pass.
    Allows raw [0, 1] tensor inputs for both clean images and adversarial perturbing loops.
    """
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x):
        return (x - self.mean) / self.std


def generate_pgd_adversarial(model, normalizer, X, y, epsilon, alpha, num_steps, device):
    """
    Generates PGD adversarial perturbation for input images X.
    Uses random initialization within the epsilon-ball, clamped to valid image range [0, 1].
    """
    model.eval()
    
    # Uniform random start within [-epsilon, epsilon] ball
    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).to(device)
    # Ensure initial point is inside valid image bounds [0, 1]
    delta = torch.clamp(X + delta, min=0.0, max=1.0) - X
    
    for _ in range(num_steps):
        delta.requires_grad = True
        perturbed_X = X + delta
        outputs = model(normalizer(perturbed_X))
        loss = F.cross_entropy(outputs, y)
        
        model.zero_grad()
        loss.backward()
        
        grad = delta.grad.detach()
        
        # Gradient ascent step
        delta = delta.detach() + alpha * grad.sign()
        # Project back into epsilon-ball
        delta = torch.clamp(delta, min=-epsilon, max=epsilon)
        # Keep total image within valid range [0, 1]
        delta = torch.clamp(X + delta, min=0.0, max=1.0) - X
        
    return (X + delta).detach()


def evaluate_checkpoint(model, normalizer, dataloader, device, epsilon, alpha, num_steps):
    """
    Evaluates a single model state on the dataloader under both clean and PGD conditions.
    Returns: (clean_loss, clean_acc, robust_loss, robust_acc)
    """
    model.eval()
    clean_loss = 0.0
    clean_correct = 0
    robust_loss = 0.0
    robust_correct = 0
    total = 0
    
    # 1. Clean evaluation pass (no gradient tracking needed)
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
            
    # 2. Robust evaluation pass (PGD-20 attack requiring gradients)
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
    """
    Extracts numerical epoch index from filename (e.g. 'epoch_105.pt' -> 105).
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1


def main():
    parser = argparse.ArgumentParser(description='Evaluate PreActResNet-18 Checkpoints with PGD-20')
    parser.add_argument('--checkpoint-dir', default='Checkpoints', type=str, help='Directory containing saved .pt checkpoints')
    parser.add_argument('--output-csv', default='Report/evaluation_results.csv', type=str, help='Path to output CSV file')
    parser.add_argument('--data-dir', default='./data', type=str, help='Dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size for evaluation')
    parser.add_argument('--epsilon', default=8.0/255.0, type=float, help='Adversarial perturbation magnitude epsilon')
    parser.add_argument('--alpha', default=2.0/255.0, type=float, help='PGD step size alpha')
    parser.add_argument('--attack-steps', default=20, type=int, help='Number of PGD attack steps (default: 20 for PGD-20)')
    parser.add_argument('--diagnostic', action='store_true', help='Diagnostic mode: evaluate on 10%% of test set')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    args = parser.parse_args()

    # 1. Set reproducible seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # 2. Hardware device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")

    # 3. Locate checkpoints
    if not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory '{args.checkpoint_dir}' does not exist.")

    ckpt_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pt') or f.endswith('.pth')]
    if not ckpt_files:
        raise FileNotFoundError(f"No .pt or .pth checkpoint files found in '{args.checkpoint_dir}'.")

    # Sort files numerically by epoch
    ckpt_files.sort(key=extract_epoch_number)

    print(f"Found {len(ckpt_files)} checkpoints in '{args.checkpoint_dir}'.")
    print(f"Attack Configuration: PGD-{args.attack_steps} | Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255) | Alpha: {args.alpha:.4f} ({args.alpha*255:.1f}/255)")
    if args.diagnostic:
        print("Running in Diagnostic Mode (10% test dataset).")

    # 4. Prepare CIFAR-10 Test Dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)

    if args.diagnostic:
        test_indices = list(range(len(test_dataset)))[:int(0.1 * len(test_dataset))]
        test_dataset = Subset(test_dataset, test_indices)
        print(f"Diagnostic test dataset size: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 5. Initialize Normalizer and Model
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    normalizer = Normalizer(mean=cifar10_mean, std=cifar10_std).to(device)

    model = PreActResNet18(num_classes=10).to(device)

    # 6. Initialize CSV File
    fieldnames = ['epoch', 'clean_loss', 'clean_acc', 'robust_loss', 'robust_acc', 'eval_time_sec']
    
    # If output directory doesn't exist, create it
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
            epoch = extract_epoch_number(ckpt_name)
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

            # Load weights into model
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

            # Write row to CSV
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
