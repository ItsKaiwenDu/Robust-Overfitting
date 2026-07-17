'''
Title: PGD Adversarial Training Pipeline for CIFAR-10
Purpose:
    This script implements a custom PyTorch training loop to perform PGD-10 
    adversarial training on a PreActResNet-18 model. It supports TensorBoard 
    metric logging, epoch-level checkpoint saving, and a local diagnostic mode.
References:
    - Setup and Hyperparameters: Replicated from Rice, Wong & Kolter (2020) 
      "Overfitting in adversarially robust deep learning" (ICML 2020).
    - Adversarial Attack Concepts: Based on Goodfellow et al. (2014) 
      "Explaining and Harnessing Adversarial Examples" (ICLR 2015).
'''
import argparse
import os
import random
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from Models.preact_resnet import PreActResNet18

# Normalizer class to handle channel-wise normalization inside the model forward pass/PGD loop
class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x):
        return (x - self.mean) / self.std

# 10-step PGD adversarial perturbation generator
def generate_pgd_adversarial(model, normalizer, X, y, epsilon, alpha, num_steps, device):
    model.eval()
    
    # Random start within epsilon ball
    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).to(device)
    # Ensure starting point is a valid image (clipped to [0, 1])
    delta = torch.clamp(X + delta, min=0.0, max=1.0) - X
    
    for _ in range(num_steps):
        delta.requires_grad = True
        perturbed_X = X + delta
        outputs = model(normalizer(perturbed_X))
        loss = F.cross_entropy(outputs, y)
        
        model.zero_grad()
        loss.backward()
        
        grad = delta.grad.detach()
        
        # Gradient ascent step: add sign of gradient to maximize loss
        delta = delta.detach() + alpha * grad.sign()
        # Project back to epsilon ball
        delta = torch.clamp(delta, min=-epsilon, max=epsilon)
        # Ensure final perturbed image is within [0, 1] bounds
        delta = torch.clamp(X + delta, min=0.0, max=1.0) - X
        
    model.train()
    return (X + delta).detach()

def evaluate(model, normalizer, dataloader, device, epsilon, alpha, num_steps):
    model.eval()
    clean_loss = 0.0
    clean_correct = 0
    robust_loss = 0.0
    robust_correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            batch_size = X.size(0)
            total += batch_size
            
            # Clean evaluation
            clean_outputs = model(normalizer(X))
            c_loss = F.cross_entropy(clean_outputs, y, reduction='sum')
            clean_loss += c_loss.item()
            clean_pred = clean_outputs.argmax(dim=1)
            clean_correct += clean_pred.eq(y).sum().item()
            
    # Robust evaluation (needs gradients, so we enable grad just for PGD generation)
    # Note: we construct the adversarial examples block-by-batch
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        with torch.enable_grad():
            X_adv = generate_pgd_adversarial(model, normalizer, X, y, epsilon, alpha, num_steps, device)
        
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

def main():
    parser = argparse.ArgumentParser(description='PreActResNet-18 PGD Adversarial Training on CIFAR-10')
    parser.add_argument('--epochs', default=200, type=int, help='total epochs (default: 200)')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size for training (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr-decay-epochs', default=[100, 150], type=int, nargs='+', help='epochs at which to decay lr')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (L2 penalty)')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--epsilon', default=8.0/255.0, type=float, help='adversarial perturbation constraint epsilon')
    parser.add_argument('--alpha', default=2.0/255.0, type=float, help='adversarial step size alpha')
    parser.add_argument('--attack-steps', default=10, type=int, help='number of PGD attack steps during training')
    parser.add_argument('--diagnostic', action='store_true', help='run in diagnostic mode (1 epoch, 10%% data)')
    parser.add_argument('--seed', default=42, type=int, help='random seed (default: 42)')
    parser.add_argument('--checkpoint-dir', default='Checkpoints', type=str, help='directory to save checkpoints')
    parser.add_argument('--runs-dir', default='runs', type=str, help='TensorBoard logging directory')
    args = parser.parse_args()
    
    # 1. Set seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    # 2. Select acceleration device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")
        
    # Adjust config for diagnostic smoke test
    if args.diagnostic:
        print("Mode: Diagnostic")
        args.epochs = 1
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, 'diagnostic')
        args.runs_dir = os.path.join(args.runs_dir, 'diagnostic')
        
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)
    
    # 3. Load CIFAR-10 dataset
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    if args.diagnostic:
        # Use 10% of CIFAR-10 training and test datasets for fast verification
        train_indices = list(range(len(train_dataset)))[:int(0.1 * len(train_dataset))]
        test_indices = list(range(len(test_dataset)))[:int(0.1 * len(test_dataset))]
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
        print(f"Diagnostic training subset size: {len(train_dataset)}")
        print(f"Diagnostic test subset size: {len(test_dataset)}")
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 4. Initialize model, normalizer, optimizer, scheduler, and writer
    # Standard normalization parameters from Rice et al.
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    normalizer = Normalizer(mean=cifar10_mean, std=cifar10_std).to(device)
    
    model = PreActResNet18(num_classes=10).to(device)
    model.train()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=0.1)
    
    writer = SummaryWriter(log_dir=args.runs_dir)
    
    print(f"Starting training pipeline: {args.epochs} epochs.")
    print(f"Hyperparameters: LR={args.lr}, Decay Epochs={args.lr_decay_epochs}, Weight Decay={args.weight_decay}, Momentum={args.momentum}")
    print(f"PGD Attack config: epsilon={args.epsilon:.4f}, alpha={args.alpha:.4f}, steps={args.attack_steps}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        train_loss = 0.0
        train_clean_correct = 0
        train_robust_correct = 0
        total = 0
        
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            batch_size = X.size(0)
            total += batch_size
            
            # Forward pass on clean inputs (no gradients tracked for accuracy reporting)
            with torch.no_grad():
                clean_outputs = model(normalizer(X))
                clean_pred = clean_outputs.argmax(dim=1)
                train_clean_correct += clean_pred.eq(y).sum().item()
                
            # Generate PGD adversarial inputs
            # Enable grad temporarily to compute input gradients inside PGD attack
            with torch.enable_grad():
                X_adv = generate_pgd_adversarial(
                    model, normalizer, X, y,
                    epsilon=args.epsilon,
                    alpha=args.alpha,
                    num_steps=args.attack_steps,
                    device=device
                )
                
            # Forward pass on adversarial inputs
            robust_outputs = model(normalizer(X_adv))
            loss = F.cross_entropy(robust_outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_size
            robust_pred = robust_outputs.argmax(dim=1)
            train_robust_correct += robust_pred.eq(y).sum().item()
            
        epoch_time = time.time() - epoch_start_time
        
        # Calculate training statistics
        epoch_train_loss = train_loss / total
        epoch_train_clean_acc = train_clean_correct / total
        epoch_train_robust_acc = train_robust_correct / total
        
        print(f"Epoch {epoch:03d} | Train Loss: {epoch_train_loss:.4f} | "
              f"Train Clean Acc: {epoch_train_clean_acc:.2%} | Train Robust Acc: {epoch_train_robust_acc:.2%} | "
              f"Time: {epoch_time:.2f}s")
        
        # Log training metrics
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/train_clean', epoch_train_clean_acc, epoch)
        writer.add_scalar('Accuracy/train_robust', epoch_train_robust_acc, epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        
        # Evaluate model on test set
        # Run every 5 epochs, or at the first and last epochs, or if diagnostic
        should_eval = (epoch % 5 == 0) or (epoch == 1) or (epoch == args.epochs) or args.diagnostic
        if should_eval:
            eval_start_time = time.time()
            test_loss, test_clean_acc, test_robust_loss, test_robust_acc = evaluate(
                model, normalizer, test_loader, device,
                epsilon=args.epsilon,
                alpha=args.alpha,
                num_steps=args.attack_steps
            )
            eval_time = time.time() - eval_start_time
            print(f"--> Evaluation: Test Clean Acc: {test_clean_acc:.2%} | Test Robust Acc: {test_robust_acc:.2%} | Time: {eval_time:.2f}s")
            
            writer.add_scalar('Loss/test_clean', test_loss, epoch)
            writer.add_scalar('Loss/test_robust', test_robust_loss, epoch)
            writer.add_scalar('Accuracy/test_clean', test_clean_acc, epoch)
            writer.add_scalar('Accuracy/test_robust', test_robust_acc, epoch)
            
        # Save checkpoints: every 5 epochs, or at the final epoch, or in diagnostic mode (at epoch 1)
        should_checkpoint = (epoch % 5 == 0) or (epoch == args.epochs) or args.diagnostic
        if should_checkpoint:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_train_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
        # Update scheduler
        scheduler.step()
        
        # If CUDA is available, log memory usage
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
            print(f"GPU VRAM Allocated: {vram_mb:.2f} MB")
            
    writer.close()
    print("Training pipeline finished.")

if __name__ == '__main__':
    main()
