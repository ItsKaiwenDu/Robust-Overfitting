# File Name: train.py
# Last Updated: July 29, 2026
# Description:
#   This is the main training script for the project. It trains a
#   PreActResNet-18 model on CIFAR-10 using adversarial training, meaning
#   the model is trained on attacked (PGD) images instead of normal ones,
#   so it learns to resist adversarial attacks. Along the way it logs
#   progress to TensorBoard and regularly saves checkpoints and runs a
#   quick evaluation, so we can later see exactly when robust overfitting
#   starts to happen.
# References:
#   * Rice, L., Wong, E., and Kolter, J. Z. (2020). Overfitting in
#     adversarially robust deep learning. ICML.
#   * Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A.
#     (2018). Towards Deep Learning Models Resistant to Adversarial
#     Attacks. ICLR. (defines the PGD attack used here)

import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

from models.preact_resnet import PreActResNet18
from scripts.dct_pgd import generate_low_frequency_dct_pgd


class Normalizer(nn.Module):
    """Rescales images using CIFAR-10's known mean and standard deviation.

    Models train better on normalized data. This is kept as its own step
    (instead of baking it into the dataset transform) so the raw,
    unnormalized image can still be used directly when generating the
    adversarial attack.
    """
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x):
        return (x - self.mean) / self.std


def generate_pgd_adversarial(model, normalizer, X, y, epsilon, alpha, num_steps, device):
    """Creates adversarial (attacked) versions of a batch of images.

    This is the PGD attack, and it's what generates the training data used
    for adversarial training. It works like this:
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
        
    # Switch back to train mode since this function is called from inside
    # the training loop, right before the model is trained on this batch.
    model.train()
    return (X + delta).detach()


def select_epoch_attack_domain(training_mode, schedule_rng):
    """Choose the attack domain used for one complete training epoch.

    Single-domain modes always return their requested domain. Mixed-domain
    mode uses a private, seeded RNG so its schedule remains reproducible even
    when other parts of the training run consume random numbers.
    """
    if training_mode == 'pixel-only':
        return 'pixel'
    if training_mode == 'low-frequency-only':
        return 'low-frequency'
    if training_mode == 'mixed-domain':
        return schedule_rng.choice(('pixel', 'low-frequency'))
    raise ValueError(f"Unknown training mode: {training_mode}")


def generate_adversarial_examples(
    attack_domain, model, normalizer, X, y, epsilon, alpha, num_steps, device,
    dct_cutoff,
):
    """Generate a batch of adversarial examples in the selected domain."""
    if attack_domain == 'pixel':
        return generate_pgd_adversarial(
            model, normalizer, X, y, epsilon, alpha, num_steps, device
        )
    if attack_domain == 'low-frequency':
        return generate_low_frequency_dct_pgd(
            model, normalizer, X, y, epsilon, alpha, num_steps,
            cutoff=dct_cutoff,
        )
    raise ValueError(f"Unknown attack domain: {attack_domain}")


def evaluate(model, normalizer, dataloader, device, epsilon, alpha, num_steps):
    """Measures the model's accuracy and loss on clean and attacked images.

    Runs through the given data in two passes:
    1. Evaluate on the original ("clean") images.
    2. Generate an adversarial attack for each batch, then evaluate on that.

    Returns the average loss and accuracy for both passes. Used during
    training to periodically check how the model is doing on held-out
    test data, separate from what it's actually being trained on.
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
    """Runs the full adversarial training pipeline.

    Overall flow:
    1. Parse settings and set up the device and random seed.
    2. Load the CIFAR-10 dataset and prepare the model, optimizer, and logger.
    3. Loop over each epoch: train on adversarial images, then periodically
       evaluate on the test set and save a checkpoint.
    """
    parser = argparse.ArgumentParser(description='PreActResNet-18 PGD Adversarial Training on CIFAR-10')
    parser.add_argument('--epochs', default=200, type=int, help='total epochs (default: 200)')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size for training (default: 128)')
    parser.add_argument('--num-workers', default=2, type=int, help='DataLoader worker processes when using an accelerator (default: 2)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr-decay-epochs', default=[100, 150], type=int, nargs='+', help='epochs at which to decay lr')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (L2 penalty)')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--epsilon', default=8.0/255.0, type=float, help='adversarial perturbation constraint epsilon')
    parser.add_argument('--alpha', default=2.0/255.0, type=float, help='adversarial step size alpha')
    parser.add_argument('--attack-steps', default=10, type=int, help='number of PGD attack steps during training')
    parser.add_argument(
        '--training-mode', default='pixel-only',
        choices=('pixel-only', 'low-frequency-only', 'mixed-domain'),
        help=(
            'adversarial-training mode: pixel-only uses pixel-space PGD, '
            'low-frequency-only uses DCT-masked PGD, and mixed-domain chooses one per epoch'
        ),
    )
    parser.add_argument(
        '--dct-cutoff', default=8, type=int,
        help='side length of the retained top-left DCT low-frequency mask',
    )
    parser.add_argument('--diagnostic', action='store_true', help='run in diagnostic mode (1 epoch, 10%% data)')
    parser.add_argument('--seed', default=42, type=int, help='random seed (default: 42)')
    parser.add_argument(
        '--run-name', default=None, type=str,
        help='name for this run inside its training-mode directory (default: seed-<seed>)',
    )
    parser.add_argument(
        '--checkpoint-dir', default=None, type=str,
        help='checkpoint output directory (default: checkpoints/<mode>/<run-name>)',
    )
    parser.add_argument(
        '--runs-dir', default=None, type=str,
        help='TensorBoard output directory (default: runs/<mode>/<run-name>)',
    )
    args = parser.parse_args()

    if not 1 <= args.dct_cutoff <= 32:
        parser.error('--dct-cutoff must be between 1 and 32 for CIFAR-10 images.')

    # Keep every condition and random seed in its own directory. This avoids
    # overwriting checkpoints or TensorBoard logs when several runs coexist.
    if args.run_name is None:
        args.run_name = f'seed-{args.seed}' if args.seed is not None else 'unseeded'
    run_parts = [args.training_mode]
    if args.diagnostic:
        run_parts.append('diagnostic')
    run_parts.append(args.run_name)
    run_relative_path = os.path.join(*run_parts)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join('checkpoints', run_relative_path)
    if args.runs_dir is None:
        args.runs_dir = os.path.join('runs', run_relative_path)
    
    # Step 1: set up reproducibility and device.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            # Trade a bit of speed for exact reproducibility on GPU.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")
        
    # Diagnostic mode runs a quick 1-epoch, 10%-of-data sanity check
    # instead of a full training run, and saves to a separate folder so it
    # doesn't overwrite real runs.
    if args.diagnostic:
        print("Mode: Diagnostic")
        args.epochs = 1
        
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)
    
    # Step 2: load the dataset and set up the model, optimizer, and logger.
    # Training images get random crop and flip augmentation; test images
    # do not, so evaluation stays consistent between runs.
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
        train_indices = list(range(len(train_dataset)))[:int(0.1 * len(train_dataset))]
        test_indices = list(range(len(test_dataset)))[:int(0.1 * len(test_dataset))]
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
        print(f"Diagnostic training subset size: {len(train_dataset)}")
        print(f"Diagnostic test subset size: {len(test_dataset)}")
        
    # This local environment cannot create PyTorch shared-memory worker
    # processes. CPU runs use the main process; accelerator runs retain the
    # configurable worker count used on Lambda Labs.
    dataloader_workers = args.num_workers if device.type != 'cpu' else 0
    pin_memory = device.type == 'cuda'
    print(f"DataLoader workers: {dataloader_workers} | Pin memory: {pin_memory}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=dataloader_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=dataloader_workers, pin_memory=pin_memory)
    
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    normalizer = Normalizer(mean=cifar10_mean, std=cifar10_std).to(device)
    
    model = PreActResNet18(num_classes=10).to(device)
    model.train()
    
    # Standard SGD with momentum and weight decay, plus a learning rate
    # schedule that drops the LR at the given milestone epochs.
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=0.1)
    
    writer = SummaryWriter(log_dir=args.runs_dir)
    
    print(f"Starting training pipeline: {args.epochs} epochs.")
    print(f"Hyperparameters: LR={args.lr}, Decay Epochs={args.lr_decay_epochs}, Weight Decay={args.weight_decay}, Momentum={args.momentum}")
    print(f"PGD Attack config: epsilon={args.epsilon:.4f}, alpha={args.alpha:.4f}, steps={args.attack_steps}")
    print(f"Training mode: {args.training_mode} | Run: {args.run_name} | DCT cutoff: {args.dct_cutoff}")
    print(f"Checkpoints: {args.checkpoint_dir} | TensorBoard: {args.runs_dir}")

    # This RNG is intentionally independent from augmentation, shuffled data,
    # and PGD random starts. Therefore the same seed produces the same
    # epoch-level mixed-domain schedule across runs.
    schedule_rng = random.Random(args.seed)
    attack_domain_schedule = []
    
    # Step 3: train for the given number of epochs.
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        attack_domain = select_epoch_attack_domain(args.training_mode, schedule_rng)
        attack_domain_schedule.append(attack_domain)
        print(f"Epoch {epoch:03d} attack domain: {attack_domain}")
        
        train_loss = 0.0
        train_clean_correct = 0
        train_robust_correct = 0
        total = 0
        
        model.train()
        # Step 3a: train on every batch for this epoch.
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            batch_size = X.size(0)
            total += batch_size
            
            # Check clean accuracy for logging only; this doesn't affect
            # training since no gradient is computed here.
            with torch.no_grad():
                clean_outputs = model(normalizer(X))
                clean_pred = clean_outputs.argmax(dim=1)
                train_clean_correct += clean_pred.eq(y).sum().item()
                
            # Generate an adversarial version of this batch, then train the
            # model on it. This is the core of adversarial training: the
            # model only ever learns from attacked images, not clean ones.
            with torch.enable_grad():
                X_adv = generate_adversarial_examples(
                    attack_domain, model, normalizer, X, y,
                    epsilon=args.epsilon, alpha=args.alpha,
                    num_steps=args.attack_steps, device=device,
                    dct_cutoff=args.dct_cutoff,
                )
                
            robust_outputs = model(normalizer(X_adv))
            loss = F.cross_entropy(robust_outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_size
            robust_pred = robust_outputs.argmax(dim=1)
            train_robust_correct += robust_pred.eq(y).sum().item()
            
        epoch_time = time.time() - epoch_start_time
        
        epoch_train_loss = train_loss / total
        epoch_train_clean_acc = train_clean_correct / total
        epoch_train_robust_acc = train_robust_correct / total
        
        print(f"Epoch {epoch:03d} ({attack_domain}) | Train Loss: {epoch_train_loss:.4f} | "
              f"Train Clean Acc: {epoch_train_clean_acc:.2%} | Train Robust Acc: {epoch_train_robust_acc:.2%} | "
              f"Time: {epoch_time:.2f}s")
        
        # Step 3b: log this epoch's training stats to TensorBoard.
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/train_clean', epoch_train_clean_acc, epoch)
        writer.add_scalar('Accuracy/train_robust', epoch_train_robust_acc, epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        writer.add_text('AttackDomain/train', attack_domain, epoch)
        
        # Step 3c: evaluate on the test set every 5 epochs (plus the first
        # and last epoch, or every epoch in diagnostic mode). Running a
        # full PGD evaluation every single epoch would be too slow.
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
            
        # Step 3d: save a checkpoint on the same schedule as evaluation
        # (every 5 epochs, plus the last epoch). These checkpoints are
        # what evaluate.py and plot_results.py later analyze to find
        # exactly where robust overfitting begins.
        should_checkpoint = (epoch % 5 == 0) or (epoch == args.epochs) or args.diagnostic
        if should_checkpoint:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_train_loss,
                'training_mode': args.training_mode,
                'dct_cutoff': args.dct_cutoff,
                'attack_domain': attack_domain,
                'attack_domain_schedule': attack_domain_schedule.copy(),
                'seed': args.seed,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
        # Advance the learning rate schedule after the epoch completes.
        scheduler.step()
        
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
            print(f"GPU VRAM Allocated: {vram_mb:.2f} MB")
            
    writer.close()
    print("Training pipeline finished.")

if __name__ == '__main__':
    main()
