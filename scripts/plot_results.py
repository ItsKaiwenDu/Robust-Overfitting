'''
Title: Robust Overfitting Plotting Script
Purpose:
    Reads Report/evaluation_results.csv and plots clean vs. robust accuracy and loss 
    across epochs, clearly marking the robust overfitting peak (Epoch 105).
'''

import csv
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt

def plot_results(csv_path='Report/evaluation_results.csv', output_path='Report/robust_overfitting_curves.png'):
    epochs = []
    clean_accs = []
    robust_accs = []
    clean_losses = []
    robust_losses = []

    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            clean_accs.append(float(row['clean_acc']) * 100)
            robust_accs.append(float(row['robust_acc']) * 100)
            clean_losses.append(float(row['clean_loss']))
            robust_losses.append(float(row['robust_loss']))

    # Find peak robust accuracy
    max_robust_acc = max(robust_accs)
    max_idx = robust_accs.index(max_robust_acc)
    best_epoch = epochs[max_idx]

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Subplot 1: Accuracy
    ax1.plot(epochs, clean_accs, label='Clean Test Accuracy', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax1.plot(epochs, robust_accs, label='Robust Test Accuracy (PGD-20)', color='#d62728', linewidth=2.5, marker='s', markersize=4)

    # Highlight Peak Robust Accuracy
    ax1.axvline(x=best_epoch, color='#2ca02c', linestyle='--', linewidth=2, label=f'Peak Robustness (Epoch {best_epoch})')
    ax1.scatter([best_epoch], [max_robust_acc], color='#2ca02c', s=120, zorder=5, marker='*')
    ax1.annotate(f'Peak: {max_robust_acc:.2f}%\n(Epoch {best_epoch})', 
                 xy=(best_epoch, max_robust_acc), 
                 xytext=(best_epoch + 8, max_robust_acc - 4),
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', color='#2ca02c',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#2ca02c', alpha=0.9))

    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('CIFAR-10 Test Accuracy vs. Epochs', fontsize=13, fontweight='bold', pad=12)
    ax1.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10)
    ax1.set_ylim(25, 90)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Loss
    ax2.plot(epochs, clean_losses, label='Clean Test Loss', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax2.plot(epochs, robust_losses, label='Robust Test Loss (PGD-20)', color='#d62728', linewidth=2.5, marker='s', markersize=4)
    ax2.axvline(x=best_epoch, color='#2ca02c', linestyle='--', linewidth=2, label=f'Peak Robustness (Epoch {best_epoch})')

    ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cross Entropy Loss', fontsize=12, fontweight='bold')
    ax2.set_title('CIFAR-10 Test Loss vs. Epochs', fontsize=13, fontweight='bold', pad=12)
    ax2.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Investigating Robust Overfitting in PreActResNet-18 (Rice et al. Replication)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved successfully as {output_path}")

if __name__ == '__main__':
    plot_results()
