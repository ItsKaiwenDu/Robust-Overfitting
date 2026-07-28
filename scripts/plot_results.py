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


def load_evaluation_data(csv_path):
    """
    Loads evaluation metrics from CSV file.
    Returns a dictionary containing lists for epochs, accuracies, and losses.
    """
    data = {
        'epochs': [],
        'clean_accs': [],
        'robust_accs': [],
        'clean_losses': [],
        'robust_losses': []
    }
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['epochs'].append(int(row['epoch']))
            data['clean_accs'].append(float(row['clean_acc']) * 100)
            data['robust_accs'].append(float(row['robust_acc']) * 100)
            data['clean_losses'].append(float(row['clean_loss']))
            data['robust_losses'].append(float(row['robust_loss']))
    return data


def plot_accuracy_subplot(ax, data, best_epoch, max_robust_acc):
    """Plots Subplot 1: Clean vs. Robust Accuracy across epochs."""
    ax.plot(data['epochs'], data['clean_accs'], label='Clean Test Accuracy', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax.plot(data['epochs'], data['robust_accs'], label='Robust Test Accuracy (PGD-20)', color='#d62728', linewidth=2.5, marker='s', markersize=4)

    # Highlight Peak Robust Accuracy
    ax.axvline(x=best_epoch, color='#2ca02c', linestyle='--', linewidth=2, label=f'Peak Robustness (Epoch {best_epoch})')
    ax.annotate(f'Peak: {max_robust_acc:.2f}%\n(Epoch {best_epoch})', 
                 xy=(best_epoch, max_robust_acc), 
                 xytext=(best_epoch + 8, max_robust_acc + 4.5),
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', color='#2ca02c',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#2ca02c', alpha=0.9))

    ax.set_xlabel('Epochs\n(↑ Higher is better)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('CIFAR-10 Test Accuracy vs. Epochs', fontsize=13, fontweight='bold', pad=12)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10)
    ax.set_ylim(25, 90)
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_loss_subplot(ax, data, best_epoch):
    """Plots Subplot 2: Clean vs. Robust Loss across epochs."""
    ax.plot(data['epochs'], data['clean_losses'], label='Clean Test Loss', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax.plot(data['epochs'], data['robust_losses'], label='Robust Test Loss (PGD-20)', color='#d62728', linewidth=2.5, marker='s', markersize=4)
    ax.axvline(x=best_epoch, color='#2ca02c', linestyle='--', linewidth=2, label=f'Peak Robustness (Epoch {best_epoch})')

    ax.set_xlabel('Epochs\n(↓ Lower is better)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Cross Entropy Loss', fontsize=12, fontweight='bold')
    ax.set_title('CIFAR-10 Test Loss vs. Epochs', fontsize=13, fontweight='bold', pad=12)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_results(csv_path='Report/evaluation_results.csv', output_path='Report/robust_overfitting_curves.png'):
    """Main orchestrator function for loading evaluation metrics and creating the figure."""
    data = load_evaluation_data(csv_path)

    max_robust_acc = max(data['robust_accs'])
    max_idx = data['robust_accs'].index(max_robust_acc)
    best_epoch = data['epochs'][max_idx]

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    plot_accuracy_subplot(ax1, data, best_epoch, max_robust_acc)
    plot_loss_subplot(ax2, data, best_epoch)

    plt.suptitle('Investigating Robust Overfitting in PreActResNet-18 (Rice et al. Replication)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved successfully as {output_path}")


if __name__ == '__main__':
    plot_results()
