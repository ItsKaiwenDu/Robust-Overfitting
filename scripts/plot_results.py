# File Name: plot_results.py
# Last Updated: September 1, 2026
# Description:
#   This script generates accuracy and loss charts for adversarial training
#   experiments on CIFAR-10. It supports both:
#     1. Post-training evaluation results from CSV files (PGD-20 attacks).
#     2. In-training dynamics from TensorBoard event logs (PGD-10 training).
#   It can produce charts for individual seeds, diagnostic checks, and
#   multi-seed aggregate summaries with shaded standard-deviation bands.
# References:
#   * Rice, L., Wong, E., and Kolter, J. Z. (2020). Overfitting in
#     adversarially robust deep learning. ICML.

import argparse
import csv
import glob
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------------------------------------------------------------------------
# Evaluation Results (CSV) Loading & Plotting
# ---------------------------------------------------------------------------

def load_evaluation_data(csv_path):
    """Reads the evaluation CSV and organizes it into lists for plotting.

    Returns a dictionary of parallel lists (all the same length, matched
    by index), one entry per checkpoint that was evaluated. Accuracy
    values are converted from a 0-1 fraction to a 0-100 percentage.
    """
    data = {
        'epochs': [], 'clean_accs': [], 'pixel_accs': [],
        'low_frequency_accs': [], 'union_accs': [], 'clean_losses': [],
        'pixel_losses': [], 'low_frequency_losses': [],
    }
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['epochs'].append(int(row['epoch']))
            data['clean_accs'].append(float(row['clean_acc']) * 100)
            data['clean_losses'].append(float(row['clean_loss']))
            # Old pixel-only CSVs use robust_*. New CSVs have separate
            # pixel, low-frequency, and union measurements.
            if 'pixel_robust_acc' in row:
                data['pixel_accs'].append(float(row['pixel_robust_acc']) * 100)
                data['pixel_losses'].append(float(row['pixel_robust_loss']))
            else:
                data['pixel_accs'].append(float(row['robust_acc']) * 100)
                data['pixel_losses'].append(float(row['robust_loss']))
            if 'low_frequency_robust_acc' in row and row['low_frequency_robust_acc']:
                data['low_frequency_accs'].append(
                    float(row['low_frequency_robust_acc']) * 100
                )
                data['union_accs'].append(float(row['union_robust_acc']) * 100)
                data['low_frequency_losses'].append(
                    float(row['low_frequency_robust_loss'])
                )
    return data


def plot_accuracy_subplot(ax, data, best_epoch, peak_accuracy, peak_label, training_mode='pixel-only'):
    """Draws the accuracy-vs-epochs chart onto the given subplot axis."""
    ax.plot(data['epochs'], data['clean_accs'], label='Clean Test Accuracy', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax.plot(data['epochs'], data['pixel_accs'], label='Pixel-PGD-20 Robustness', color='#d62728', linewidth=2.5, marker='s', markersize=4)
    if data['low_frequency_accs']:
        ax.plot(data['epochs'], data['low_frequency_accs'], label='Low-Frequency-PGD-20 Robustness', color='#ff7f0e', linewidth=2.5, marker='^', markersize=4)
        ax.plot(data['epochs'], data['union_accs'], label='Union Robustness', color='#9467bd', linewidth=2.5, marker='D', markersize=4)

    # Highlight the best-robustness epoch with a vertical line and callout box.
    ax.axvline(x=best_epoch, color='#2ca02c', linestyle='--', linewidth=2, label=f'Peak {peak_label} (Epoch {best_epoch})')
    if training_mode == 'low-frequency-only':
        xytext = (best_epoch - 42, 48)
    elif training_mode == 'mixed-domain':
        x_text_offset = -40 if best_epoch > 140 else 8
        xytext = (best_epoch + x_text_offset, peak_accuracy + 12)
    elif best_epoch > 150:
        xytext = (best_epoch - 38, peak_accuracy + 4.5)
    else:
        xytext = (best_epoch + 8, peak_accuracy + 4.5)

    ax.annotate(f'Peak {peak_label}: {peak_accuracy:.2f}%\n(Epoch {best_epoch})',
                 xy=(best_epoch, peak_accuracy),
                 xytext=xytext,
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', color='#2ca02c',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#2ca02c', alpha=0.9))

    ax.set_xlabel('Epochs\n(↑ Higher is better)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('CIFAR-10 Test Accuracy vs. Epochs', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(-2, 100)
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_loss_subplot(ax, data, best_epoch, peak_label):
    """Draws the loss-vs-epochs chart onto the given subplot axis."""
    ax.plot(data['epochs'], data['clean_losses'], label='Clean Test Loss', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax.plot(data['epochs'], data['pixel_losses'], label='Pixel-PGD-20 Loss', color='#d62728', linewidth=2.5, marker='s', markersize=4)
    if data['low_frequency_losses']:
        ax.plot(data['epochs'], data['low_frequency_losses'], label='Low-Frequency-PGD-20 Loss', color='#ff7f0e', linewidth=2.5, marker='^', markersize=4)
    ax.axvline(x=best_epoch, color='#2ca02c', linestyle='--', linewidth=2)

    ax.set_xlabel('Epochs\n(↓ Lower is better)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Cross Entropy Loss', fontsize=12, fontweight='bold')
    ax.set_title('CIFAR-10 Test Loss vs. Epochs', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_results(csv_path, output_path, training_mode='pixel-only', title_suffix=None):
    """Builds the full 2-panel evaluation chart and saves it as a PNG file."""
    data = load_evaluation_data(csv_path)

    if training_mode == 'low-frequency-only':
        peak_values = data['low_frequency_accs']
        peak_label = 'Low-Freq Robustness'
    elif training_mode == 'mixed-domain':
        peak_values = data['union_accs'] if data['union_accs'] else data['pixel_accs']
        peak_label = 'Union Robustness'
    else:
        peak_values = data['pixel_accs']
        peak_label = 'Pixel Robustness'

    peak_accuracy = max(peak_values)
    max_idx = peak_values.index(peak_accuracy)
    best_epoch = data['epochs'][max_idx]

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    plot_accuracy_subplot(ax1, data, best_epoch, peak_accuracy, peak_label, training_mode=training_mode)
    plot_loss_subplot(ax2, data, best_epoch, peak_label)

    mode_titles = {
        'pixel-only': 'Pixel-Only Adversarial Training',
        'low-frequency-only': 'Low-Frequency-Only Adversarial Training',
        'mixed-domain': 'Mixed-Domain Adversarial Training'
    }
    condition_title = mode_titles.get(training_mode, training_mode)
    if title_suffix:
        condition_title = f"{condition_title} ({title_suffix})"
    plt.suptitle(f'Robust Overfitting Investigation: {condition_title}', fontsize=15, fontweight='bold', y=0.98)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=len(labels), frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Evaluation chart saved successfully as {output_path}")


# ---------------------------------------------------------------------------
# TensorBoard Training Logs Loading & Plotting
# ---------------------------------------------------------------------------

def load_tensorboard_training_data(tb_dir):
    """Reads TensorBoard scalar events from a run directory.

    Extracts training and in-training test accuracy and loss curves across
    all epochs.
    """
    event_files = sorted(glob.glob(os.path.join(tb_dir, 'events.out.tfevents.*')))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {tb_dir}")

    # Use the latest event file if multiple exist.
    ea = EventAccumulator(event_files[-1])
    ea.Reload()
    tags = ea.Tags().get('scalars', [])

    data = {
        'train_epochs': [], 'train_clean_accs': [], 'train_robust_accs': [], 'train_losses': [],
        'test_epochs': [], 'test_clean_accs': [], 'test_robust_accs': [],
        'test_clean_losses': [], 'test_robust_losses': [],
    }

    if 'Loss/train' in tags:
        for ev in ea.Scalars('Loss/train'):
            data['train_epochs'].append(ev.step)
            data['train_losses'].append(ev.value)
    if 'Accuracy/train_clean' in tags:
        data['train_clean_accs'] = [ev.value * 100 for ev in ea.Scalars('Accuracy/train_clean')]
    if 'Accuracy/train_robust' in tags:
        data['train_robust_accs'] = [ev.value * 100 for ev in ea.Scalars('Accuracy/train_robust')]

    if 'Loss/test_clean' in tags:
        for ev in ea.Scalars('Loss/test_clean'):
            data['test_epochs'].append(ev.step)
            data['test_clean_losses'].append(ev.value)
    if 'Loss/test_robust' in tags:
        data['test_robust_losses'] = [ev.value for ev in ea.Scalars('Loss/test_robust')]
    if 'Accuracy/test_clean' in tags:
        data['test_clean_accs'] = [ev.value * 100 for ev in ea.Scalars('Accuracy/test_clean')]
    if 'Accuracy/test_robust' in tags:
        data['test_robust_accs'] = [ev.value * 100 for ev in ea.Scalars('Accuracy/test_robust')]

    return data


def plot_training_results(tb_dir, output_path, training_mode='pixel-only', title_suffix=None):
    """Builds the full 2-panel training dynamics chart and saves it as a PNG file."""
    data = load_tensorboard_training_data(tb_dir)

    attack_names = {
        'pixel-only': 'PGD-10',
        'low-frequency-only': 'Low-Freq PGD-10',
        'mixed-domain': 'Mixed-Domain PGD-10'
    }
    attack_label = attack_names.get(training_mode, 'PGD-10')

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Subplot 1: Training & Test Accuracy
    if data['train_clean_accs']:
        ax1.plot(data['train_epochs'], data['train_clean_accs'], label='Train Clean Acc', color='#1f77b4', linestyle='--', linewidth=2)
    if data['train_robust_accs']:
        ax1.plot(data['train_epochs'], data['train_robust_accs'], label=f'Train Robust Acc ({attack_label})', color='#1f77b4', linestyle='-', linewidth=2.5)
    if data['test_clean_accs']:
        ax1.plot(data['test_epochs'], data['test_clean_accs'], label='Test Clean Acc', color='#d62728', linestyle='--', linewidth=2)
    if data['test_robust_accs']:
        ax1.plot(data['test_epochs'], data['test_robust_accs'], label='Test Robust Acc (PGD-10)', color='#d62728', linestyle='-', linewidth=2.5)

    ax1.set_xlabel('Epochs\n(↑ Higher is better)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Test Accuracy (TensorBoard Log)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Training & Test Loss
    if data['train_losses']:
        ax2.plot(data['train_epochs'], data['train_losses'], label='Train Robust Loss', color='#1f77b4', linestyle='-', linewidth=2.5)
    if data['test_clean_losses']:
        ax2.plot(data['test_epochs'], data['test_clean_losses'], label='Test Clean Loss', color='#d62728', linestyle='--', linewidth=2)
    if data['test_robust_losses']:
        ax2.plot(data['test_epochs'], data['test_robust_losses'], label='Test Robust Loss', color='#d62728', linestyle='-', linewidth=2.5)

    ax2.set_xlabel('Epochs\n(↓ Lower is better)', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training & Test Loss (TensorBoard Log)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle='--', alpha=0.6)

    mode_titles = {
        'pixel-only': 'Pixel-Only Adversarial Training',
        'low-frequency-only': 'Low-Frequency-Only Adversarial Training',
        'mixed-domain': 'Mixed-Domain Adversarial Training'
    }
    condition_title = mode_titles.get(training_mode, training_mode)
    if title_suffix:
        condition_title = f"{condition_title} ({title_suffix})"
    plt.suptitle(f'Training Dynamics: {condition_title}', fontsize=15, fontweight='bold', y=0.98)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=len(labels), frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Training chart saved successfully as {output_path}")


# ---------------------------------------------------------------------------
# Multi-Seed Aggregation & Plotting (Evaluation & Training)
# ---------------------------------------------------------------------------

def aggregate_evaluation_data(report_mode_dir, output_csv_path=None):
    """Aggregates evaluation CSVs across all seed directories in a mode folder.

    Computes per-epoch mean and standard deviation for all evaluation metrics
    and optionally saves the result to an aggregate CSV.
    """
    seed_dirs = sorted(glob.glob(os.path.join(report_mode_dir, 'seed-*')))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed-* directories found in {report_mode_dir}")

    all_seed_data = []
    for s_dir in seed_dirs:
        csv_file = os.path.join(s_dir, 'evaluation_results.csv')
        if os.path.exists(csv_file):
            all_seed_data.append(load_evaluation_data(csv_file))

    if not all_seed_data:
        raise FileNotFoundError(f"No evaluation_results.csv files found under {report_mode_dir}/seed-*")

    epochs = all_seed_data[0]['epochs']
    num_seeds = len(all_seed_data)

    def mean_std(key):
        arr = np.array([seed[key] for seed in all_seed_data])
        return np.mean(arr, axis=0), np.std(arr, axis=0, ddof=1 if num_seeds > 1 else 0)

    agg = {'epochs': epochs, 'num_seeds': num_seeds}
    for metric in ['clean_accs', 'pixel_accs', 'low_frequency_accs', 'union_accs',
                   'clean_losses', 'pixel_losses', 'low_frequency_losses']:
        if all_seed_data[0][metric]:
            m, s = mean_std(metric)
            agg[f'{metric}_mean'] = m
            agg[f'{metric}_std'] = s
        else:
            agg[f'{metric}_mean'] = []
            agg[f'{metric}_std'] = []

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        has_lf = len(agg['low_frequency_accs_mean']) > 0
        fieldnames = [
            'epoch', 'clean_loss_mean', 'clean_loss_std', 'clean_acc_mean', 'clean_acc_std',
            'pixel_robust_loss_mean', 'pixel_robust_loss_std', 'pixel_robust_acc_mean', 'pixel_robust_acc_std'
        ]
        if has_lf:
            fieldnames.extend([
                'low_frequency_robust_loss_mean', 'low_frequency_robust_loss_std',
                'low_frequency_robust_acc_mean', 'low_frequency_robust_acc_std',
                'union_robust_acc_mean', 'union_robust_acc_std'
            ])

        with open(output_csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, epoch in enumerate(epochs):
                row = {
                    'epoch': epoch,
                    'clean_loss_mean': f"{agg['clean_losses_mean'][i]:.6f}",
                    'clean_loss_std': f"{agg['clean_losses_std'][i]:.6f}",
                    'clean_acc_mean': f"{agg['clean_accs_mean'][i] / 100.0:.6f}",
                    'clean_acc_std': f"{agg['clean_accs_std'][i] / 100.0:.6f}",
                    'pixel_robust_loss_mean': f"{agg['pixel_losses_mean'][i]:.6f}",
                    'pixel_robust_loss_std': f"{agg['pixel_losses_std'][i]:.6f}",
                    'pixel_robust_acc_mean': f"{agg['pixel_accs_mean'][i] / 100.0:.6f}",
                    'pixel_robust_acc_std': f"{agg['pixel_accs_std'][i] / 100.0:.6f}",
                }
                if has_lf:
                    row['low_frequency_robust_loss_mean'] = f"{agg['low_frequency_losses_mean'][i]:.6f}"
                    row['low_frequency_robust_loss_std'] = f"{agg['low_frequency_losses_std'][i]:.6f}"
                    row['low_frequency_robust_acc_mean'] = f"{agg['low_frequency_accs_mean'][i] / 100.0:.6f}"
                    row['low_frequency_robust_acc_std'] = f"{agg['low_frequency_accs_std'][i] / 100.0:.6f}"
                    row['union_robust_acc_mean'] = f"{agg['union_accs_mean'][i] / 100.0:.6f}"
                    row['union_robust_acc_std'] = f"{agg['union_accs_std'][i] / 100.0:.6f}"
                writer.writerow(row)
        print(f"Aggregated evaluation CSV saved to {output_csv_path}")

    return agg


def plot_aggregate_evaluation_results(report_mode_dir, output_png_path, training_mode='pixel-only', output_csv_path=None):
    """Draws aggregated evaluation curves with mean and shaded standard deviation bands."""
    agg = aggregate_evaluation_data(report_mode_dir, output_csv_path=output_csv_path)
    epochs = np.array(agg['epochs'])

    if training_mode == 'low-frequency-only':
        peak_means = agg['low_frequency_accs_mean']
        peak_stds = agg['low_frequency_accs_std']
        peak_label = 'Low-Freq Robustness'
    elif training_mode == 'mixed-domain':
        peak_means = agg['union_accs_mean'] if len(agg['union_accs_mean']) > 0 else agg['pixel_accs_mean']
        peak_stds = agg['union_accs_std'] if len(agg['union_accs_std']) > 0 else agg['pixel_accs_std']
        peak_label = 'Union Robustness'
    else:
        peak_means = agg['pixel_accs_mean']
        peak_stds = agg['pixel_accs_std']
        peak_label = 'Pixel Robustness'

    max_idx = int(np.argmax(peak_means))
    peak_accuracy_mean = peak_means[max_idx]
    peak_accuracy_std = peak_stds[max_idx]
    best_epoch = epochs[max_idx]

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Panel 1: Accuracy
    ax1.plot(epochs, agg['clean_accs_mean'], label='Clean Test Accuracy (Mean)', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax1.fill_between(epochs, agg['clean_accs_mean'] - agg['clean_accs_std'], agg['clean_accs_mean'] + agg['clean_accs_std'], color='#1f77b4', alpha=0.18)

    ax1.plot(epochs, agg['pixel_accs_mean'], label='Pixel-PGD-20 Robustness (Mean)', color='#d62728', linewidth=2.5, marker='s', markersize=4)
    ax1.fill_between(epochs, agg['pixel_accs_mean'] - agg['pixel_accs_std'], agg['pixel_accs_mean'] + agg['pixel_accs_std'], color='#d62728', alpha=0.18)

    if len(agg['low_frequency_accs_mean']) > 0:
        ax1.plot(epochs, agg['low_frequency_accs_mean'], label='Low-Frequency-PGD-20 Robustness (Mean)', color='#ff7f0e', linewidth=2.5, marker='^', markersize=4)
        ax1.fill_between(epochs, agg['low_frequency_accs_mean'] - agg['low_frequency_accs_std'], agg['low_frequency_accs_mean'] + agg['low_frequency_accs_std'], color='#ff7f0e', alpha=0.18)

        ax1.plot(epochs, agg['union_accs_mean'], label='Union Robustness (Mean)', color='#9467bd', linewidth=2.5, marker='D', markersize=4)
        ax1.fill_between(epochs, agg['union_accs_mean'] - agg['union_accs_std'], agg['union_accs_mean'] + agg['union_accs_std'], color='#9467bd', alpha=0.18)

    # Highlight the peak
    if training_mode == 'low-frequency-only':
        xytext = (best_epoch - 42, 48)
    elif training_mode == 'mixed-domain':
        x_text_offset = -40 if best_epoch > 140 else 8
        xytext = (best_epoch + x_text_offset, peak_accuracy_mean + 12)
    elif best_epoch > 150:
        xytext = (best_epoch - 38, peak_accuracy_mean + 4.5)
    else:
        xytext = (best_epoch + 8, peak_accuracy_mean + 4.5)

    ax1.axvline(x=best_epoch, color='#2ca02c', linestyle='--', linewidth=2, label=f'Peak {peak_label} (Epoch {best_epoch})')
    ax1.annotate(f'Peak {peak_label}:\n{peak_accuracy_mean:.2f}% ± {peak_accuracy_std:.2f}%\n(Epoch {best_epoch})',
                 xy=(best_epoch, peak_accuracy_mean),
                 xytext=xytext,
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', color='#2ca02c',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#2ca02c', alpha=0.9))

    ax1.set_xlabel('Epochs\n(↑ Higher is better)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'CIFAR-10 Test Accuracy vs. Epochs ({agg["num_seeds"]} Seeds Mean ± SD)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylim(-2, 100)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Panel 2: Loss
    ax2.plot(epochs, agg['clean_losses_mean'], label='Clean Test Loss (Mean)', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax2.fill_between(epochs, agg['clean_losses_mean'] - agg['clean_losses_std'], agg['clean_losses_mean'] + agg['clean_losses_std'], color='#1f77b4', alpha=0.18)

    ax2.plot(epochs, agg['pixel_losses_mean'], label='Pixel-PGD-20 Loss (Mean)', color='#d62728', linewidth=2.5, marker='s', markersize=4)
    ax2.fill_between(epochs, agg['pixel_losses_mean'] - agg['pixel_losses_std'], agg['pixel_losses_mean'] + agg['pixel_losses_std'], color='#d62728', alpha=0.18)

    if len(agg['low_frequency_losses_mean']) > 0:
        ax2.plot(epochs, agg['low_frequency_losses_mean'], label='Low-Frequency-PGD-20 Loss (Mean)', color='#ff7f0e', linewidth=2.5, marker='^', markersize=4)
        ax2.fill_between(epochs, agg['low_frequency_losses_mean'] - agg['low_frequency_losses_std'], agg['low_frequency_losses_mean'] + agg['low_frequency_losses_std'], color='#ff7f0e', alpha=0.18)

    ax2.axvline(x=best_epoch, color='#2ca02c', linestyle='--', linewidth=2)
    ax2.set_xlabel('Epochs\n(↓ Lower is better)', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Cross Entropy Loss', fontsize=12, fontweight='bold')
    ax2.set_title(f'CIFAR-10 Test Loss vs. Epochs ({agg["num_seeds"]} Seeds Mean ± SD)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle='--', alpha=0.6)

    mode_titles = {
        'pixel-only': 'Pixel-Only Adversarial Training',
        'low-frequency-only': 'Low-Frequency-Only Adversarial Training',
        'mixed-domain': 'Mixed-Domain Adversarial Training'
    }
    condition_title = mode_titles.get(training_mode, training_mode)
    plt.suptitle(f'Robust Overfitting Investigation: {condition_title} ({agg["num_seeds"]}-Seed Overall)', fontsize=15, fontweight='bold', y=0.98)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=len(labels), frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    os.makedirs(os.path.dirname(output_png_path) or '.', exist_ok=True)
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Aggregated evaluation chart saved successfully as {output_png_path}")


def aggregate_training_data(runs_mode_dir):
    """Aggregates TensorBoard training scalars across all seed runs in a mode."""
    seed_dirs = sorted(glob.glob(os.path.join(runs_mode_dir, 'seed-*')))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed-* directories found in {runs_mode_dir}")

    all_seed_data = []
    for s_dir in seed_dirs:
        try:
            all_seed_data.append(load_tensorboard_training_data(s_dir))
        except FileNotFoundError:
            continue

    if not all_seed_data:
        raise FileNotFoundError(f"No TensorBoard logs found under {runs_mode_dir}/seed-*")

    train_epochs = np.array(all_seed_data[0]['train_epochs'])
    test_epochs = np.array(all_seed_data[0]['test_epochs'])
    num_seeds = len(all_seed_data)

    def mean_std(arr_list):
        arr = np.array(arr_list)
        return np.mean(arr, axis=0), np.std(arr, axis=0, ddof=1 if num_seeds > 1 else 0)

    agg = {'train_epochs': train_epochs, 'test_epochs': test_epochs, 'num_seeds': num_seeds}

    for key in ['train_clean_accs', 'train_robust_accs', 'train_losses']:
        m, s = mean_std([d[key] for d in all_seed_data])
        agg[f'{key}_mean'] = m
        agg[f'{key}_std'] = s

    for key in ['test_clean_accs', 'test_robust_accs', 'test_clean_losses', 'test_robust_losses']:
        m, s = mean_std([d[key] for d in all_seed_data])
        agg[f'{key}_mean'] = m
        agg[f'{key}_std'] = s

    return agg


def plot_aggregate_training_results(runs_mode_dir, output_png_path, training_mode='pixel-only'):
    """Draws aggregated training curves with mean and shaded standard deviation bands."""
    agg = aggregate_training_data(runs_mode_dir)
    train_epochs = agg['train_epochs']
    test_epochs = agg['test_epochs']

    attack_names = {
        'pixel-only': 'PGD-10',
        'low-frequency-only': 'Low-Freq PGD-10',
        'mixed-domain': 'Mixed-Domain PGD-10'
    }
    attack_label = attack_names.get(training_mode, 'PGD-10')

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Subplot 1: Training & Test Accuracy
    ax1.plot(train_epochs, agg['train_clean_accs_mean'], label='Train Clean Acc (Mean)', color='#1f77b4', linestyle='--', linewidth=2)
    ax1.fill_between(train_epochs, agg['train_clean_accs_mean'] - agg['train_clean_accs_std'], agg['train_clean_accs_mean'] + agg['train_clean_accs_std'], color='#1f77b4', alpha=0.15)

    ax1.plot(train_epochs, agg['train_robust_accs_mean'], label=f'Train Robust Acc ({attack_label}, Mean)', color='#1f77b4', linestyle='-', linewidth=2.5)
    ax1.fill_between(train_epochs, agg['train_robust_accs_mean'] - agg['train_robust_accs_std'], agg['train_robust_accs_mean'] + agg['train_robust_accs_std'], color='#1f77b4', alpha=0.15)

    ax1.plot(test_epochs, agg['test_clean_accs_mean'], label='Test Clean Acc (Mean)', color='#d62728', linestyle='--', linewidth=2)
    ax1.fill_between(test_epochs, agg['test_clean_accs_mean'] - agg['test_clean_accs_std'], agg['test_clean_accs_mean'] + agg['test_clean_accs_std'], color='#d62728', alpha=0.15)

    ax1.plot(test_epochs, agg['test_robust_accs_mean'], label='Test Robust Acc (PGD-10, Mean)', color='#d62728', linestyle='-', linewidth=2.5)
    ax1.fill_between(test_epochs, agg['test_robust_accs_mean'] - agg['test_robust_accs_std'], agg['test_robust_accs_mean'] + agg['test_robust_accs_std'], color='#d62728', alpha=0.15)

    ax1.set_xlabel('Epochs\n(↑ Higher is better)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Training & Test Accuracy ({agg["num_seeds"]} Seeds Mean ± SD)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Training & Test Loss
    ax2.plot(train_epochs, agg['train_losses_mean'], label='Train Robust Loss (Mean)', color='#1f77b4', linestyle='-', linewidth=2.5)
    ax2.fill_between(train_epochs, agg['train_losses_mean'] - agg['train_losses_std'], agg['train_losses_mean'] + agg['train_losses_std'], color='#1f77b4', alpha=0.15)

    ax2.plot(test_epochs, agg['test_clean_losses_mean'], label='Test Clean Loss (Mean)', color='#d62728', linestyle='--', linewidth=2)
    ax2.fill_between(test_epochs, agg['test_clean_losses_mean'] - agg['test_clean_losses_std'], agg['test_clean_losses_mean'] + agg['test_clean_losses_std'], color='#d62728', alpha=0.15)

    ax2.plot(test_epochs, agg['test_robust_losses_mean'], label='Test Robust Loss (Mean)', color='#d62728', linestyle='-', linewidth=2.5)
    ax2.fill_between(test_epochs, agg['test_robust_losses_mean'] - agg['test_robust_losses_std'], agg['test_robust_losses_mean'] + agg['test_robust_losses_std'], color='#d62728', alpha=0.15)

    ax2.set_xlabel('Epochs\n(↓ Lower is better)', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title(f'Training & Test Loss ({agg["num_seeds"]} Seeds Mean ± SD)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle='--', alpha=0.6)

    mode_titles = {
        'pixel-only': 'Pixel-Only Adversarial Training',
        'low-frequency-only': 'Low-Frequency-Only Adversarial Training',
        'mixed-domain': 'Mixed-Domain Adversarial Training'
    }
    condition_title = mode_titles.get(training_mode, training_mode)
    plt.suptitle(f'Training Dynamics: {condition_title} ({agg["num_seeds"]}-Seed Overall)', fontsize=15, fontweight='bold', y=0.98)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=len(labels), frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    os.makedirs(os.path.dirname(output_png_path) or '.', exist_ok=True)
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Aggregated training chart saved successfully as {output_png_path}")


# ---------------------------------------------------------------------------
# CLI & Execution Entrypoint
# ---------------------------------------------------------------------------

def process_mode_plots(training_mode, seed=None, run_name=None, diagnostic=False,
                       plot_type='both', aggregate=False, all_seeds=False,
                       csv_path=None, output_path=None, runs_dir=None):
    """Processes plotting workflows for a specific mode."""
    report_mode_dir = os.path.join('report', training_mode)
    runs_mode_dir = os.path.join('runs', training_mode)
    mode_prefixes = {
        'pixel-only': 'po',
        'low-frequency-only': 'lfo',
        'mixed-domain': 'mixed',
    }
    mode_prefix = mode_prefixes[training_mode]

    # 1. Overall plotting if requested or if all_seeds is requested
    if aggregate or all_seeds:
        print(f"\n--- Generating Overall Results for {training_mode} ---")
        agg_report_dir = os.path.join(report_mode_dir, 'overall')
        if plot_type in ('both', 'eval'):
            agg_eval_png = os.path.join(agg_report_dir, f'{mode_prefix}_eval_results_curves.png')
            agg_eval_csv = os.path.join(agg_report_dir, 'evaluation_results.csv')
            plot_aggregate_evaluation_results(report_mode_dir, agg_eval_png, training_mode=training_mode, output_csv_path=agg_eval_csv)
        if plot_type in ('both', 'train'):
            agg_train_png = os.path.join(agg_report_dir, f'{mode_prefix}_train_results_curves.png')
            plot_aggregate_training_results(runs_mode_dir, agg_train_png, training_mode=training_mode)

    # 2. Identify target runs to plot
    if all_seeds:
        seed_dirs = sorted(glob.glob(os.path.join(report_mode_dir, 'seed-*')))
        target_runs = [os.path.basename(sd) for sd in seed_dirs]
        if os.path.exists(os.path.join(report_mode_dir, 'baseline')):
            target_runs.insert(0, 'baseline')
    elif run_name is not None:
        target_runs = [run_name]
    elif seed is not None:
        target_runs = [f'seed-{seed}']
    elif not aggregate:
        # Default single run fallback
        target_runs = ['seed-42']
    else:
        target_runs = []

    # 3. Plot individual runs
    for r_name in target_runs:
        print(f"\n--- Processing Run: {training_mode}/{r_name} ---")
        run_parts = [training_mode]
        if diagnostic:
            run_parts.append('diagnostic')
        run_parts.append(r_name)
        run_rel = os.path.join(*run_parts)

        cur_csv = csv_path or os.path.join('report', run_rel, 'evaluation_results.csv')
        cur_eval_png = output_path or os.path.join('report', run_rel, f'{mode_prefix}_eval_results_curves.png')
        cur_runs_dir = runs_dir or os.path.join('runs', run_rel)
        cur_train_png = os.path.join('report', run_rel, f'{mode_prefix}_train_results_curves.png')

        seed_label = r_name.replace('seed-', 'Seed ') if r_name.startswith('seed-') else r_name
        if plot_type in ('both', 'eval') and os.path.exists(cur_csv):
            plot_results(cur_csv, cur_eval_png, training_mode=training_mode, title_suffix=seed_label)
        if plot_type in ('both', 'train') and os.path.exists(cur_runs_dir):
            try:
                plot_training_results(cur_runs_dir, cur_train_png, training_mode=training_mode, title_suffix=seed_label)
            except Exception as e:
                print(f"Warning: Could not generate training plot for {cur_runs_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Plot evaluation and training results for adversarial training runs.')
    parser.add_argument('--training-mode', default='low-frequency-only', choices=('pixel-only', 'low-frequency-only', 'mixed-domain'),
                        help='training mode to plot (default: low-frequency-only)')
    parser.add_argument('--seed', default=None, type=int, help='specific random seed to plot (e.g. 42)')
    parser.add_argument('--run-name', default=None, help='run directory name, such as seed-42 or baseline')
    parser.add_argument('--diagnostic', action='store_true', help='read from a diagnostic run directory')
    parser.add_argument('--plot-type', default='both', choices=('both', 'eval', 'train'),
                        help='type of plots to generate: eval, train, or both (default: both)')
    parser.add_argument('--overall', '--aggregate', dest='overall', action='store_true',
                        help='generate multi-seed evaluation and training plots in overall/')
    parser.add_argument('--all-seeds', action='store_true', help='generate plots for all seeds in this mode + overall')
    parser.add_argument('--all-modes', action='store_true', help='generate all seed and overall plots across all 3 modes')
    parser.add_argument('--csv-path', default=None, help='override the default evaluation CSV path')
    parser.add_argument('--output-path', default=None, help='override the default evaluation chart path')
    parser.add_argument('--runs-dir', default=None, help='override the default TensorBoard runs directory')
    args = parser.parse_args()

    if args.all_modes:
        modes = ['low-frequency-only', 'pixel-only', 'mixed-domain']
        for mode in modes:
            process_mode_plots(
                training_mode=mode,
                seed=args.seed,
                run_name=args.run_name,
                diagnostic=args.diagnostic,
                plot_type=args.plot_type,
                aggregate=True,
                all_seeds=True,
                csv_path=args.csv_path,
                output_path=args.output_path,
                runs_dir=args.runs_dir
            )
    else:
        process_mode_plots(
            training_mode=args.training_mode,
            seed=args.seed,
            run_name=args.run_name,
            diagnostic=args.diagnostic,
            plot_type=args.plot_type,
            aggregate=args.overall,
            all_seeds=args.all_seeds,
            csv_path=args.csv_path,
            output_path=args.output_path,
            runs_dir=args.runs_dir
        )


if __name__ == '__main__':
    main()
