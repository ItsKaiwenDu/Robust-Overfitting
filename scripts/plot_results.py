# File Name: plot_results.py
# Last Updated: July 29, 2026
# Description:
#   This script turns the evaluation results CSV (from evaluate.py) into a
#   chart. It creates two side-by-side plots: one showing accuracy over
#   training epochs, and one showing loss over training epochs, each
#   comparing clean (normal) performance against robust (PGD-20 attacked)
#   performance. The chart also marks the epoch with the best robust
#   accuracy, which is the point we care about most for spotting robust
#   overfitting.
# References:
#   * Rice, L., Wong, E., and Kolter, J. Z. (2020). Overfitting in
#     adversarially robust deep learning. ICML.

import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt


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
            if 'low_frequency_robust_acc' in row:
                data['low_frequency_accs'].append(
                    float(row['low_frequency_robust_acc']) * 100
                )
                data['union_accs'].append(float(row['union_robust_acc']) * 100)
                data['low_frequency_losses'].append(
                    float(row['low_frequency_robust_loss'])
                )
    return data


def plot_accuracy_subplot(ax, data, best_epoch, peak_accuracy, peak_label):
    """Draws the accuracy-vs-epochs chart onto the given subplot axis.

    Plots clean accuracy and robust accuracy as two lines over training,
    then marks the epoch with the best robust accuracy with a vertical
    line and a text callout. That marked epoch is the best checkpoint to
    use if robust overfitting has already started to hurt later epochs.
    """
    ax.plot(data['epochs'], data['clean_accs'], label='Clean Test Accuracy', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax.plot(data['epochs'], data['pixel_accs'], label='Pixel-PGD-20 Robustness', color='#d62728', linewidth=2.5, marker='s', markersize=4)
    if data['low_frequency_accs']:
        ax.plot(data['epochs'], data['low_frequency_accs'], label='Low-Frequency-PGD-20 Robustness', color='#ff7f0e', linewidth=2.5, marker='^', markersize=4)
        ax.plot(data['epochs'], data['union_accs'], label='Union Robustness', color='#9467bd', linewidth=2.5, marker='D', markersize=4)

    # Highlight the best-robustness epoch with a vertical line and a
    # labeled callout box, so it stands out at a glance.
    ax.axvline(x=best_epoch, color='#2ca02c', linestyle='--', linewidth=2, label=f'Peak {peak_label} (Epoch {best_epoch})')
    ax.annotate(f'Peak {peak_label}: {peak_accuracy:.2f}%\n(Epoch {best_epoch})',
                 xy=(best_epoch, peak_accuracy),
                 xytext=(best_epoch + 8, peak_accuracy + 4.5),
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', color='#2ca02c',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#2ca02c', alpha=0.9))

    ax.set_xlabel('Epochs\n(↑ Higher is better)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('CIFAR-10 Test Accuracy vs. Epochs', fontsize=13, fontweight='bold', pad=12)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10, loc='lower right')
    ax.set_ylim(-2, 100)
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_loss_subplot(ax, data, best_epoch, peak_label):
    """Draws the loss-vs-epochs chart onto the given subplot axis.

    Same idea as the accuracy chart, but for loss instead. The same
    best-robust-accuracy epoch is marked here too, so both charts line
    up and can be compared side by side.
    """
    ax.plot(data['epochs'], data['clean_losses'], label='Clean Test Loss', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax.plot(data['epochs'], data['pixel_losses'], label='Pixel-PGD-20 Loss', color='#d62728', linewidth=2.5, marker='s', markersize=4)
    if data['low_frequency_losses']:
        ax.plot(data['epochs'], data['low_frequency_losses'], label='Low-Frequency-PGD-20 Loss', color='#ff7f0e', linewidth=2.5, marker='^', markersize=4)
    ax.axvline(x=best_epoch, color='#2ca02c', linestyle='--', linewidth=2, label=f'Peak {peak_label} (Epoch {best_epoch})')

    ax.set_xlabel('Epochs\n(↓ Lower is better)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Cross Entropy Loss', fontsize=12, fontweight='bold')
    ax.set_title('CIFAR-10 Test Loss vs. Epochs', fontsize=13, fontweight='bold', pad=12)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_results(csv_path, output_path, training_mode='pixel-only'):
    """Builds the full 2-panel chart and saves it as an image file.

    Steps:
    1. Load the evaluation results from the CSV.
    2. Find the epoch with the best robust accuracy for this condition.
    3. Draw the accuracy chart and the loss chart side by side.
    4. Save the finished chart to a PNG file.
    """
    # Step 1: load the evaluation results.
    data = load_evaluation_data(csv_path)

    # Step 2: find the epoch with the best robust accuracy based on training mode.
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

    # Step 3: draw both charts side by side.
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    plot_accuracy_subplot(ax1, data, best_epoch, peak_accuracy, peak_label)
    plot_loss_subplot(ax2, data, best_epoch, peak_label)

    mode_titles = {
        'pixel-only': 'Pixel-Only Adversarial Training',
        'low-frequency-only': 'Low-Frequency-Only Adversarial Training',
        'mixed-domain': 'Mixed-Domain Adversarial Training'
    }
    condition_title = mode_titles.get(training_mode, training_mode)
    plt.suptitle(f'Robust Overfitting Investigation: {condition_title}', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Step 4: save the chart to a file.
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved successfully as {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot one condition and seed from evaluation results.')
    parser.add_argument('--training-mode', default='pixel-only', choices=('pixel-only', 'low-frequency-only', 'mixed-domain'))
    parser.add_argument('--seed', default=None, type=int, help='random seed (e.g. 42)')
    parser.add_argument('--run-name', default=None, help='run directory name, such as seed-42')
    parser.add_argument('--diagnostic', action='store_true', help='read from a diagnostic run directory')
    parser.add_argument('--csv-path', default=None, help='override the default evaluation CSV path')
    parser.add_argument('--output-path', default=None, help='override the default chart path')
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f'seed-{args.seed}' if args.seed is not None else 'seed-42'

    run_parts = [args.training_mode]
    if args.diagnostic:
        run_parts.append('diagnostic')
    run_parts.append(args.run_name)
    run_relative_path = os.path.join(*run_parts)
    csv_path = args.csv_path or os.path.join('report', run_relative_path, 'evaluation_results.csv')
    output_path = args.output_path or os.path.join('report', run_relative_path, 'robust_overfitting_curves.png')
    plot_results(csv_path, output_path, training_mode=args.training_mode)
