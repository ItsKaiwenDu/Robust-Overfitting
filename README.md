# Investigating Robust Overfitting in Adversarial Training

This is GitHub repository for research on robust overfitting in adversarial training.

---

## Project Overview
Deep neural networks can be easily fooled by adversarial attacks, which are small, hidden changes to inputs that cause model to make wrong predictions. Adversarial training helps fix this, but models often run into a problem known as **robust overfitting**. This means that later in training, model's performance on test attacks gets worse even though its training loss keeps improving.

The project first reproduced robust overfitting with pixel-space PGD adversarial training. Its next phase investigates whether robust overfitting changes when adversarial perturbations are restricted to different image-frequency bands.

Current Research Objectives:

* Maintain the pixel-space PGD result as a baseline for robust overfitting using a PreActResNet-18 model on CIFAR-10.
* Investigate whether robust overfitting differs when adversarial perturbations are restricted to low-, middle-, or high-frequency image components.
* Compare the frequency-band results with the pixel-space baseline to identify how the selected frequency band affects robust accuracy over training.

## Research Question and Hypothesis

**Research question:** How does restricting adversarial perturbations to low-, middle-, or high-frequency image bands affect the timing and severity of robust overfitting during PGD adversarial training of PreActResNet-18 on CIFAR-10?

**Hypothesis:** With the training and evaluation setup held constant, restricting perturbations to different frequency bands will produce different robust-accuracy curves, including differences in the peak robust-accuracy epoch and the amount of post-peak decline.

---

## Research Team
* Principal Investigator: Dr. Nicholas Q. Tran (Department of Mathematics and Computer Science)
* Student Researcher: Kaiwen Du (Computer Science)

---

## Local Setup Instructions

1. **Create virtual environment** *(an isolated Python workspace that keeps this project's dependencies separate from other Python projects on your system)*:
   ```bash
   python3 -m venv .venv
   ```
2. **Activate & install dependencies**:
   ```bash
   source .venv/bin/activate
   pip3 install -r requirements.txt
   ```
3. **Verify setup**:
   ```bash
   python3 scripts/verify_setup.py
   ```

> **Note:** The **CIFAR-10** dataset (~170 MB) will be downloaded automatically to `data/` on the first training run. No manual download is required.

---

## Project Directory Structure

```text
Robust-Overfitting/
├── Checkpoints/                       # Saved model checkpoints during training
│   └── diagnostic/                    # Local diagnostic run checkpoints (e.g., epoch_1.pt)
├── Models/                            # Model architecture definitions
│   └── preact_resnet.py               # PreActResNet-18 model architecture in PyTorch
├── Notes/                             # Reading literature notes
│   ├── goodfellow.md                  # Literature notes on FGSM and adversarial training
│   └── rice.md                        # Literature notes on robust overfitting
├── Report/                            # Presentations and evaluation outputs
│   ├── slides.pdf                     # Research presentation slides
│   ├── evaluation_results.csv         # Raw evaluation metrics across checkpoints
│   ├── robust_overfitting_curves.png  # Robust overfitting accuracy/loss plot
│   └── training_results_curves.png    # Training & evaluation performance curves
├── scripts/                           # Python scripts for training, evaluation, plotting, and setup
│   ├── evaluate.py                    # Checkpoint evaluation script (PGD-20)
│   ├── plot_results.py                # Plotting script for accuracy and loss curves
│   ├── train.py                       # Core adversarial PGD training script
│   └── verify_setup.py                # Setup verification script
├── data/                              # [Ignored] CIFAR-10 dataset files (downloaded automatically)
├── runs/                              # [Ignored] TensorBoard logging directories
├── .gitignore                         # Files and folders ignored by Git
├── goals.md                           # Weekly goals, objectives, and expectations
├── progress.md                        # Weekly progress reports
├── proposal.md                        # Project proposal document
├── README.md                          # Project documentation and setup
├── setup_lambda_labs.md               # Cloud GPU setup guide for Lambda Labs
└── requirements.txt                   # Python package dependencies
```

---

## Weekly Goals & Progress

Weekly research objectives, detailed action items, expectations, and deliverables are tracked in [`goals.md`](goals.md).

For weekly execution logs and detailed progress notes, see [`progress.md`](progress.md).

---

## References
* Goodfellow, I. J., Shlens, J., and Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
* Rice, L., Wong, E., and Kolter, J. Z. (2020). *Overfitting in adversarially robust deep learning.* ICML.
