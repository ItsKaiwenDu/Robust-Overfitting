# Investigating Robust Overfitting in Adversarial Training

This is GitHub repository for research on robust overfitting in adversarial training.

---

## Project Overview
Deep neural networks can be easily fooled by adversarial attacks, which are small, hidden changes to inputs that cause model to make wrong predictions. Adversarial training helps fix this, but models often run into a problem known as **robust overfitting**. This means that later in training, model's performance on test attacks gets worse even though its training loss keeps improving.

The project first reproduced robust overfitting with pixel-space PGD adversarial training. Its next phase tests whether robust overfitting changes when the adversarial-training attack domain is randomized across epochs: standard pixel-space PGD or low-frequency DCT-constrained PGD.

Current Research Objectives:

* Maintain the existing pixel-space PGD result as the robust-overfitting baseline using PreActResNet-18 on CIFAR-10.
* Establish a low-frequency-only DCT-constrained PGD baseline using the same model and training schedule.
* Train a mixed-domain model that randomly selects pixel-space or low-frequency PGD once per epoch.
* Compare robust-overfitting curves under both component attack domains and their per-example worst case.

## Research Question and Hypothesis

**Research question:** During PGD adversarial training of PreActResNet-18 on CIFAR-10, how does randomly alternating pixel-space PGD and low-frequency DCT-constrained PGD across epochs affect the timing and severity of robust overfitting, compared with pixel-only and low-frequency-only training?

**Hypothesis:** With the architecture, dataset, training schedule, perturbation budget, and evaluation schedule held constant, mixed-domain training will produce robust-accuracy curves that differ from the single-domain baselines. The peak epoch may shift, the peak may flatten, or the post-peak decline may change under one or both evaluation attacks.

---

## Planned Experiment Design

The pre-implementation specification—including the fixed DCT mask, attack
update, randomized schedule, checkpoint evaluation, run artifacts, and
diagnostic acceptance criteria—is in
[`experimental_specification.md`](experimental_specification.md).

### Training conditions

All conditions use the same PreActResNet-18 architecture, CIFAR-10 data, optimizer, learning-rate schedule, number of epochs, training PGD step count, and random seed policy.

1. **Pixel-only baseline:** Train with standard pixel-space PGD in every epoch. This is the completed Rice et al. replication.
2. **Low-frequency-only baseline:** Train with DCT-constrained PGD in every epoch. The adversarial perturbation is restricted by a predefined low-frequency mask before being transformed back to image space.
3. **Mixed-domain condition:** At the start of each epoch, use a seeded fair random choice to select either pixel-space PGD or low-frequency DCT-constrained PGD. Every batch in that epoch uses the selected attack domain.

The low-frequency mask and the attack budget will be saved with each run configuration. The frequency attack will use the same image-space L-infinity budget and the same number of PGD steps as the pixel-space attack unless a documented diagnostic shows that an adjustment is necessary.

### Evaluation protocol

At every saved checkpoint, evaluate each model on:

1. clean CIFAR-10 test images;
2. test images attacked with pixel-space PGD;
3. test images attacked with low-frequency DCT-constrained PGD; and
4. a union summary that records whether either attack succeeds on each test image.

The primary robust-overfitting measurements are the peak robust-accuracy epoch and the peak-to-final robust-accuracy drop for the pixel, low-frequency, and union evaluations. Reporting all three avoids mistaking improved robustness to one attack for an overall improvement.

### Scope

This is a controlled robust-overfitting study, not an attempt to reproduce Multi Steepest Descent or TaFD. The model architecture remains unchanged. The only planned training intervention is the adversarial attack domain used during each epoch.

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
├── checkpoints/                       # Saved model checkpoints during training
│   └── diagnostic/                    # Local diagnostic run checkpoints (e.g., epoch_1.pt)
├── models/                            # Model architecture definitions
│   └── preact_resnet.py               # PreActResNet-18 model architecture in PyTorch
├── notes/                             # Reading literature notes
│   ├── goodfellow.md                  # Literature notes on FGSM and adversarial training
│   └── rice.md                        # Literature notes on robust overfitting
├── report/                            # Presentations and evaluation outputs
│   ├── slides.pdf                     # Research presentation slides
│   ├── evaluation_results.csv         # Raw evaluation metrics across checkpoints
│   ├── robust_overfitting_curves.png  # Robust overfitting accuracy/loss plot
│   └── training_results_curves.png    # Training & evaluation performance curves
├── scripts/                           # Python scripts for training, evaluation, plotting, and setup
│   ├── dct_pgd.py                     # Low-frequency DCT-masked PGD implementation
│   ├── evaluate.py                    # Checkpoint evaluation script (PGD-20)
│   ├── plot_results.py                # Plotting script for accuracy and loss curves
│   ├── train.py                       # Core adversarial PGD training script
│   └── verify_setup.py                # Setup verification script
├── data/                              # [Ignored] CIFAR-10 dataset files (downloaded automatically)
├── runs/                              # [Ignored] TensorBoard logging directories
├── .gitignore                         # Files and folders ignored by Git
├── experimental_specification.md       # Fixed mixed-domain experiment protocol
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
* Chen, Y., Ren, Q., and Yan, J. (2022). *Rethinking and Improving Robustness of Convolutional Neural Networks: A Shapley Value-based Approach in Frequency Domain.* NeurIPS 35.
* Guo, C., Frank, J. S., and Weinberger, K. Q. (2019). *Low Frequency Adversarial Perturbation.* UAI 2019.
* Yu, C., Han, B., Shen, L., Yu, J., Gong, C., Gong, M., and Liu, T. (2022). *Understanding Robust Overfitting of Adversarial Training and Beyond.* ICML 2022, PMLR 162.
* Bu, Q., Huang, D., and Cui, H. (2023). *Towards Building More Robust Models with Frequency Bias.* ICCV 2023.
* Kim, Y., Kim, S., Seo, I., and Shin, B. (2023). *Phase-shifted Adversarial Training.* UAI 2023, PMLR 216.
* Li, F., Li, K., Wu, H., Tian, J., and Zhou, J. (2024). *DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in Frequency Domain.* NeurIPS 2024.
* Tramèr, F., and Boneh, D. (2019). *Adversarial Training and Robustness for Multiple Perturbations.* NeurIPS 32, pp. 5866-5876. arXiv:1904.13000.
* Maini, P., Wong, E., and Kolter, J. Z. (2020). *Adversarial Robustness Against the Union of Multiple Perturbation Models.* ICML 2020, PMLR 119, pp. 6640-6650. arXiv:1909.04068.
* Xie, M., He, Y., and Fang, M. (2026). *TaFD: Threat-Aware Frequency Decoupling for Adversarial Robustness against Heterogeneous Attacks.* arXiv:2606.17540.
