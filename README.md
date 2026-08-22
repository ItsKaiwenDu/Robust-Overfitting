# Investigating Robust Overfitting in Adversarial Training

This is GitHub repository for research on robust overfitting in adversarial training.
Last updated: August 19, 2026

---

## Project Overview
Deep neural networks can be easily tricked by adversarial attacks, which are small, human-imperceptible changes to inputs that cause model to make wrong predictions. Adversarial training helps fix this, but models often run into a problem known as **robust overfitting**. This means that later in training, model's performance on test attacks gets worse even though its training loss keeps improving.

The project first reproduced robust overfitting with pixel-space PGD adversarial training. Its next phase tests whether robust overfitting changes when adversarial-training attack domain is randomized across epochs: standard pixel-space PGD or low-frequency DCT-masked PGD.

## Research Question & Hypothesis

**Research question:** During PGD adversarial training of PreActResNet-18 on CIFAR-10, how does randomly alternating pixel-space PGD and low-frequency DCT-masked PGD across epochs affect timing and severity of robust overfitting, compared with pixel-only and low-frequency-only training?

**Hypothesis:** With architecture, dataset, training schedule, perturbation budget, and evaluation schedule held constant, mixed-domain training will produce robust-accuracy curves that differ from single-domain baselines. The peak epoch may shift, peak may flatten, or post-peak decline may change under one or both evaluation attacks.

---

## Experiment Design

### Model and Dataset

* **Model:** PreActResNet-18: Standard deep residual network used in adversarial training research, sourced from the Rice et al. (2020) codebase.
* **Dataset:** CIFAR-10: Contains 50,000 training images and 10,000 test images across 10 classes, each 32×32 pixels.

### Training Configurations

All conditions use the same PreActResNet-18 architecture, CIFAR-10 data, optimizer, learning-rate schedule, 200 epochs, 10-step training PGD, perturbation budget, and random-seed policy.

1. **Pixel-only:** Train with standard pixel-space PGD in every epoch. This is the completed Rice et al. replication.
2. **Low-frequency-only:** Train with low-frequency DCT-masked PGD in every epoch. The adversarial perturbation is restricted by a predefined low-frequency mask before being transformed back to image space.
3. **Mixed-domain (pixel or low-frequency):** At the start of each epoch, use a seeded fair random choice to select either pixel-space PGD or low-frequency DCT-masked PGD. Every batch in that epoch uses the selected attack domain.

### Evaluation Configurations

Every saved checkpoint (40 per condition) is evaluated against the full 10,000-image CIFAR-10 test set using four metrics:

1. **Clean accuracy:** test the model on unmodified images with no attack.
2. **Pixel-space robustness:** attack each test image with pixel-space PGD-20 (20 steps, epsilon = 8/255) and measure how often the model still predicts correctly.
3. **Low-frequency robustness:** attack each test image with DCT-masked PGD-20 using the same budget and measure robustness.
4. **Union robustness:** per image, count it as correct only if the model resisted *both* the pixel-space and low-frequency attacks. This means measuring whether the model is robust to both attack domains for the same image.

For each metric, we record the **peak epoch** (when accuracy was highest) and the **post-peak decline** (how much it dropped by epoch 200). These are the primary numbers compared across conditions.

---

## Setup Guide

**Local Initialization**:

1. Clone this repository and navigate into this repo.
   ```bash
   git clone https://github.com/ItsKaiwenDu/Robust-Overfitting.git
   cd Robust-Overfitting
   ```
2. Create and activate a Python virtual environment:
```bash
python3 -m venv .venv && source .venv/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Verify everything works:
```bash
python3 scripts/verify_setup.py
```
5. Run a quick diagnostic (1 epoch, 10% of data) to confirm the training pipeline:
```bash
python3 scripts/train.py --diagnostic
```

The training domain is selected with `--training-mode`:

```bash
# Pixel-only baseline (default)
python3 scripts/train.py --training-mode pixel-only

# Low-frequency DCT-masked PGD
python3 scripts/train.py --training-mode low-frequency-only --dct-cutoff 8

# Seeded, fair pixel/DCT selection once per epoch
python3 scripts/train.py --training-mode mixed-domain --seed 42 --dct-cutoff 8
```

By default, each run uses a separate directory named after its condition and
seed. Diagnostics add a `diagnostic/` level, so the three one-epoch checks do
not overwrite one another:

```text
checkpoints/<training-mode>/[diagnostic/]<run-name>/
runs/<training-mode>/[diagnostic/]<run-name>/
report/<training-mode>/[diagnostic/]<run-name>/evaluation_results.csv
```

`<run-name>` defaults to `seed-<seed>` and can be changed with `--run-name`.
For example, run the low-frequency diagnostic and then its matching evaluation:

```bash
python3 scripts/train.py --training-mode low-frequency-only --diagnostic --seed 42
python3 scripts/evaluate.py --training-mode low-frequency-only --diagnostic --seed 42
```

Mixed-domain checkpoints record the selected domain for the checkpoint epoch
and the complete schedule so far, together with the seed and DCT cutoff.

**Option A: Lambda Labs (Cloud)**

6a. See [`setup_lambda_labs.md`](setup_lambda_labs.md) for the complete step-by-step guide, including instance provisioning, SSH access, code syncing, running training in the background, monitoring with TensorBoard, downloading results, and terminating the instance.

**Option B: Local**

6b. Run training:
```bash
python3 scripts/train.py
```
7b. Run evaluation across all checkpoints:
```bash
python3 scripts/evaluate.py
```
8b. Plot results:
```bash
python3 scripts/plot_results.py
```

> **Note:** The **CIFAR-10** dataset (~170 MB) will be downloaded automatically to `data/` on first training run. No manual download is required.

### Pretrained Checkpoints

To inspect the completed pixel-space PGD training run without reproducing all 200 epochs, download the 40 released checkpoints from [Hugging Face](https://huggingface.co/KaiwenDu/robust-overfitting-checkpoints). The checkpoint at epoch 105 achieved the highest measured PGD-20 robust accuracy; `epoch_200.pt` is the final checkpoint.

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
├── goals.md                           # Weekly goals, objectives, and expectations
├── progress.md                        # Weekly progress reports
├── proposal.md                        # Project proposal document
├── README.md                          # Project documentation and setup
├── setup_lambda_labs.md               # Cloud GPU setup guide for Lambda Labs
└── requirements.txt                   # Python package dependencies
```

---

## Weekly Goals & Progress

* [`goals.md`](goals.md): tracks our weekly research objectives, detailed action items, and expectations.
* [`progress.md`](progress.md): tracks our weekly execution logs, progress notes, and deliverables.

---

## Research Team
* Principal Investigator: Dr. Nicholas Q. Tran (Department of Mathematics and Computer Science)
* Student Researcher: Kaiwen Du (Computer Science)

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
