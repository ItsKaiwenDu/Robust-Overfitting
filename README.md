![Banner](assets/banner.png)

# Robust Overfitting Across Pixel-Space and Low-Frequency Adversarial Training

This repository contains the code, evaluation scripts, and experimental results for the research paper [*"Robust Overfitting Across Pixel-Space and Low-Frequency Adversarial Training"*](report/main.tex) (Du, September 2026).

---

## Project Overview
Deep neural networks can be highly accurate on unmodified images while being vulnerable to small, deliberately chosen perturbations. Adversarial training addresses this threat by optimizing a model on adversarially perturbed examples. However, robust performance often peaks and subsequently degrades as training continues—a phenomenon known as **robust overfitting** (Rice et al., 2020).

This study investigates how the perturbation domain used during adversarial training affects the timing and severity of robust overfitting. We train PreActResNet-18 models on CIFAR-10 across five seeds (42–46) under three controlled conditions:
1. **Pixel-only:** Standard pixel-space PGD-10 control.
2. **Low-frequency-only:** DCT-masked low-frequency PGD-10 (retaining an 8×8 top-left DCT coefficient block).
3. **Mixed-domain:** Seeded, independent equal-probability choice between pixel and low-frequency PGD at the start of each epoch.

### Key Contributions
* **Controlled Multi-Domain Evaluation:** Implements and evaluates three controlled adversarial-training conditions under matched architectures, optimization schedules, and budgets.
* **Dense Checkpoint Tracking:** Tracks all 40 saved checkpoints (epochs 5–200, every 5 epochs) across four accuracy metrics: clean, pixel-PGD-20, low-frequency DCT-masked PGD-20, and joint (union) accuracy.
* **Robust Overfitting Dynamics:** Measures exact peak epochs and post-peak declines (robust-overfitting gaps) across seeds and conditions.
* **Domain Specialization Discovery:** Demonstrates that once-per-epoch random domain mixing yields schedule-dependent domain specialization ($r = 0.971$) rather than stable simultaneous robustness across both evaluated attacks.

---

## Research Question & Hypothesis

**Research question:** During PGD adversarial training of PreActResNet-18 on CIFAR-10, how does independently selecting pixel-space or low-frequency DCT-masked PGD with equal probability at the start of each epoch affect the timing and severity of robust overfitting, compared with training exclusively against either attack?

**Hypothesis:** With architecture, dataset, optimization schedule, perturbation budget, and checkpoint-evaluation schedule held fixed, mixed-domain training will produce robust-accuracy curves that differ from those of the single-domain conditions. In particular, the peak epoch, peak shape, or post-peak decline may change under pixel-space and/or low-frequency evaluation.

---

## Experiment Design

### Model and Dataset

* **Model:** PreActResNet-18 (He et al., 2016), sourced from the Rice et al. (2020) codebase.
* **Dataset:** CIFAR-10 (50,000 training images, 10,000 test images across 10 classes, 32×32 resolution).

### Training Configurations

All conditions use the same PreActResNet-18 architecture, CIFAR-10 data, SGD optimizer (momentum 0.9, weight decay $5 \times 10^{-4}$), 200 epochs, MultiStep learning rate decay (0.1 decaying by 0.1 at epochs 100 and 150), 10-step training PGD ($\epsilon = 8/255$, $\alpha = 2/255$), and random seeds (42–46).

1. **Pixel-only:** Train with standard pixel-space PGD-10 in every epoch.
2. **Low-frequency-only:** Train with DCT-masked low-frequency PGD-10 in every epoch with cutoff 8 (64 out of 1,024 coefficients retained per channel).
3. **Mixed-domain:** Dedicated seeded RNG selects pixel or low-frequency PGD with equal probability ($p = 0.5$) at the start of each epoch; all batches in that epoch use the selected attack.

### Evaluation Configurations

For each run, all 40 saved checkpoints (epochs 5–200, evaluated every 5 epochs) are evaluated on the full 10,000-image CIFAR-10 test set across four metrics:

1. **Clean accuracy:** Evaluated on unmodified images without perturbation.
2. **Pixel-space robustness:** Evaluated against pixel-space PGD-20 ($\epsilon = 8/255$, $\alpha = 2/255$).
3. **Low-frequency robustness:** Evaluated against DCT-masked PGD-20 ($\epsilon = 8/255$, $\alpha = 2/255$, cutoff 8).
4. **Joint accuracy (Union robustness):** Per-image correctness under *both* separately generated pixel and low-frequency attacks. (Labeled as `union robustness` in CSV files and plot legends).

### Results Snapshot & Summary Table

#### Five-Seed Checkpoint Evaluation Results (Table 3 from paper)

**Panel A: Final Accuracy at Epoch 200 (Mean ± SD across 5 seeds, %)**

| Condition | Clean Acc (%) | Pixel-PGD-20 (%) | Low-Freq PGD-20 (%) | Joint Acc (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Pixel-only** | 84.40 ± 0.09 | 42.66 ± 0.22 | 76.48 ± 0.26 | 42.66 ± 0.22 |
| **Low-frequency-only** | 94.28 ± 0.22 | 0.00 ± 0.00 | 92.70 ± 0.28 | 0.00 ± 0.00 |
| **Mixed-domain** | 90.33 ± 3.45 | 19.33 ± 14.28 | 85.20 ± 4.29 | 19.33 ± 14.27 |

**Panel B: Peak Accuracy and Robust-Overfitting Gaps (ROGs, percentage points)**

| Condition | Pixel Peak (Epoch) | Pixel ROG (pp) | Low-Freq Peak (Epoch) | Low-Freq ROG (pp) |
| :--- | :---: | :---: | :---: | :---: |
| **Pixel-only** | 51.22% (Epoch 105) | 8.56 | 78.36% (Epoch 155) | 1.88 |
| **Low-frequency-only** | 0.00% (Epoch 5) | 0.00 | 92.74% (Epoch 195) | 0.05 |
| **Mixed-domain** | 40.80% (Epoch 85) | 21.47 | 86.92% (Epoch 195) | 1.72 |

* **Pixel-only Control:** Exhibits clear robust overfitting under pixel-PGD-20, peaking at **51.22%** at epoch 105 before falling by **8.56 pp** to **42.66%** by epoch 200. Every individual seed peaked at epoch 105.
* **Low-frequency-only Training:** Maintains high matched robustness late into training, peaking at **92.74%** at epoch 195 and finishing at **92.70%** (a tiny **0.05 pp** decline). However, transfer to pixel-space PGD-20 is **0.00%** across all checkpoints.
* **Mixed-domain Training:** Does not yield stable simultaneous robustness. It produces schedule-dependent oscillations between domain-specialized states: mean pixel robustness is **42.36%** immediately after pixel-training epochs versus **3.53%** after low-frequency-training epochs (point-biserial correlation $r = 0.971$). The apparent aggregate peak at epoch 85 occurs because all five seeds randomly trained on pixel PGD during epoch 85, whereas only seed 42 did so at epoch 200.

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
5. Run a quick diagnostic (1 epoch, 10% of data) to confirm training pipeline:
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
seed. Diagnostics add a `diagnostic/` level, so three one-epoch checks do
not overwrite one another:

```text
checkpoints/<training-mode>/[diagnostic/]<run-name>/
runs/<training-mode>/[diagnostic/]<run-name>/
report/<training-mode>/[diagnostic/]<run-name>/evaluation_results.csv
```

`<run-name>` defaults to `seed-<seed>` and can be changed with `--run-name`.
For example, run low-frequency diagnostic and then its matching evaluation:

```bash
python3 scripts/train.py --training-mode low-frequency-only --diagnostic --seed 42
python3 scripts/evaluate.py --training-mode low-frequency-only --diagnostic --seed 42
```

Mixed-domain checkpoints record selected domain for checkpoint epoch
and complete schedule so far, together with seed and DCT cutoff.

**Option A: Lambda Labs (Cloud)**

6a. See [`setup_lambda_labs.md`](setup_lambda_labs.md) for complete step-by-step guide, including instance provisioning, SSH access, code syncing, running training in background, monitoring with TensorBoard, downloading results, and terminating instance.

**Option B: Local**

6b. Run training:
```bash
python3 scripts/train.py
```
7b. Run evaluation across all checkpoints:
```bash
python3 scripts/evaluate.py
```
8b. Plot matching run's results (replace mode and seed as needed):
```bash
# Plot a single seed run (generates vector PDF figures)
python3 scripts/plot_results.py --training-mode pixel-only --seed 42

# Generate aggregate 5-seed curves and summaries in overall/
python3 scripts/plot_results.py --training-mode pixel-only --overall

# Group mixed-domain checkpoint results by the most recent training attack
python3 scripts/plot_results.py --training-mode mixed-domain --group-mixed-domain
```

> **Note:** The **CIFAR-10** dataset (~170 MB) will be downloaded automatically to `data/` on first training run. No manual download is required.

### Pretrained Checkpoints

All 600 checkpoints from the three training conditions are available on
[Hugging Face](https://huggingface.co/KaiwenDu/robust-overfitting-checkpoints).
They are organized by condition, seed (42–46), and epoch, with 40 checkpoints
per run from epochs 5 through 200. The original legacy pixel-only replication
is documented separately under `report/pixel-only/baseline/`.

### Report Implementation and Attack Checks

After required checkpoints and CIFAR-10 test data are available locally,
run report checks with:

```bash
python3 scripts/run_report_checks.py
```

The command performs three reproducible diagnostics without overwriting the
primary checkpoint-evaluation CSVs:

1. verifies DCT round-trip accuracy, masked projection, and perturbation budget;
2. measures out-of-mask DCT energy introduced by final image clipping for a
   low-frequency-PGD-20 attack; and
3. compares primary one-restart pixel PGD-20 attack with a three-restart,
   per-example maximum-loss PGD-50 attack at pixel-only seed-42 peak
   (epoch 105) and final (epoch 200) checkpoints.

By default, these are performed on a fixed, approximately class-balanced
256-image CIFAR-10 test subset selected with seed `20260912`. They are
diagnostic checks, not replacements for primary full-10,000-image,
one-restart PGD-20 checkpoint evaluations. The generated results are saved in
`report/checks/attack_strength_checks.csv` and
`report/checks/implementation_checks.json`.

---

## Project Directory Structure

```text
Robust-Overfitting/
├── checkpoints/                       # [Ignored] Checkpoints saved every 5 epochs (epochs 5–200)
│   ├── pixel-only/
│   │   ├── baseline/                  # Completed original pixel-PGD replication (epochs 5–200)
│   │   ├── diagnostic/seed-42/        # Local diagnostic checkpoint (1 epoch)
│   │   └── seed-42/ ... seed-46/      # Completed full 5-seed control checkpoints
│   ├── low-frequency-only/
│   │   ├── diagnostic/seed-42/        # Local diagnostic checkpoint (1 epoch)
│   │   └── seed-42/ ... seed-46/      # Completed full 5-seed run checkpoints
│   └── mixed-domain/
│       ├── diagnostic/seed-42/        # Local diagnostic checkpoint (1 epoch)
│       └── seed-42/ ... seed-46/      # Completed full 5-seed run checkpoints
├── models/                            # Model architecture definitions
│   ├── __init__.py                    # Exports PreActResNet variants
│   └── preact_resnet.py               # PreActResNet-18 model architecture in PyTorch
├── notes/                             # Reading literature notes
│   ├── bu_et_al.md                    # Literature notes on frequency bias in robust models
│   ├── chen_et_al.md                  # Literature notes on Shapley-value frequency domain analysis
│   ├── goodfellow_et_al.md            # Literature notes on FGSM and adversarial training
│   ├── guo_et_al.md                   # Literature notes on low-frequency adversarial perturbation
│   ├── kim_et_al.md                   # Literature notes on phase-shifted adversarial training
│   ├── li_et_al.md                    # Literature notes on DAT frequency domain amplitude mix-up
│   ├── maini_et_al.md                 # Literature notes on robustness against union of perturbations
│   ├── rice_et_al.md                  # Literature notes on robust overfitting
│   ├── tramer_et_al.md                # Literature notes on adversarial training for multiple perturbations
│   ├── xie_et_al.md                   # Literature notes on threat-aware frequency decoupling
│   └── yu_et_al.md                    # Literature notes on understanding robust overfitting
├── report/                            # Research paper, presentations, and evaluation outputs
│   ├── main.tex                       # LaTeX paper / technical report source
│   ├── slides.pptx                    # Research presentation slides
│   ├── checks/                        # Reproducible DCT/leakage and stronger-attack diagnostics
│   │   ├── attack_strength_checks.csv
│   │   └── implementation_checks.json
│   ├── pixel-only/
│   │   ├── baseline/                  # Completed legacy CSV, summary, and vector PDF plots
│   │   ├── diagnostic/seed-42/        # Diagnostic evaluation CSV
│   │   ├── seed-42/ ... seed-46/      # Completed per-seed CSVs, summaries, and vector PDF plots
│   │   └── overall/                   # Completed 5-seed aggregate curves (PDF) and summary CSV
│   ├── low-frequency-only/
│   │   ├── diagnostic/seed-42/        # Diagnostic evaluation CSV
│   │   ├── seed-42/ ... seed-46/      # Completed per-seed CSVs, summaries, and vector PDF plots
│   │   └── overall/                   # Completed 5-seed aggregate curves (PDF) and summary CSV
│   └── mixed-domain/
│       ├── diagnostic/seed-42/        # Diagnostic evaluation CSV
│       ├── seed-42/ ... seed-46/      # Completed per-seed CSVs, summaries, and vector PDF plots
│       └── overall/                   # Completed 5-seed aggregate curves (PDF) and summary CSV
├── scripts/                           # Python scripts for training, evaluation, plotting, and setup
│   ├── dct_pgd.py                     # Low-frequency DCT-masked PGD implementation
│   ├── evaluate.py                    # Four-metric checkpoint evaluation script (PGD-20)
│   ├── plot_results.py                # Evaluation & training plotting script (per-seed & aggregate vector PDF)
│   ├── run_report_checks.py           # DCT, leakage, perturbation-budget, and stronger-attack checks
│   ├── train.py                       # Core adversarial PGD training script
│   └── verify_setup.py                # Setup verification script
├── data/                              # [Ignored] CIFAR-10 dataset files (downloaded automatically)
├── runs/                              # [Ignored] TensorBoard logs, created and grouped like checkpoints/
│   ├── pixel-only/[diagnostic/]seed-<seed>/
│   ├── low-frequency-only/[diagnostic/]seed-<seed>/
│   └── mixed-domain/[diagnostic/]seed-<seed>/
├── run_logs/                          # [Ignored] Training log files from background runs
├── .gitignore                         # Files and folders ignored by Git
├── goals.md                           # Weekly goals, objectives, and expectations
├── LICENSE                            # Project license
├── progress.md                        # Weekly progress reports
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
