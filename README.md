# Investigating Robust Overfitting in Adversarial Training

This is GitHub repository for research on robust overfitting in adversarial training.

---

## Project Overview
Deep neural networks can be easily fooled by adversarial attacks, which are small, hidden changes to inputs that cause model to make wrong predictions. Adversarial training helps fix this, but models often run into a problem known as **robust overfitting**. This means that later in training, model's performance on test attacks gets worse even though its training loss keeps improving.

Research Objectives:
* Replicate adversarial training results from Rice et al. (2020) paper using a PreActResNet-18 model on CIFAR-10.
* Find exact point where model stops learning real robustness and starts memorizing specific perturbation patterns.

---

## Research Team
* Principal Investigator: Dr. Nicholas Q. Tran (Department of Mathematics and Computer Science)
* Student Researcher: Kaiwen Du (Computer Science)

---

## Week 1 (Completed)
* **What we did:** Read and analyzed four papers for this project:
  1. "Explaining and Harnessing Adversarial Examples" by Goodfellow, Shlens, and Szegedy
  2. "Overfitting in Adversarially Robust Deep Learning" by Rice, Wong, and Kolter
  3. "Design and Analysis of a Watermarking System for Care Labels" by Ragan-Kelley and Tran
  4. "The Normalized Compression Distance and Image Distinguishability" by Tran
* **Deliverable:** Reading notes have been written and are available in [`Notes/`](Notes/) directory:
  * [`Notes/Goodfellow.md`](Notes/Goodfellow.md): Covers linearity hypothesis, FGSM, adversarial training results on MNIST, and connections to our research.
  * [`Notes/Rice.md`](Notes/Rice.md): Covers robust overfitting phenomenon, effect of early stopping, comparison of regularization methods, and connections to our research.
  * [`Notes/Ragankelley.md`](Notes/Ragankelley.md): Covers a digital watermarking method for embedding care labels directly in fabric patterns and analyzes its visual distortion and robustness to fading.
  * [`Notes/Tran.md`](Notes/Tran.md): Covers an evaluation of Normalized Compression Distance (NCD) for predicting human visual similarity and its limitations in digital watermarking.

---

## Week 2 (Current)
* **Objective:** Set up the local software environment, configured dependencies, implemented the PreActResNet-18 model architecture, and documented the cloud setup guide.
* **Deliverables & Setup Files:**
  * [`.gitignore`](.gitignore): Configured to ignore environment folders, datasets, cache files, and model checkpoints.
  * [`requirements.txt`](requirements.txt): Lists project dependencies (`torch`, `torchvision`, `numpy`, `matplotlib`, `tensorboard`).
  * [`Models/preact_resnet.py`](Models/preact_resnet.py): PyTorch implementation of `PreActResNet18` tailored for CIFAR-10 with standard pre-activation blocks, final BatchNorm and ReLU, and correct dimensions.
  * [`verify_setup.py`](verify_setup.py): Script to verify python imports, system device capabilities (CPU/MPS/CUDA), and run a forward pass sanity check.
  * [`cloud_setup.md`](cloud_setup.md): Guide for deploying runs to cloud instances (Lambda Labs), ssh configurations, code syncing, `tmux` sessions, and TensorBoard port forwarding.

### Local Setup Instructions

1. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   ```
2. **Activate & install dependencies**:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Verify setup**:
   ```bash
   python verify_setup.py
   ```

---

## Week 3 (Up Next)
* **Objective:** Implement PGD-based adversarial training pipeline and verify on a small test run.

---

## Week 4
* **Objective:** Run full baseline adversarial training on CIFAR-10, verify overfitting is reproducible.

---

## Week 5
* **Objective:** Track training and test robust accuracy across epochs to identify the overfitting point.

---

## Week 6
* **Objective:** Run additional training runs with different random seeds to verify overfitting point consistency.

---

## Week 7
* **Objective:** Analyze results, produce learning curve visualizations, and draft findings.

---

## Week 8
* **Objective:** Complete the final report and prepare for publication and poster presentation.

---

## References
* Goodfellow, I. J., Shlens, J., and Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
* Ragan-Kelley, B., and Tran, N. (2007). *Design and Analysis of a Watermarking System for Care Labels.* PCM.
* Rice, L., Wong, E., and Kolter, J. Z. (2020). *Overfitting in adversarially robust deep learning.* ICML.
* Tran, N. (2007). *The Normalized Compression Distance and Image Distinguishability.* Proceedings of SPIE, Human Vision and Electronic Imaging XII.