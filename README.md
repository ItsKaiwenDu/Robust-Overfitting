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
* Graduate Collaborator: Milo Guan (M.S. Computer Science), Fidelity Measurement (Voluntary)

---

## Week 1 (Current)
* **What we did:** Read and analyzed four papers for this project:
  1. "Explaining and Harnessing Adversarial Examples" by Goodfellow, Shlens, and Szegedy
  2. "Overfitting in Adversarially Robust Deep Learning" by Rice, Wong, and Kolter
  3. "Design and Analysis of a Watermarking System for Care Labels" by Ragan-Kelley and Tran
  4. "The Normalized Compression Distance and Image Distinguishability" by Tran
* **Deliverable:** Reading notes have been written and are available in [`Notes/`](Notes/) directory:
  * [`Notes/Goodfellow.md`](Notes/Goodfellow.md): Covers linearity hypothesis, FGSM, adversarial training results on MNIST, and connections to our research.
  * [`Notes/Rice.md`](Notes/Rice.md): Covers robust overfitting phenomenon, effect of early stopping, comparison of regularization methods, and connections to our research.
  * [`Notes/Ragankelley.md`](Notes/Ragankelley.md): Summary only (full notes in progress).
  * [`Notes/Tran.md`](Notes/Tran.md): Summary only (full notes in progress).

---

## Week 2 (Up Next)
* **What we are doing:** Setting up the software environment, cloud computing resources, and model implementations.

> Note: Code, setup instructions, and data will be added to this repository once we begin the implementation phase next week.

---

## References
* Goodfellow, I. J., Shlens, J., and Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
* Ragan-Kelley, B., and Tran, N. (2007). *Design and Analysis of a Watermarking System for Care Labels.* PCM.
* Rice, L., Wong, E., and Kolter, J. Z. (2020). *Overfitting in adversarially robust deep learning.* ICML.
* Tran, N. (2007). *The Normalized Compression Distance and Image Distinguishability.* Proceedings of SPIE, Human Vision and Electronic Imaging XII.