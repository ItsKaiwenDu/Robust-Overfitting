# Investigating Robust Overfitting in Adversarial Training

This is the GitHub repository for research on robust overfitting in adversarial training.

---

## Project Overview
Deep neural networks can be easily fooled by adversarial attacks, which are small, hidden changes to inputs that cause the model to make wrong predictions. Adversarial training helps fix this, but models often run into a problem known as **robust overfitting**. This means that later in training, the model's performance on test attacks gets worse even though its training loss keeps improving.

Research Objectives:
* Replicate the adversarial training results from the Rice et al. (2020) paper using a PreActResNet-18 model on CIFAR-10.
* Find the exact point where the model stops learning real robustness and starts memorizing specific perturbation patterns.

---

## Research Team
* Principal Investigator: Dr. Nicholas Q. Tran (Department of Mathematics and Computer Science)
* Student Researcher: Kaiwen Du (Computer Science)

---

## Week 1 (Current)
* **What we are doing:** Reading and analyzing the following papers:
  1. "Explaining and Harnessing Adversarial Examples" by Goodfellow, Shlens, and Szegedy
  2. "Overfitting in Adversarially Robust Deep Learning" by Rice, Wong, and Kolter
* **By the end of the week:** We will write down our notes and document what we learned from these articles.

> Note: Code, setup instructions, and data will be added to this repository once we begin the implementation phase next week.

---

## References
* Goodfellow, I. J., Shlens, J., and Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
* Rice, L., Wong, E., and Kolter, J. Z. (2020). *Overfitting in adversarially robust deep learning.* ICML.