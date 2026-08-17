# Notes: "Adversarial Training and Robustness for Multiple Perturbations" (Tramèr & Boneh, 2019)

## Main Idea

Tramèr and Boneh ask whether one classifier can be adversarially robust to more than one perturbation type at the same time. They show that robustness to one threat model does not automatically transfer to another, and that there can be a real trade-off: improving robustness to one type can reduce robustness to another.

They study unions of attacks such as different norm-bounded perturbations (for example, L-infinity and L1) and pixel-space perturbations combined with rotations/translations. Their main contribution is both conceptual and experimental: define what multi-threat robustness means, demonstrate the trade-offs, and compare simple ways to train against multiple attacks.

## Key Terms

- **Perturbation model / threat model:** The set of changes an attacker is allowed to make to an input. A pixel-space PGD attack and a frequency-restricted DCT attack are different threat models.
- **Union robustness:** Robustness against an attacker allowed to choose any perturbation from any of several threat models. For each input, the attacker uses whichever permitted attack is most harmful.
- **Average robustness:** The average robust error or accuracy across the individual attack types. This can look good even when the model fails badly on one attack.
- **Worst-case (union) robustness:** Performance against the strongest allowed attack for each input. This is the stricter security-oriented measure.
- **Mutually Exclusive Perturbations (MEPs):** A theoretical case in which robustness to one attack type necessarily conflicts with robustness to another.

## Multi-Perturbation Training Strategies

For each training input, the authors generate adversarial examples under every considered perturbation type. They compare two simple aggregation strategies:

1. **AVG strategy:** Train on adversarial examples from all attack types. This treats the attack types as additional adversarial training data and aims to reduce average error across them.
2. **MAX strategy:** Generate adversarial examples from all attack types, then train only on the one with the highest loss for that input. This is intended to approximate the worst-case loss over the union of threat models.

These are not random alternating strategies. Both require considering all attacks for each training input before deciding how to update the model.

## Important Findings

- Models adversarially trained against one perturbation type can remain weak against another type, and can sometimes become more vulnerable to it.
- Multi-attack training has a cost relative to training a separate model for a single attack type. The authors observe a noticeable robustness trade-off on MNIST and CIFAR-10.
- AVG and MAX offer different objectives: AVG emphasizes typical performance across attacks, while MAX emphasizes the most damaging allowed attack.
- Strong evaluation matters. The authors identify cases where first-order attacks give a misleading impression of robustness because of gradient masking.

## Why This Matters for Our Project

This paper gives the conceptual framework for treating pixel-space PGD and low-frequency PGD as two distinct perturbation models. It also explains why a model trained on both should be evaluated against both individually and against their per-example worst case, rather than reporting only one robust-accuracy curve.

Our proposed experiment is deliberately simpler than AVG or MAX training. We plan to choose one training attack domain per epoch, either pixel-space PGD or frequency-restricted PGD, using a reproducible randomized schedule. The goal is not to claim the strongest union-robust model. Instead, we ask whether alternating threat models changes the timing or severity of **robust overfitting**.

This paper motivates two necessary controls:

- Compare the mixed-domain model with pixel-only and low-frequency-only training baselines.
- Evaluate each checkpoint with both pixel-space and low-frequency attacks, and optionally report the lower of the two accuracies as a simple union-robustness summary.

## Important Caution

Pixel-space PGD and low-frequency DCT-constrained PGD are closer to one another than the heterogeneous attack pairs emphasized in this paper (such as pixel noise and rotations). We should therefore treat a robustness trade-off as an empirical possibility to test, not an inevitable result.

## Works Cited

Tramèr, Florian, and Dan Boneh. "Adversarial Training and Robustness for Multiple Perturbations." *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, pp. 5866-5876. arXiv:1904.13000.
