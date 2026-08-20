# Notes: "Adversarial Robustness Against the Union of Multiple Perturbation Models" (Maini, Wong & Kolter, 2020)

## Main Idea

Maini, Wong, and Kolter study adversarial training when attacker can choose from a union of several perturbation models. They argue that simple multi-attack aggregation can be difficult to tune and can leave a model robust to some attacks but weak to others.

Their main method, **Multi Steepest Descent (MSD)**, extends PGD so that a single adversarial example is optimized over multiple norm-bounded threat models. At each PGD step, MSD considers candidate steepest-ascent step for every threat model and continues from candidate that currently produces highest loss. The resulting adversarial example approximates worst-case example in union of allowed threat sets.

## Key Terms

- **Union of perturbation models:** A threat set made by combining several allowed perturbation sets. The attacker may choose most damaging perturbation from any member of union.
- **Inner maximization:** The attack-generation part of adversarial training: find allowed perturbation that maximizes loss for current model and input.
- **Steepest-ascent direction:** For a given constraint geometry, allowed small step that most increases loss. Different norms produce different directions.
- **MSD:** Multi Steepest Descent. An iterative attack that chooses worst candidate step among threat models at every iteration.
- **MAX training:** Generate one PGD attack for each threat model, then train on final adversarial example with highest loss.
- **AVG training:** Generate attacks for every threat model and train on all of them as adversarial augmentation.

## What MSD Changes

Standard PGD uses one perturbation geometry throughout attack. For example, pixel-space L-infinity PGD repeatedly takes sign of pixel gradient and projects back into an L-infinity ball.

MSD instead does following at each attack iteration:

1. Starting from current perturbation, calculate one candidate projected step for each threat model.
2. Evaluate loss for candidates.
3. Keep candidate that gives highest loss.
4. Use final MSD adversarial example for model update.

This is more tightly connected to worst-case union objective than first completing a separate attack for each model and then aggregating their final outputs.

## Important Findings

- On their MNIST and CIFAR-10 experiments, MSD achieved stronger worst-case robustness across unions of L-infinity, L2, and L1 attacks than their compared AVG and MAX baselines.
- MAX and AVG can converge to imbalanced, dataset-dependent trade-offs: performance may be strong under one attack but poor under another.
- Robustness should be evaluated against **worst attack per input** in union, not only by averaging separately measured accuracies.
- Combining multiple threat models requires careful attack evaluation; weak attacks can conceal gradient masking or other apparent robustness.

## Why This Matters for Our Project

MSD is not method we are currently proposing to implement. It chooses a worst-case threat-model direction **within every PGD attack step**, whereas our proposed experiment chooses training domain, either pixel-space PGD or low-frequency frequency-domain PGD, once per epoch according to a reproducible random schedule.

The paper is nevertheless important for experimental design. It tells us that a mixed-training model must be evaluated under both component threats. A mixed model that improves under low-frequency PGD but declines under pixel-space PGD has not necessarily become more robust overall.

For our robust-overfitting analysis, we should therefore track, at each saved checkpoint:

1. clean accuracy;
2. robust accuracy against pixel-space PGD;
3. robust accuracy against low-frequency PGD; and
4. optionally, a union score based on per-example worst result across two attacks.

The proposed outcome is not a claim that randomized training improves union robustness. The outcome of interest is whether each curve's peak epoch and peak-to-final decline differ from pixel-only baseline.

## Important Caution

The paper's attacks are unions of Lp-norm threat models. A frequency-restricted attack requires its own well-specified constraint and projection rule, so MSD cannot be copied unchanged into our code. Its contribution to our project is union-robustness framework and evaluation standard, not a drop-in implementation.

## Works Cited

Maini, Pratyush, Eric Wong, and J. Zico Kolter. "Adversarial Robustness Against Union of Multiple Perturbation Models." *Proceedings of 37th International Conference on Machine Learning*, PMLR 119, pp. 6640-6650, 2020. arXiv:1909.04068.
