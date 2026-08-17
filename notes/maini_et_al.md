# Notes: "Adversarial Robustness Against the Union of Multiple Perturbation Models" (Maini, Wong & Kolter, 2020)

## Main Idea

Maini, Wong, and Kolter study adversarial training when the attacker can choose from a union of several perturbation models. They argue that simple multi-attack aggregation can be difficult to tune and can leave a model robust to some attacks but weak to others.

Their main method, **Multi Steepest Descent (MSD)**, extends PGD so that a single adversarial example is optimized over multiple norm-bounded threat models. At each PGD step, MSD considers the candidate steepest-ascent step for every threat model and continues from the candidate that currently produces the highest loss. The resulting adversarial example approximates the worst-case example in the union of the allowed threat sets.

## Key Terms

- **Union of perturbation models:** A threat set made by combining several allowed perturbation sets. The attacker may choose the most damaging perturbation from any member of the union.
- **Inner maximization:** The attack-generation part of adversarial training: find the allowed perturbation that maximizes loss for the current model and input.
- **Steepest-ascent direction:** For a given constraint geometry, the allowed small step that most increases loss. Different norms produce different directions.
- **MSD:** Multi Steepest Descent. An iterative attack that chooses the worst candidate step among the threat models at every iteration.
- **MAX training:** Generate one PGD attack for each threat model, then train on the final adversarial example with highest loss.
- **AVG training:** Generate attacks for every threat model and train on all of them as adversarial augmentation.

## What MSD Changes

Standard PGD uses one perturbation geometry throughout the attack. For example, pixel-space L-infinity PGD repeatedly takes the sign of the pixel gradient and projects back into an L-infinity ball.

MSD instead does the following at each attack iteration:

1. Starting from the current perturbation, calculate one candidate projected step for each threat model.
2. Evaluate the loss for the candidates.
3. Keep the candidate that gives the highest loss.
4. Use the final MSD adversarial example for the model update.

This is more tightly connected to the worst-case union objective than first completing a separate attack for each model and then aggregating their final outputs.

## Important Findings

- On their MNIST and CIFAR-10 experiments, MSD achieved stronger worst-case robustness across unions of L-infinity, L2, and L1 attacks than their compared AVG and MAX baselines.
- MAX and AVG can converge to imbalanced, dataset-dependent trade-offs: performance may be strong under one attack but poor under another.
- Robustness should be evaluated against the **worst attack per input** in the union, not only by averaging the separately measured accuracies.
- Combining multiple threat models requires careful attack evaluation; weak attacks can conceal gradient masking or other apparent robustness.

## Why This Matters for Our Project

MSD is not the method we are currently proposing to implement. It chooses a worst-case threat-model direction **within every PGD attack step**, whereas our proposed experiment chooses the training domain, either pixel-space PGD or low-frequency frequency-domain PGD, once per epoch according to a reproducible random schedule.

The paper is nevertheless important for experimental design. It tells us that a mixed-training model must be evaluated under both component threats. A mixed model that improves under low-frequency PGD but declines under pixel-space PGD has not necessarily become more robust overall.

For our robust-overfitting analysis, we should therefore track, at each saved checkpoint:

1. clean accuracy;
2. robust accuracy against pixel-space PGD;
3. robust accuracy against low-frequency PGD; and
4. optionally, a union score based on the per-example worst result across the two attacks.

The proposed outcome is not a claim that randomized training improves union robustness. The outcome of interest is whether each curve's peak epoch and peak-to-final decline differ from the pixel-only baseline.

## Important Caution

The paper's attacks are unions of Lp-norm threat models. A frequency-restricted attack requires its own well-specified constraint and projection rule, so MSD cannot be copied unchanged into our code. Its contribution to our project is the union-robustness framework and the evaluation standard, not a drop-in implementation.

## Works Cited

Maini, Pratyush, Eric Wong, and J. Zico Kolter. "Adversarial Robustness Against the Union of Multiple Perturbation Models." *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, pp. 6640-6650, 2020. arXiv:1909.04068.
