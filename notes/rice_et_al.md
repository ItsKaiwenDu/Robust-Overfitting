# Notes: "Overfitting in Adversarially Robust Deep Learning" (Rice, Wong & Kolter, 2020)

## Main Idea

Rice et al. document **robust overfitting** in adversarial training: robust test performance improves early in training, reaches a best checkpoint, and then becomes worse even while robust training loss continues to decrease. This differs from standard training, where training longer often does not harm test performance as severely.

For CIFAR-10 with a PreActResNet-18 and L-infinity PGD adversarial training, their robust test error reached 43.2% shortly after first learning-rate decrease, then rose to 51.4% by end of training. In accuracy terms, this is same pattern as robust accuracy peaking and then declining.

## Key Terms

- **Robust overfitting:** Robust test accuracy peaks and later declines during adversarial training, while robust training loss continues to improve.
- **Best-versus-final gap:** The difference between best robust test result during training and final result. This measures robust-overfitting severity.
- **Early stopping:** Select checkpoint with strongest robust validation performance instead of automatically using final epoch.
- **Robust test error:** Error after an adversarial attack. It is inverse of robust accuracy: lower robust error means higher robust accuracy.

## Main Findings

- Robust overfitting appeared across SVHN, CIFAR-10, CIFAR-100, and ImageNet, under both L-infinity and L2 perturbation constraints.
- The timing was closely related to learning-rate decreases: robust test performance often improved briefly after first decrease and then worsened with additional training.
- Early stopping selected with a held-out validation set recovered most of benefit without using test set for model selection.
- On their experiments, common regularization and augmentation methods did not outperform early stopping when used alone. Semi-supervised data augmentation combined with early stopping was exception.
- Larger models still showed robust overfitting, so double descent did not explain it away.

## Why This Matters for Our Project

Rice et al. provides our baseline phenomenon and measurement plan: save checkpoints, evaluate robust accuracy throughout training, record peak epoch, and compare peak and final accuracy. Our original pixel-space PGD result reproduces this setup before we change perturbation frequency band.

The paper does not study image frequency. Our project extends it by asking whether allowed perturbation band changes timing or severity of this robust-overfitting curve.

## Relationship to Yu et al. (2022)

These papers are complementary, not duplicates. Rice et al. establishes that robust overfitting occurs and shows that early stopping is an effective practical response. Yu et al. later use data-ablation experiments to propose an explanation: under stronger adversaries, small-loss adversarial training examples can drive later decline. Yu et al. then propose MLCAT to reduce that effect.

## Works Cited

Rice, Leslie, Eric Wong, and J. Zico Kolter. "Overfitting in Adversarially Robust Deep Learning." *Proceedings of 37th International Conference on Machine Learning*, PMLR 119, 2020.
