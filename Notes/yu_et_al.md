# Notes: "Understanding Robust Overfitting of Adversarial Training and Beyond" (Yu, Han, Shen, Yu, Gong, Gong & Liu, 2022)

## Main Idea

Yu et al. investigate why robust overfitting occurs in adversarial training. They compare training with weak and strong adversaries, where adversary strength means the allowed perturbation size (epsilon). Robust overfitting appeared under stronger adversaries but not under weaker ones.

Their data-ablation experiments attribute robust overfitting under a strong adversary to some **small-loss adversarial training examples**: examples whose current adversarial loss has become low enough that they are no longer challenging for the model. Removing these examples eliminated robust overfitting in their experiments, while removing large-loss examples did not.

## Key Terms

- **Robust overfitting:** Robust test accuracy reaches a peak and then decreases as adversarial training continues.
- **Robustness gap:** Best robust accuracy during training minus robust accuracy at the final epoch. This is the paper's measure of robust-overfitting severity.
- **Small-loss example:** A training example whose adversarial loss is below a chosen threshold. It is easy for the current model even after attack.
- **Strong adversary:** An adversary with a larger allowed perturbation budget. In this paper, this refers to perturbation size, not frequency band.

## Why This Matters for Our Project

This paper is directly relevant because it studies robust-overfitting curves using PGD adversarial training of PreActResNet-18 on CIFAR-10. It also defines severity in the same useful way as our project: the difference between peak and final robust accuracy.

It gives a possible mechanism to examine after our experiments: a frequency band may produce different robust-overfitting curves because it changes how many training perturbations become small-loss examples over time. This is only a possible explanation; Yu et al. did not test frequency-restricted perturbations.

The paper changes perturbation magnitude, not frequency. It does not compare low-, middle-, and high-frequency bands, so it does not answer our research question. Our project holds the perturbation budget and training setup fixed while changing the band, then compares the peak epoch and post-peak decline.

## Relationship to Rice et al. (2020)

These papers are complementary, not duplicates. Rice et al. first documents robust overfitting and establishes early stopping as a strong practical baseline. Yu et al. later investigate a possible cause with data-ablation experiments and propose MLCAT as a mitigation.

## Important Results

- With standard PGD adversarial training, the authors observed robust overfitting under stronger perturbation budgets.
- Removing small-loss examples eliminated robust overfitting in their ablations; removing large-loss examples did not.
- They proposed minimum loss constrained adversarial training (MLCAT), which raises the loss of small-loss examples instead of discarding them. Its weight-perturbation version greatly reduced the robustness gap and improved robustness in their tests.

MLCAT changes the training objective, so it is outside the scope of our controlled band comparison. Its central lesson for us is to record robust-accuracy curves carefully and, if needed, inspect the distribution of adversarial training losses by frequency band.

## Works Cited

Yu, Chaojian, Bo Han, Li Shen, Jun Yu, Chen Gong, Mingming Gong, and Tongliang Liu. "Understanding Robust Overfitting of Adversarial Training and Beyond." *Proceedings of the 39th International Conference on Machine Learning*, PMLR 162, 2022.
