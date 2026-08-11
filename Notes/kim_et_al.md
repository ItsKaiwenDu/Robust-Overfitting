# Notes: "Phase-shifted Adversarial Training" (Kim, Kim, Seo & Shin, 2023)

## Main Idea

Kim et al. study adversarial training (AT) through the frequency principle: neural networks tend to learn lower-frequency parts of a target function before higher-frequency parts. They show, under mathematical assumptions, that this principle also holds during adversarial training.

Their experiments on CIFAR-10 show that ordinary adversarial training learns high-frequency information more slowly than standard training. The authors propose PhaseAT, which shifts selected high-frequency components into a lower-frequency range so the model can learn them faster. PhaseAT improved clean and robust accuracy in their experiments.

## Key Terms

- **Frequency principle:** Neural networks usually learn smoother, lower-frequency patterns before finer, higher-frequency patterns.
- **Response frequency:** Frequency of the model's input-output mapping. This is the kind of frequency analyzed in this paper.
- **PhaseAT:** An adversarial-training method that uses a phase-shifted, multi-headed network to help the model learn high-frequency response components faster.
- **Adaptive attack:** An evaluation attack designed with knowledge of a defense. Kim et al. use this to test whether PhaseAT remains robust when its stochastic frequency-selection mechanism is accounted for.

## Why This Matters for Our Project

This paper gives a useful reason to study frequency in adversarial training: frequency-related behavior can affect how quickly robust models learn and how robust they become. It also uses CIFAR-10 and evaluates a ResNet-18 model against PGD and AutoAttack, which makes its setting closer to ours than the Guo et al. paper.

However, the paper studies the frequency of the **model response**, not the frequency band of the **perturbation**. PhaseAT changes the training method and model architecture; it does not train standard PGD with perturbations restricted to low-, middle-, or high-frequency image bands. It also does not measure robust overfitting as peak robust accuracy followed by a later decline.

Our project therefore asks a different question: with the training and evaluation setup otherwise held constant, does the frequency band allowed in the adversarial perturbation change the timing or severity of robust overfitting?

## Important Results

- In their CIFAR-10 frequency-error analysis, PhaseAT learned high-frequency response components faster than ordinary adversarial training, while their low-frequency errors were similar.
- On CIFAR-10, PhaseAT had higher clean and robust accuracy than the compared baselines under their experimental settings.
- PhaseAT was evaluated against adaptive PGD attacks that account for its stochastic frequency selection. Its robustness decreased somewhat under the stronger adaptive attack but remained substantial.

These results support frequency as a meaningful factor in adversarial training, but they do not predict which perturbation band will have the earliest robust-accuracy peak or the largest post-peak decline in our experiments.

## Works Cited

Kim, Yeachan, Seongyeon Kim, Ihyeok Seo, and Bonggun Shin. "Phase-shifted Adversarial Training." *Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence*, PMLR 216:1068-1077, 2023.
