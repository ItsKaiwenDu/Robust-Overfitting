# Notes: "Phase-shifted Adversarial Training" (Kim, Kim, Seo & Shin, 2023)

## 1. Summary

This paper studies why adversarial training (AT) tends to converge slowly and often hurts performance on normal, unperturbed data. The authors look at this problem by focusing on frequency, meaning how quickly a network learns different components of the data when those components are broken down using a Fourier transform. They find that AT causes networks to learn high-frequency information much more slowly than standard training does, which results in unstable, oscillating predictions near each data point. To address this, they first prove that a known pattern in standard training, called the frequency principle (F-principle), where networks learn low-frequency information before high-frequency information, still holds under AT. Based on that theoretical result, they propose phase-shifted AT (PhaseAT), a method that deliberately shifts high-frequency components into the low-frequency range during training, where learning happens faster. Their experiments on CIFAR-10 and ImageNet-100 show that PhaseAT improves both standard and robust accuracy compared to several existing AT methods, even when evaluated against an attack specifically designed to target PhaseAT's mechanism.

## 2. Key Terms

- **F-Principle (Frequency Principle)**: The pattern where neural networks tend to learn the low-frequency, smoother parts of a target function before they learn the high-frequency, more detailed or oscillatory parts.
- **Adversarial Training (AT)**: A defense method that trains a model on adversarially perturbed inputs instead of clean ones, so the model learns to resist small, worst-case input changes.
- **PhaseDNN / Phase Shift**: A technique that shifts high-frequency content of a function into the low-frequency range before training, so a network can learn it with the same fast convergence normally reserved for low frequencies.
- **White-Box vs. Black-Box Attack**: A white-box attack assumes the attacker has full access to the model's internals, including gradients, while a black-box attack assumes the attacker can only query the model's outputs.
- **Adaptive Attack**: An attack specifically designed with knowledge of a particular defense's mechanism, used to test whether that defense is robust in a fair, non-overstated way rather than only being tested against generic attacks.

## 3. Problem: AT Converges Slowly

First of all, AT converges slowly means the error on a given piece of information, like a high-frequency component, stays high for many training epochs instead of dropping quickly.

The authors start by directly measuring how quickly a network learns different frequency components of the CIFAR-10 dataset during training, comparing standard training against AT. They do this by splitting the dataset's information into low-frequency and high-frequency parts using a filtering method based on the Fourier transform, then tracking the error in each part separately across training epochs.

The result is shown in their Figure 1: for low-frequency components, standard training and AT behave fairly similarly. For high-frequency components, however, AT converges noticeably more slowly than standard training. In practice, this means that the network is much slower at learning the fine, detailed structure of the data when it is also being trained to resist adversarial perturbations, and this slow convergence results in highly oscillatory predictions near each data point, meaning the model's output can change abruptly for very small changes in input, which is closely related to why the model remains vulnerable to adversarial examples in the first place.

## 4. F-Principle Holds in AT

The F-principle, the idea that networks generally learn low-frequency information before high-frequency information, had already been established for standard training in earlier work. A central contribution of this paper is proving mathematically that this same principle also holds in AT, not just observing it empirically.

To do this, the authors represent the total training loss in the frequency domain using the Fourier transform, then split that loss into a low-frequency part and a high-frequency part. Their main theorem states that the rate at which the high-frequency part of the loss decreases is bounded by a term that shrinks as frequency increases, meaning that higher-frequency components take mathematically longer to be learned by the network. This holds specifically under certain smoothness conditions on the network's activation function and the target function it is learning, described using Sobolev spaces, which are a mathematical way of describing how smooth or well-behaved a function is, including how many times it can be differentiated while staying bounded. The proof also holds specifically for perturbations bounded by the ℓ∞ norm, the same setting used throughout their experiments.

It is worth being precise about what kind of claim this is. Unlike the robust overfitting phenomenon in other papers, which was identified purely by observing training curves, this F-principle result in AT is a formally proven theorem, derived under stated mathematical assumptions, not just an empirical pattern. The authors do note, however, that the empirical observation in Figure 1 is what motivated them to investigate and ultimately prove this theorem, so the discovery process itself started empirically even though the final claim is a proof.

One additional detail from the theorem: networks with smoother activation functions, such as tanh or sigmoid, are predicted to converge even more slowly on high-frequency information than networks using less smooth activation functions like ReLU. This gives a concrete, testable link between a network's choice of activation function and how quickly it can learn fine-detail information under AT.

## 5. Method: PhaseAT

Motivated by their theoretical result, the authors build a method to directly speed up how quickly a network learns high-frequency information during AT, based on an earlier idea called PhaseDNN.

The core idea of phase-shifting is to take a high-frequency component of a function and mathematically shift it into the low-frequency range before training, exploiting the fact that low frequencies are learned faster. However, the original PhaseDNN was designed for one-dimensional data, such as physical wave signals, and could not scale to high-dimensional data like images, since the number of frequencies to track grows extremely fast as the number of input dimensions increases.

To adapt this idea for AT on image data, the authors make two main changes:

- **Dimensionality reduction via projection**: Instead of tracking frequency across every dimension of a high-dimensional image, they project each input onto its first principal component, a single direction that captures the most variation in the data, and analyze frequency along that one direction instead. This keeps the frequency analysis tractable for high-dimensional inputs.
- **Multi-headed architecture**: Instead of training a completely separate network for every frequency being tracked, which would use a large amount of memory, they use one shared feature-extracting network with several small, frequency-specific output heads attached to it. Each head is responsible for handling one shifted frequency range.

During training, the specific frequencies chosen for phase-shifting are not fixed in advance. Instead, the algorithm measures how much the Fourier coefficients differ between a clean batch of data and its adversarially perturbed version, and uses that difference to randomly sample which frequencies to shift for each head at each training step. The authors also include a regularization term that encourages the phase-shifted model to make different predictions than a version of the same model with no phase-shifting applied, specifically to prevent an attacker from being able to bypass PhaseAT's benefit by simply ignoring the phase-shift mechanism.

## 6. Results: Does PhaseAT Actually Improve Robustness?

Before testing performance, the authors first confirm that PhaseAT actually does what it claims. Repeating the same frequency-error measurement from their earlier analysis, they find that PhaseAT and standard AT perform similarly on low-frequency components, but PhaseAT converges noticeably faster on high-frequency components, matching their intended design.

On CIFAR-10, using both ResNet-18 and WideResNet-34-10 architectures, PhaseAT outperformed several existing AT methods, including PGD-AT, TRADES, GAIRAT, and AWP (Adversarial Weight Perturbation), across clean accuracy and robust accuracy under both PGD and AutoAttack (AA), a stronger, standardized attack benchmark. For example, on ResNet-18, PhaseAT improved standard accuracy by 5.3 percentage points and PGD-based robust accuracy by 8.5 percentage points compared to AWP, while remaining competitive on AutoAttack accuracy.

On the larger ImageNet-100 dataset, PhaseAT again showed comparable clean accuracy to the strongest baselines, while achieving a noticeably higher robust accuracy against AutoAttack than all other non-iterative methods tested.

An important part of the evaluation is that the authors test PhaseAT against an adaptive attack, meaning an attack designed with full knowledge of how PhaseAT works, including its strategy for randomly selecting which frequencies to shift. This is a stricter and more honest test than only using generic, off-the-shelf attacks, since a defense that only appears robust against attacks that don't know its inner mechanism can give a misleadingly optimistic result. Even under this adaptive attack, which the authors note costs about ten times more computation to generate than standard attacks, PhaseAT still maintained meaningfully strong robustness, though its accuracy dropped somewhat compared to being tested with non-adaptive attacks.

The authors also show that PhaseAT converges faster during training on both clean and robust accuracy compared to standard AT, and that its additional computational cost, from the Fourier transform step and the extra prediction heads, is small compared to its performance gains.

## 7. Works Cited

Kim, Yeachan, Seongyeon Kim, Ihyeok Seo, and Bonggun Shin. "Phase-shifted Adversarial Training." *Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence*, PMLR 216:1068–1077, 2023.