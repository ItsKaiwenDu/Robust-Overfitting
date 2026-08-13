# Notes: "Towards Building More Robust Models with Frequency Bias" (Bu, Huang & Cui, 2023)

> **Note:** This is supporting background on why frequency may affect robust learning, but not a direct test of low-, middle-, and high-band perturbations under a fixed setup.

## Main Idea

Bu et al. start from the observation that adversarially trained CNNs tend to rely more on **low-frequency** information than standard CNNs. They ask whether deliberately increasing this low-frequency preference can improve adversarial robustness.

Their answer is the **Frequency Preference Control Module (FPCM)**. Rather than filtering the input image, FPCM is inserted inside a CNN and reweights the low- and high-frequency parts of intermediate feature maps. The goal is to retain useful high-frequency detail while making the model rely more on low-frequency features.

## Key Terms

- **Frequency bias:** a model's tendency to rely more on some frequency components than others.
- **Low frequency:** coarse, smooth, large-scale image structure.
- **High frequency:** fine detail, sharp edges, and texture.
- **FPCM:** a module that low-pass filters an internal feature map, then learns how much low- and high-frequency information to keep in each channel.
- **Cutoff frequency:** the boundary that determines how much high-frequency information the low-pass filter removes.

## What FPCM Does

For each selected internal feature map, FPCM:

1. transforms it to the frequency domain;
2. makes a low-frequency version with a Gaussian low-pass filter;
3. mixes that version with the remaining high-frequency information using learned channel weights; and
4. sends the reweighted features to the rest of the network.

The weights are biased toward low-frequency information but do not completely discard high-frequency information. Bu et al. place FPCM near the end of each ResNet stage, where they observe high-frequency feature content accumulating.

They also gradually lower the cutoff frequency during training, so later training emphasizes lower-frequency information more strongly. The paper states that the same cutoff schedule is used for the perturbations generated during training; evaluation uses a fixed low cutoff.

## Important Findings

- In their CIFAR-10 ResNet-18 experiments, adding FPCM to PGD adversarial training improved PGD-20 robust accuracy from 50.17% to 52.50% and AutoAttack accuracy from 47.42% to 49.26%.
- When more high-frequency information was removed, clean accuracy generally fell while robust accuracy rose. This shows a robustness-clean-accuracy trade-off rather than a free improvement.
- Their robust-accuracy curves show that models given very little low-frequency weight perform worse throughout training. The authors interpret this as evidence that low-frequency features help the robust-learning process, not only the final robust accuracy.

## Why This Matters for Our Project

This paper is useful because it gives a concrete reason to expect frequency to affect both the **level** and the **trajectory** of robust accuracy during adversarial training. It uses CIFAR-10, ResNet-18, PGD-style training, and frequency-related training curves, all close to our setting.

However, it is not a direct answer to our question. It mainly pushes the model toward low-frequency internal features, changes the architecture with FPCM, and changes the cutoff schedule during training. It does not hold every other part of training fixed while comparing low-, middle-, and high-frequency **perturbation bands**. It also does not measure a peak robust-accuracy epoch or post-peak decline.

So we can cite Bu et al. as motivation for the hypothesis that frequency should change robust-accuracy curves, but not as evidence that any particular band will have the earliest peak or the largest robust-overfitting drop.

**Takeaway:** Low-frequency information appears important for robust learning, but our experiment is needed to test how restricting the *attack itself* to low, middle, or high bands changes robust overfitting.

## Works Cited

Bu, Q., Huang, D., & Cui, H. (2023). *Towards building more robust models with frequency bias.* Proceedings of the IEEE/CVF International Conference on Computer Vision.
