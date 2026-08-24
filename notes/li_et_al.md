# Notes: "DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in Frequency Domain" (Li et al., 2024)

> **Note:** This is optional background on frequency-domain robustness, not a core paper for our low/middle/high-band robust-overfitting experiment.

## Main Idea

Li et al. study adversarial training through an image's Fourier **amplitude** and **phase**. They observe that attacks appear to damage phase patterns more than amplitude patterns, and propose **Dual Adversarial Training (DAT)** to make model rely less on amplitude and more on phase.

DAT creates a new version of each training image by keeping its phase but mixing its amplitude with an adversarially generated amplitude. The model is adversarially trained on both original and recombined images.

## Key Terms

- **Amplitude:** size or strength of each Fourier component. It is often associated with image appearance or style.
- **Phase:** alignment of Fourier components. It often carries more of an image's spatial structure and object information.
- **Recombined image:** an image reconstructed with original image's phase and a mixed amplitude.
- **Adversarial Amplitude Generator (AAG):** a small generator trained to produce an amplitude that makes learning harder, so classifier cannot depend too much on amplitude.
- **Split Batch Normalization:** separate batch-normalization statistics for original and recombined images, whose distributions differ.

## What Paper Observes

The authors form hybrid test images: one has adversarial image's phase and clean image's amplitude; other has adversarial amplitude and clean phase. Their standard CIFAR-10 model performs worse when adversarial **phase** is used. They take this as evidence that attacks affect phase-related information more strongly.

This is an experimental observation from their setup, not proof that phase always contains all semantic information or that amplitude never matters. In fact, DAT keeps some original amplitude because completely replacing it hurts performance.

## What DAT Changes

DAT is much more than a frequency mask:

1. The AAG creates a harmful amplitude for each training image.
2. DAT mixes that amplitude with image's own amplitude and keeps original phase.
3. It trains on PGD-style adversarial examples from both original and recombined images, with a consistency loss that encourages both versions to receive same prediction.
4. It uses split batch normalization and a modified adversarial-example loss to reduce number of attack steps needed during training.

The authors also give a conditional theoretical result: under their augmentation assumption and a linear softmax classifier on top of learned features, weights on amplitude-derived features shrink. This supports their proposed mechanism, but it is not a universal proof that all robust models should ignore amplitude.

## Important Results

For ResNet-18 on CIFAR-10, DAT reports 57.55% robust accuracy under PGD-20 and 51.36% under AutoAttack, vs. 51.30% and 47.63% for their PGD-AT baseline. DAT also reports higher clean accuracy (84.17% vs. 82.78%).

Their ablations show that AAG, amplitude mix-up, and split batch normalization all matter; removing split batch normalization produces largest robustness drop in their table.

## Why This Matters for Our Project

This paper is useful evidence that *how* information is changed in frequency domain can affect adversarial robustness. It also uses CIFAR-10, ResNet-18, and PGD-style adversarial training, so its setting is close enough to help motivate our work.

But it does **not** answer our research question directly. Amplitude vs. phase is different from low-, middle-, and high-frequency bands. DAT changes several things at once - a generator, image augmentation, losses, and batch-normalization layers - while our experiment should keep training and evaluation setup fixed and change only allowed perturbation band.

Li et al. mention robust overfitting only as a limitation: DAT without AWP must be trained for a limited number of epochs. They do not compare peak robust-accuracy epochs or post-peak decline across frequency bands. That is gap our robust-accuracy curves address.

**Takeaway:** Li et al. show that frequency-domain structure can help robustness, but our experiment isolates a cleaner question: which *allowed perturbation frequency band* changes robust overfitting, when everything else stays same?

## Works Cited

Li, F., Li, K., Wu, H., Tian, J., & Zhou, J. (2024). *DAT: Improving adversarial robustness via generative amplitude mix-up in frequency domain.* Advances in Neural Information Processing Systems, 38.
