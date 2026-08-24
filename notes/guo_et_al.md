# Notes: "Low Frequency Adversarial Perturbation" (Guo, Frank & Weinberger, 2019)

## Main Idea

Guo et al. study low-frequency adversarial perturbations. They use discrete cosine transform (DCT) to represent an image in frequency space, then restrict an attack so that it changes only low-frequency DCT coefficients. After an inverse DCT, these coefficient changes become smooth pixel-space perturbations.

The authors mainly study black-box attacks, where an attacker cannot access model gradients and must repeatedly query model. On ImageNet, limiting attack to low frequencies reduced number of queries needed by their tested attacks. Their results also show that a low-frequency subspace can still contain effective adversarial directions.

## Key Terms

- **DCT:** A transform that represents an image as frequency coefficients. Lower-frequency coefficients produce smoother, broader image changes; higher-frequency coefficients produce finer changes.
- **Frequency-restricted perturbation:** A perturbation made by allowing changes only to a chosen set of frequency coefficients, then converting those changes back to pixel space.
- **Black-box attack:** An attack that cannot access target model's gradients and instead learns from model queries. This is paper's main setting, not ours.

## Why This Matters for Our Project

This paper supports idea that frequency-restricted perturbations are feasible and meaningful. It gives a possible implementation pattern: transform a perturbation to DCT space, keep only selected coefficients with a frequency mask, and transform it back before applying it to image.

However, this paper does not study PGD adversarial training, CIFAR-10, robust overfitting, or comparisons among low-, middle-, and high-frequency bands. Our project extends frequency-restriction idea by asking whether different bands change when robust accuracy peaks and how much it declines afterward.

## Important Result

For their ImageNet black-box experiments, low-frequency versions of Boundary Attack and NES attack used fewer model queries than their standard pixel-space versions. This is evidence that restricting an attack to low frequencies does not automatically make it ineffective, but it does not predict how robust overfitting will differ across frequency bands during PGD adversarial training.

## Works Cited

Guo, Chuan, Jared S. Frank, and Kilian Q. Weinberger. "Low Frequency Adversarial Perturbation." *Proceedings of 35th Conference on Uncertainty in Artificial Intelligence (UAI 2019)*, 2019.
