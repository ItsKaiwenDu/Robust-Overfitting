# Notes: "Rethinking and Improving Robustness of Convolutional Neural Networks: a Shapley Value-based Approach in Frequency Domain" (Chen, Ren & Yan, 2022)

## Main Idea

Chen et al. use the discrete Fourier transform (DFT) and Shapley values to see how much does each frequency coefficient of one image help or hurt a Convolutional Neural Network's (CNN) score for the correct class?

They find that frequency contribution is not determined only by whether a coefficient is low or high frequency. A coefficient can help the correct prediction or hurt it, and this can vary across images and classes. This helps explain why attacks restricted to either low or high frequencies can still be effective.

## Key Terms

- **DFT:** A transform that represents an image as frequency coefficients. Low frequencies describe broad image structure; high frequencies describe finer detail.
- **Shapley value:** A way to estimate one component's average contribution to an outcome while accounting for combinations with other components. Here, the outcome is the model's score for the correct class.
- **Positive/negative frequency component (PFC/NFC):** A coefficient with a positive Shapley value helps the correct prediction; one with a negative value hurts it.
- **Frequency mask:** A mask that keeps selected frequency coefficients and zeros the rest before applying an inverse DFT. This is the same general mechanism we can use to restrict perturbations to a chosen band.

## Findings About Frequency and Robustness

For a standard-trained ResNet-18 on CIFAR-10, the authors found that low-frequency coefficients contributed positively on average for clean images, while high-frequency coefficients had near-zero average contribution. On adversarial examples, high-frequency coefficients had negative average contribution, suggesting that the standard-trained model was especially vulnerable to them.

For their adversarially trained model, high-frequency coefficients had near-zero average contribution for both clean and adversarial images. The authors interpret this as adversarial training gaining robustness partly by relying less on high-frequency information. They present this as a hypothesis about the clean-accuracy/robustness trade-off, not a proven causal explanation.

The paper also masks a generated PGD perturbation into low- or high-frequency parts. The high-frequency part was more effective than the low-frequency part in their standard-trained CIFAR-10 model, but the low-frequency part still caused many errors. PFCs and NFCs occurred across the spectrum, so frequency alone does not decide whether a coefficient helps or hurts the model.

## Why This Matters for Our Project

This is strong background for our design because it studies CIFAR-10, ResNet-18, PGD attacks, adversarial training, and frequency-masked perturbations. It supports treating frequency band as an experimental variable rather than assuming that only one band matters.

However, Chen et al. compare low and high frequencies only; they do not study a middle band. Their main goal is explaining frequency contributions and improving robustness with a class-wise augmentation method called CSA, not measuring robust-overfitting curves. They do not test whether frequency-restricted PGD adversarial training changes the peak robust-accuracy epoch or the post-peak decline.

Our project extends their frequency-masking idea by holding the training and evaluation setup constant, training with low-, middle-, or high-band perturbations, and comparing robust accuracy throughout training.

## Important Result

CSA uses negative frequency components from same-class images as an augmentation during PGD adversarial training or TRADES. It improved robust accuracy in their experiments, but it changes the training data and is outside the scope of our controlled frequency-band comparison.

## Works Cited

Chen, Yiting, Qibing Ren, and Junchi Yan. "Rethinking and Improving Robustness of Convolutional Neural Networks: a Shapley Value-based Approach in Frequency Domain." *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*, 2022.
