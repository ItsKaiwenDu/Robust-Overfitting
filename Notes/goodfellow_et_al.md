# Notes: "Explaining and Harnessing Adversarial Examples" (Goodfellow, Shlens & Szegedy, 2015)

## Main Idea

Goodfellow et al. explain adversarial examples with a simple idea: in a high-dimensional image, many tiny, coordinated changes can add up to a large change in a model's output. They introduce the **Fast Gradient Sign Method (FGSM)**, a fast one-step white-box attack, and show that training on these adversarial examples can improve robustness.

Their explanation is a hypothesis about why attacks work: neural networks are often locally linear enough for a gradient-based perturbation to be effective. The important practical result for us is the attack and training procedure, not whether linearity is the full explanation.

## Why Adversarial Attacks Work

### Earlier Assumption Was Nonlinearity

People first suspected that adversarial examples came from neural networks being extremely complex and **nonlinear**. In this story, a model's decision boundary has many unpredictable bends or small "weird spots," and a tiny change can accidentally cross one of them.

Nonlinear here means that the model's response can change in a complicated, location-dependent way: doubling an input change does not have to double its effect on the output. This intuition sounds reasonable, but by itself it does not explain why simple models can be attacked so reliably.

### Goodfellow et al.'s Argument: Small Changes Accumulate Linearly

Goodfellow et al. argued that the more useful explanation is almost the opposite. A model can be vulnerable because, near one input, its loss behaves **approximately like a weighted sum** of the pixel changes. An attacker makes a tiny change to many pixels at once, choosing each direction to increase the loss. Each individual change is small, but thousands of them add together into a large change in the model's score.

Their evidence is that even a simple linear softmax/logistic-regression model is vulnerable, and FGSM can attack several different models efficiently. For a linear model, their gradient-sign perturbation is the exact worst-case perturbation under the $L_\infty$ budget. So the "neural networks are vulnerable only because they are bizarrely nonlinear" story cannot be the entire explanation.

This does not prove that nonlinearity never matters. It shows that nonlinearity is not required: the simpler high-dimensional, approximately linear explanation already predicts the attacks they observe.

**Memory line:** An adversarial attack is not necessarily a tiny step into a strange nonlinear trap; it can be many tiny, coordinated pushes that add up in the same harmful direction.

> **Note:** This use of *linear* and *nonlinear* is not simply about whether a function looks like $y = mx + b$ or $2^x$. Here, it means whether the model behaves approximately like a weighted sum for small changes around the current image, even if the full neural network is nonlinear overall.

## Key Terms

- **Adversarial example:** an input changed slightly to make the model predict the wrong class.
- **White-box attack:** an attack that uses the model's gradient.
- **Gradient:** the direction that increases the loss most quickly near the current input.
- **$L_\infty$ budget ($\epsilon$):** the maximum amount any one pixel value may change.
- **FGSM:** a one-step attack that changes every pixel in the sign of the loss gradient.
- **Adversarial training (AT):** train on adversarial examples as well as clean examples.

## FGSM and Its Connection to PGD

FGSM creates an adversarial example with

$$
x_{\mathrm{adv}} = x + \epsilon\,\operatorname{sign}\!\left(\nabla_x J(\theta, x, y)\right),
$$

where $J$ is the loss, $x$ is the clean image, $y$ is its label, and $\theta$ is the model. The sign operation makes every pixel move by the largest allowed amount in the loss-increasing direction.

PGD uses the same basic idea, but takes many smaller gradient steps and projects the result back into the allowed perturbation set after each step. So FGSM is the simple one-step foundation for the multi-step PGD attacks and PGD adversarial training used in our project.

## Why This Matters for Our Project

This paper gives the basic reason we can create an attack during training: use the loss gradient to find a harmful perturbation, then train the model to handle it. Our experiment keeps that idea but restricts which perturbations are allowed by applying low-, middle-, or high-frequency masks.

The frequency mask changes the set of allowed perturbation patterns; it does not change the goal of increasing the loss. That makes Goodfellow et al. the foundation for our attack/training setup, while the frequency-band comparison is our extension.

This paper does **not** study frequency bands, DCT/DFT masks, robust overfitting, peak robust-accuracy epoch, or post-peak decline. We need later papers, especially Rice et al., for those questions.

## Important Result

On MNIST maxout networks, FGSM adversarial training reduced error on FGSM adversarial examples from 89.4% to 17.9%, while also slightly improving clean test error (0.94% to 0.84%). This shows why adversarial training became a useful baseline.

However, the paper evaluates one-step FGSM adversaries. Our robust-accuracy curves should be evaluated with the fixed multi-step PGD attack in our research setup, not FGSM alone.

## Works Cited

Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and harnessing adversarial examples.* International Conference on Learning Representations.
