# Notes: "Explaining and Harnessing Adversarial Examples" (Goodfellow, Shlens & Szegedy, 2015)

## Main Idea

Goodfellow et al. explain adversarial examples with a simple idea: in a high-dimensional image, many small, coordinated changes can add up to a large change in a model's output. They introduce **Fast Gradient Sign Method (FGSM)**, a fast one-step white-box attack, and show that training on these adversarial examples can improve robustness.

Their explanation is a hypothesis about why attacks work: neural networks are often locally linear enough for a gradient-based perturbation to be effective. The important practical result for us is attack and training procedure, not whether linearity is full explanation.

## Why Adversarial Attacks Work

### Earlier Assumption Was Nonlinearity

People first suspected that adversarial examples came from neural networks being extremely complex and **nonlinear**. In this story, a model's decision boundary has many unpredictable bends or small "weird spots," and a small change can accidentally cross one of them.

Nonlinear here means that model's response can change in a complicated, location-dependent way: doubling an input change does not have to double its effect on output. This intuition sounds reasonable, but by itself it does not explain why simple models can be attacked so reliably.

### Goodfellow et al.'s Argument: Small Changes Accumulate Linearly

Goodfellow et al. argued that more useful explanation is almost opposite. A model can be vulnerable because, near one input, its loss behaves **approximately like a weighted sum** of pixel changes. An attacker makes a small change to many pixels at once, choosing each direction to increase loss. Each individual change is small, but thousands of them add together into a large change in model's score.

Their evidence is that even a simple linear softmax/logistic-regression model is vulnerable, and FGSM can attack several different models efficiently. For a linear model, their gradient-sign perturbation is exact worst-case perturbation under $L_\infty$ budget. So "neural networks are vulnerable only because they are bizarrely nonlinear" story cannot be entire explanation.

This does not prove that nonlinearity never matters. It shows that nonlinearity is not required: simpler high-dimensional, approximately linear explanation already predicts attacks they observe.

**Memory line:** An adversarial attack is not necessarily a small step into a strange nonlinear trap; it can be many small, coordinated pushes that add up in same harmful direction.

> **Note:** This use of *linear* and *nonlinear* is not simply about whether a function looks like $y = mx + b$ or $2^x$. Here, it means whether model behaves approximately like a weighted sum for small changes around current image, even if full neural network is nonlinear overall.

## Key Terms

- **Adversarial example:** an input changed slightly to make model predict wrong class.
- **White-box attack:** an attack that uses model's gradient.
- **Gradient:** direction that increases loss most quickly near current input.
- **$L_\infty$ budget ($\epsilon$):** maximum amount any one pixel value may change.
- **FGSM:** a one-step attack that changes every pixel in sign of loss gradient.
- **Adversarial training (AT):** train on adversarial examples as well as clean examples.

## FGSM and Its Connection to PGD

FGSM creates an adversarial example with

$$
x_{\mathrm{adv}} = x + \epsilon\,\mathrm{sign}\!\left(\nabla_x J(\theta, x, y)\right),
$$

where $J$ is loss, $x$ is clean image, $y$ is its label, and $\theta$ is model. The sign operation makes every pixel move by largest allowed amount in loss-increasing direction.

PGD uses same basic idea, but takes many smaller gradient steps and projects result back into allowed perturbation set after each step. So FGSM is simple one-step foundation for multi-step PGD attacks and PGD adversarial training used in our project.

## Why This Matters for Our Project

This paper gives basic reason we can create an attack during training: use loss gradient to find a harmful perturbation, then train model to handle it. Our experiment keeps that idea but restricts which perturbations are allowed by applying low-, middle-, or high-frequency masks.

The frequency mask changes set of allowed perturbation patterns; it does not change goal of increasing loss. That makes Goodfellow et al. foundation for our attack/training setup, while frequency-band comparison is our extension.

This paper does **not** study frequency bands, DCT/DFT masks, robust overfitting, peak robust-accuracy epoch, or post-peak decline. We need later papers, especially Rice et al., for those questions.

## Important Result

On MNIST maxout networks, FGSM adversarial training reduced error on FGSM adversarial examples from 89.4% to 17.9%, while also slightly improving clean test error (0.94% to 0.84%). This shows why adversarial training became a useful baseline.

However, paper evaluates one-step FGSM adversaries. Our robust-accuracy curves should be evaluated with fixed multi-step PGD attack in our research setup, not FGSM alone.

## Works Cited

Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and harnessing adversarial examples.* International Conference on Learning Representations.
