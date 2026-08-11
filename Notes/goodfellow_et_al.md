# Notes: "Explaining and Harnessing Adversarial Examples" (Goodfellow, Shlens & Szegedy, 2015)

## 1. Summary

This paper focuses on a weird problem in machine learning: neural networks that get near-perfect accuracy on clean data can be tricked into confidently misclassifying an image after it's been changed by a perturbation so small that a human cannot even see it. Before this paper, people mostly blamed this on neural networks being too nonlinear and too complex to generalize properly. Goodfellow, Shlens, and Szegedy argue the opposite: the real cause is that modern models are actually too *linear*, not too nonlinear, and this linear behavior in high-dimensional space is what makes adversarial examples possible. Based on this insight, they introduce an efficient way to generate adversarial examples called the Fast Gradient Sign Method (FGSM), and they show that training on these adversarial examples (adversarial training) acts as a regularizer that makes models more robust, reducing error rates on adversarial inputs and even slightly improving performance on clean test data.

## 2. Key Terms

- **Adversarial Example**: An input (like an image) that has been deliberately and slightly modified so a model misclassifies it, even though the change is often invisible to a human.
- **Fast Gradient Sign Method (FGSM)**: An efficient method for generating adversarial examples by taking the sign of the gradient of the loss function with respect to the input, then scaling it by a small epsilon.
- **Adversarial Training**: A training strategy where the model is trained on a mix of normal examples and adversarial examples generated on the fly, so it learns to resist these attacks while training.
- **Linearity**: A system where the output changes predictably and proportionally to its input.
- **Nonlinearity**: A system where the output does not change in a simple, proportional manner and can respond abruptly depending on the input.

## 3. The Problem: Adversarial Attacks

Even state-of-the-art neural networks can be tricked by inputs that are only slightly different from correctly classified examples. What makes this especially strange is two things the paper highlights:

- **Cross-model transferability**: An adversarial example crafted to trick one model often tricks *other* models too, even if those models have different architectures or were trained on totally different subsets of data.
- **Imperceptibility**: The perturbation added to the input can be so small that it's invisible to a human, yet it's enough to flip the model's prediction with very high confidence.

The paper's famous example is the panda image: adding a tiny, carefully calculated noise pattern (scaled by only 0.007) causes GoogLeNet to reclassify a panda as a "gibbon" with 99.3% confidence, even though the image still looks exactly like a panda to a human.

## 4. How Adversarial Examples Are Generated (FGSM)

The Fast Gradient Sign Method works by exploiting how a model's loss function reacts to changes in the input.

**Idea**: Instead of doing something computationally expensive like an optimization search to find the "best" way to trick a model, FGSM does one efficient step. It looks at the gradient of the cost function J with respect to the input x, and takes the *sign* of that gradient (either +1 or -1 for each pixel/feature).

**Formula**:

```
η = ε · sign(∇x J(θ, x, y))
```

Where:
- `θ` = model parameters
- `x` = the original input
- `y` = the true label
- `J(θ, x, y)` = the cost function used to train the model
- `∇x J(...)` = gradient of the cost with respect to the input (tells you which direction increases the loss)
- `ε` = a small scalar controlling perturbation strength

You then create the adversarial example as:

```
x̃ = x + η
```

**Why "sign"?** Because the perturbation is constrained by a max-norm limit (each pixel can only change by at most ε), the *most damaging* direction to move each pixel is either fully positive or fully negative, not some in-between fractional value. Taking the sign of the gradient tells you, for every single pixel, "should I nudge this up or down to hurt the model's prediction the most," and then you nudge it by exactly ε.

This is fast because it only requires **one backpropagation pass** to get the gradient, unlike older attack methods that needed expensive iterative optimization.

## 5. Why Does This Work? (It's Linearity, Not Nonlinearity)

**Old assumption**: Before this paper, people figured neural networks were vulnerable to adversarial examples *because* they are complicated, nonlinear functions. The thinking was: nonlinear models are hard to interpret, so maybe they have weird "blind spots" scattered randomly through their decision space, kind of like potholes you cannot predict.

**Paper disproves old assumption**: Neural networks (and even simple linear models like logistic regression) are, in practice, designed to behave in a fairly *linear* way. Components like ReLUs, LSTMs, and maxout units are intentionally built to be close to linear because linear functions are much easier to optimize with gradient descent. The problem is that **linear behavior in high-dimensional space is dangerous**, even though linear behavior in low-dimensional space feels totally safe.

**Why high dimensions matter (the key intuition)**: Picture a simple linear model computing a weighted sum: `w · x`. If you nudge every single input feature `x_i` by just a tiny amount `ε` (small enough to be imperceptible, like one pixel value out of 255), that's a tiny change *per feature*. But if you have hundreds or thousands of features (like pixels in an image), and you nudge *all of them* in the direction that increases the model's error, those tiny nudges add up. The change in the model's output can grow **linearly with the number of dimensions** (n), even though each individual nudge stays imperceptibly small. The paper calls this a kind of "accidental steganography," where the model is secretly very sensitive to a coordinated pattern across many small changes, even though it looks robust to any single change.

**Explanation on transferability**: If different models trained on the same task all learn roughly similar linear decision boundaries (because they're all trying to solve the same underlying problem), then a perturbation direction that tricks one model's linear boundary is likely to also trick another model's very similar linear boundary. This is why the same adversarial image can trick multiple different networks.

**Supporting evidence for "linearity is the cause," not nonlinearity**:
- FGSM, a method based purely on linear reasoning, reliably tricks models across datasets (MNIST, CIFAR-10, ImageNet).
- Even a plain linear logistic regression model (which has no nonlinearity to blame) is vulnerable to the exact same kind of attack.
- RBF networks, which behave in a *strongly nonlinear* way (unlike ReLU/maxout networks), are actually much more resistant to adversarial examples. This directly supports the idea that linearity, not nonlinearity, is the root cause.

## 6. How to Defend?

The paper's proposed defense is **adversarial training**. It **trains the model on adversarial examples, not just clean ones**, so the model learns to be robust to the kind of perturbation that would normally trick it.

**The modified loss function** used during training combines the normal loss with the loss on an adversarially perturbed version of the same input:

```
J̃(θ, x, y) = α · J(θ, x, y) + (1 - α) · J(θ, x + ε·sign(∇x J(θ, x, y)), y)
```

Authors used `α = 0.5`, meaning the model is trained equally on the clean loss and the adversarial loss. Because the adversarial perturbation is recalculated at every training step based on the model's *current* parameters, the model is essentially chasing a constantly updating "worst case" version of each input, which forces it to generalize better rather than just memorize.

**Results reported in the paper**:
- Without adversarial training, a maxout network misclassified 89.4% of adversarial examples.
- With adversarial training, that error rate on adversarial examples dropped to 17.9%.
- Adversarial training also had a small regularization benefit on *clean* test data, reducing error from 0.94% to 0.84% on MNIST, and to as low as 0.782% with a larger model and early stopping tuned specifically on the adversarial validation error.
- The learned weights of the adversarially trained model also became visibly more localized and interpretable, rather than looking like noisy, diffuse patterns.

**Note**: Adversarial training is not a perfect fix. Even after that, the model still makes confident mistakes on adversarial examples that do trick it (average confidence of 81.4% on the ones that still fail). It reduces vulnerability, it doesn't eliminate it.

## 7. Works Cited

Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and Harnessing Adversarial Examples." *International Conference on Learning Representations (ICLR)*, 2015. *arXiv*, arXiv:1412.6572v3, 20 Mar. 2015.