# Notes: "Overfitting in Adversarially Robust Deep Learning" (Rice, Wong & Kolter, 2020)

## 1. Summary

This paper studies a phenomenon in adversarial training, which is the process of training a model on adversarially perturbed inputs so that it becomes resistant to attacks. It is known that standard deep learning models rarely overfit in a harmful way, even when trained for a long time on overparameterized networks. Rice, Wong, and Kolter show that this is not true for adversarially robust training. They find that after the first learning rate decay, continuing to train actually makes the model's robust test error get worse, a phenomenon they call "robust overfitting." Based on this finding, they show that simply using early stopping, meaning stopping training at the checkpoint with the best validation performance instead of training until convergence, can match or beat the performance of several newer, more complicated adversarial training algorithms. They also test common overfitting remedies like regularization and data augmentation and find that, on their own, none of them outperform early stopping.

## 2. Key Terms

- **Robust Overfitting**: The observed pattern where robust test error gets worse the longer adversarial training continues, even as training error keeps improving. It is an empirical finding, not a phenomenon defined by a formal equation.
- **Robust Test Error vs. Standard Test Error**: Standard test error is accuracy on clean data; robust test error is accuracy after each test example is attacked with PGD. A model can score well on one and poorly on the other.
- **PGD (Projected Gradient Descent) Adversarial Training**: A training method that generates adversarial examples by taking repeated small gradient steps to worsen the loss, then constraining the perturbation back within an allowed size.
- **Early Stopping**: Halting training at an earlier checkpoint, chosen using a hold-out validation set, instead of training to full convergence.
- **Double Descent**: A pattern where test performance improves, dips, and then improves again as model complexity grows. The paper finds it does not explain robust overfitting.

## 3. Problem: Robust Overfitting

In standard deep learning, it is common practice to train large models for as long as possible. Convergence curves in that setting typically show that test loss keeps improving alongside training loss, so overfitting is not usually a major concern.

The authors find that adversarial training behaves differently. Specifically, shortly after the first learning rate decay, the robust test error briefly drops to its best value, but then it starts climbing back up as training continues, even while the robust training error keeps getting better. This gap between the best robust test error seen during training and the final robust test error at the end of training is what the authors call "robust overfitting."

A key point about this term: the authors are not proposing a formal mathematical definition or a specific test/threshold for robust overfitting. They observed this pattern by running the experiments and plotting the resulting error curves across multiple datasets and threat models, and they use the term descriptively to refer to that observed pattern. It is a phenomenon documented through empirical observation, not something derived analytically beforehand.

They demonstrate that this is not a one-off result. It appears across four datasets, SVHN (Street View House Numbers, a dataset of digit images cropped from real street-level photos), CIFAR-10, CIFAR-100, and ImageNet, and across two types of perturbation constraints, ℓ∞ and ℓ2 (two different mathematical ways of measuring how large an adversarial perturbation is allowed to be).

## 4. Experimental Setup

Adversarial training solves a min-max optimization problem: for each training example, the method first finds the worst-case perturbation within an allowed size (the "max" step, solved approximately using PGD), and then updates the model's weights to minimize loss against that worst-case version (the "min" step).

A PGD attack works by starting with a small random perturbation, then repeatedly nudging it in the direction that increases the model's loss, and constraining it back within the allowed perturbation size after each step. The number of steps used in this attack process matters for how strong the evaluation is, more steps generally produce a stronger attack.

One detail worth being precise about: the paper explicitly states the number of PGD steps used for evaluation only in the direct comparison against TRADES (Theoretically Principled Trade-off between Robustness and Accuracy, a specific adversarial training algorithm from prior work), where they specify a 20-step PGD adversary to match the evaluation setup used in the original TRADES paper. For their broader experiments across datasets in Table 1, the paper does not clearly specify whether a 10-step or 20-step PGD adversary was used for evaluation, so that detail should not be assumed one way or the other outside of the TRADES comparison.

## 5. Early Stopping Beats Complex Algorithms

This is the paper's most important and most interesting finding. Since robust test error does not improve the longer training continues, the authors argue that a lot of the recent algorithmic improvements to adversarial training may actually just be capturing the benefit of stopping earlier, rather than genuinely training a better model.

To test this, they focus on TRADES, a method that was reported to outperform standard PGD-based adversarial training. TRADES's originally reported result was 43.4% robust test error against an ℓ∞ PGD adversary (with perturbation radius 8/255) on CIFAR-10, a number that was viewed as a real algorithmic improvement over plain PGD training.

When Rice, Wong, and Kolter reproduce TRADES and let it train all the way to convergence, its robust test error actually degrades to 50.6%. TRADES's strong reported number depended on picking an earlier checkpoint, not on training to completion.

More strikingly, when they take plain, vanilla PGD-based adversarial training (the older, simpler method) and just apply early stopping using the best checkpoint on the test set, it reaches 42.3% robust test error, using the same model architecture and the same 20-step PGD evaluation used for the TRADES comparison. That is on par with, and even slightly better than, TRADES's own best reported result. In other words, the "improvement" credited to a newer algorithm could be reproduced almost entirely by simply not overtraining the older, simpler one.

This same pattern held for publicly released pretrained ImageNet models, where continuing to train past the best checkpoint made robust test error dramatically worse (for example, degrading from 62.7% to 85.5% under one perturbation setting), reinforcing that this is a broad property of adversarial training and not just an artifact of one dataset or codebase.

The authors also verify that this improvement is not just an artifact of peeking at the test set. Using a proper held-out validation set (separate from the test set) to decide when to stop, they still recover nearly the same performance (46.9% vs. 46.7% robust test error on CIFAR-10 with a pre-activation ResNet18), showing that early stopping's benefit is real and not just test set leakage.

## 6. Regularization & Data Augmentation Do Not Work

Given how large the effect of robust overfitting is, the authors test whether standard tools for fighting overfitting can fix the underlying problem, rather than just relying on early stopping.

They test the following approaches on CIFAR-10:

- **ℓ1 and ℓ2 regularization** (classical weight penalty methods, with ℓ2 regularization also commonly known as weight decay): both reduce robust overfitting to some degree, but neither matches the performance of simple early stopping, and pushing the regularization strength too high just makes the model underfit instead.
- **Cutout**: a data augmentation method that randomly masks out a square patch of the input image during training. Best result was 48.8% robust test error, still worse than early stopping's 46.9%.
- **Mixup**: a data augmentation method that trains on blended combinations of two images and their labels at once. Best result was 49.1% robust test error, also worse than early stopping.
- **Semi-supervised learning**: a technique that adds a large amount of unlabeled data, automatically labeled by an existing classifier, into training. This was the one exception: on its own it had very high variance in performance, but when combined with early stopping, it reached 40.2% robust test error, meaningfully better than early stopping alone.

The overall conclusion of this section is that no single classical or modern regularization technique, used by itself, prevents robust overfitting as effectively as simply stopping training early. The one method that improved meaningfully beyond early stopping required combining it with early stopping rather than replacing it.

The paper also checks whether double descent (the idea that test performance can improve again once model complexity grows large enough, even past the point of perfectly fitting the training data) explains away robust overfitting. They find that increasing model size (width) does still produce a double descent pattern in robust test performance, and larger models are still generally better, but robust overfitting still shows up at every model size tested. This means double descent and robust overfitting are separate phenomena. One does not explain or cancel out the other.

## 7. Works Cited

Rice, Leslie, Eric Wong, and J. Zico Kolter. "Overfitting in Adversarially Robust Deep Learning." *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 2020.