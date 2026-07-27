## Reading Notes: Overfitting in Adversarially Robust Deep Learning
### Rice, Wong & Kolter (ICML 2020)

> **Summary:** When training a standard deep learning model over many epochs, performance on new test data typically continues to improve even after the model has memorized the training set. This paper demonstrates that this pattern breaks down under adversarial training—the process of training a model to resist small, imperceptible perturbations designed to fool it. The authors show that extended adversarial training causes severe test performance degradation: robust test error increases substantially even as robust training error approaches zero. The authors term this phenomenon **Robust Overfitting** and show that it occurs consistently across diverse datasets, network architectures, and attack modalities. Crucially, simply applying early stopping at the optimal validation checkpoint resolves the vast majority of the problem, matching or exceeding the robust accuracy of far more complex defense methods. Furthermore, while common regularization and data augmentation techniques (e.g., $L_1$/$L_2$ decay, Cutout, Mixup) offer minor benefits, none match the efficacy of simple early stopping on their own.

---

### Definitions

* **Adversarial Training**: A defense methodology that hardens neural networks by generating adversarial inputs during training batches, compelling the model to learn representations resistant to norm-bounded perturbations.
* **CIFAR-10 / CIFAR-100**: Standard computer vision benchmarks consisting of 60,000 $32 \times 32$ pixel color images spanning 10 and 100 object categories, respectively.
* **Cutout**: A data augmentation technique that randomly zeroes out square patches of input images during training to simulate occlusion.
* **Double Descent**: A phenomenon in deep learning where test error decreases initially, increases as model capacity approaches the interpolation threshold, and then decreases again in over-parameterized regimes. It is distinct from robust overfitting.
* **Early Stopping**: An implicit regularization technique where training is terminated at the epoch exhibiting peak validation performance before over-parameterization degrades generalization.
* **Fast Gradient Sign Method (FGSM)**: A one-step adversarial attack that computes the gradient of the loss function with respect to the input and steps in the direction that maximizes loss (Goodfellow et al., 2015).
* **ImageNet**: A large-scale visual classification benchmark containing over 1.2 million high-resolution images across 1,000 object categories.
* **$\ell_\infty$ Perturbation**: An attack constraint bounding the maximum allowable change to any individual pixel. For CIFAR-10, a standard budget is $\epsilon = 8/255$, which is imperceptible to human eyes.
* **$\ell_2$ Perturbation**: An attack constraint bounding the overall Euclidean norm of pixel changes between original and perturbed inputs.
* **Learning Rate Decay**: A training schedule where the learning rate step size is reduced at predefined epochs (e.g., epochs 100 and 150) to facilitate fine-grained optimization convergence.
* **Mixup**: A data augmentation strategy that trains models on linear convex combinations of image pairs and their target labels.
* **Projected Gradient Descent (PGD)**: An iterative, multi-step attack method that takes small gradient steps to identify worst-case adversarial perturbations within an $\ell_p$-norm ball. PGD serves as the standard evaluation benchmark for adversarial robustness.
* **Regularization**: Techniques designed to penalize model complexity or prevent excessive memorization of training data, such as weight penalties ($L_1$/$L_2$) or structural dropout.
* **Robust Overfitting**: A failure mode in adversarial training where continued training reduces training robust loss while increasing test robust error.
* **Semi-Supervised Learning**: A training paradigm leveraging a small labeled dataset alongside a larger unlabeled dataset (pseudo-labeled by a model) to improve generalization.
* **SVHN (Street View House Numbers)**: A real-world image classification dataset containing over 70,000 cropped $32 \times 32$ digit images harvested from Google Street View house numbers.
* **TRADES**: An adversarial training objective that explicitly balances standard accuracy and robust accuracy by adding a regularizing KL-divergence loss term.

---

### Core Problem: Standard Overfitting Rules Fail in Adversarial Training

* In standard deep neural network training, training large models for many epochs rarely degrades generalization performance, a phenomenon linked to over-parameterization and double descent.
* Rice et al. establish that adversarial training violates this paradigm: after a specific point in training, further optimization continues to drive down robust training loss while actively increasing robust test error.
* The authors designate this persistent divergence between train and test robust error as **Robust Overfitting**.
* Robust overfitting occurs across all evaluated datasets (SVHN, CIFAR-10, CIFAR-100, ImageNet) and under both $\ell_\infty$ and $\ell_2$ perturbation metrics.

---

### Figure 1: Learning Curve Dynamics

* The paper's core empirical result is illustrated by four learning curves tracked across 200 training epochs on CIFAR-10:
  - **Train Robust Error (Orange)**: Decreases smoothly toward zero, showing that the model successfully learns to defend its training samples.
  - **Test Robust Error (Blue)**: Drops initially, but spikes sharply upward immediately following the first learning rate step drop at epoch 100 and continues rising.
  - **Clean Errors**: Train standard error (Red) approaches zero, while test standard error (Green) remains stable throughout training.
* **Quantified Impact:** At the first learning rate decay (epoch 100), the model achieves a peak test robust error of **43.2%**. By epoch 200, test robust error degrades to **51.4%**—an 8.2 percentage point loss in defense capability.
* **Key Finding:** Robustness is far more fragile than standard accuracy: standard test performance is unaffected by extended training, whereas robust test performance degrades severely.

---

### Early Stopping as an Efficacious Solution

* Because peak robust test performance occurs directly after the first learning rate decay step, saving model parameters at that checkpoint and halting training provides an effective fix.
* The authors show that standard PGD adversarial training (Madry et al., 2017) combined with early stopping reaches **43.2%** test robust error on CIFAR-10, matching or exceeding complex alternative methods like TRADES (**43.4%**).
* For $\ell_2$ adversarial training on CIFAR-10, early stopping reduces robust test error from **31.1% down to 28.4%**.
* **Implication for Robustness Research:** Many algorithmic modifications proposed as improvements over baseline PGD training may simply be recovering performance lost to robust overfitting, gains that early stopping provides at zero computational overhead.

---

### Background: Adversarial Defense Lineage

* **First Generation:** FGSM introduced single-step gradient attacks for adversarial training (Goodfellow et al., 2015).
* **Iterative Attacks:** FGSM was extended to multi-step optimization via the Basic Iterative Method (Kurakin et al., 2016) and modernized with random restarts as PGD adversarial training (Madry et al., 2017).
* **Algorithmic Variants:** Subsequent extensions built upon PGD include momentum-based adversaries (Dong et al., 2018), logit pairing (Mosbach et al., 2018), feature denoising (Xie et al., 2019), and TRADES (Zhang et al., 2019c), which optimizes a regularized trade-off between clean and robust loss.
* **Efficiency Focus:** Parallel research explored single-step or fast adversarial training to minimize computational costs (Shafahi et al., 2019; Wong et al., 2020).

---

### Background: Vulnerabilities & Broken Defenses

* Numerous non-adversarial defense mechanisms were subsequently shown to fail under stronger evaluations. Examples include defensive distillation (Papernot et al., 2016; broken by Carlini & Wagner, 2017) and detection heuristics (Metzen et al., 2017; Feinman et al., 2017).
* Athalye et al. (2018) demonstrated that many proposed defenses relied on *obfuscated gradients* (giving a false sense of security against weak attacks) and broke completely when evaluated with unbounded or optimization-based attacks.

---

### Formal Minimax Optimization Objective

* Adversarial training is formulated as a continuous minimax optimization problem over model parameters $\theta$:

$$\min_{\theta} \sum_{i} \max_{\delta \in \Delta} \ell(f_{\theta}(x_i + \delta), y_i)$$

* Where $f_\theta$ is the neural network, $(x_i, y_i)$ represents a training example, $\ell$ is the loss function, and $\Delta$ defines the set of permissible perturbations.
* The perturbation set $\Delta$ is defined as an $\ell_p$-norm ball: $\Delta = \{\delta : ||\delta||_p \leq \epsilon\}$ for perturbation budget $\epsilon > 0$.
* The inner maximization problem (finding worst-case perturbations) is solved approximately via PGD iterations:

$$\tilde{\delta} = \delta^{(t)} + \alpha \cdot \operatorname{sign}(\nabla_x \ell(f_{\theta}(x), y))$$
$$\delta^{(t+1)} = \max(\min(\tilde{\delta}, \epsilon), -\epsilon)$$

* This inner step runs for $K$ iterations with step size $\alpha$, projecting back onto the $\ell_\infty$ ball at each step.
* The outer minimization updates parameters $\theta$ using stochastic gradient descent.

---

### Table 1: Robust Overfitting Across Benchmark Datasets

* The authors evaluate adversarial training across four image datasets and two perturbation norms, recording peak test robust error (Best) and final test robust error at training completion (Final):

| Dataset   | Norm           | Radius  | Best  | Final | Difference |
|-----------|----------------|---------|-------|-------|------------|
| SVHN      | $\ell_\infty$  | 8/255   | 39.0% | 45.6% | +6.6%      |
| SVHN      | $\ell_2$       | 128/255 | 25.2% | 26.4% | +1.2%      |
| CIFAR-10  | $\ell_\infty$  | 8/255   | 43.2% | 51.4% | +8.2%      |
| CIFAR-10  | $\ell_2$       | 128/255 | 28.4% | 31.1% | +2.7%      |
| CIFAR-100 | $\ell_\infty$  | 8/255   | 71.9% | 78.6% | +6.7%      |
| CIFAR-100 | $\ell_2$       | 128/255 | 56.8% | 62.5% | +5.7%      |
| ImageNet  | $\ell_\infty$  | 4/255   | 62.7% | 85.5% | +22.8%     |
| ImageNet  | $\ell_2$       | 76/255  | 63.0% | 94.8% | +31.8%     |

* **Key Takeaway:** Performance degradation is severe on ImageNet, where full training degrades robust test error by **22.8%** ($\ell_\infty$) and **31.8%** ($\ell_2$). Because ImageNet is larger and more complex, models have more capacity to overfit robustly.
* Robust overfitting also impacts fast single-step methods (FGSM training) and regularized methods (TRADES).

---

### Learning Rate Schedules and Robust Overfitting (Figure 2)

* Because robust overfitting begins immediately following the first step drop in learning rate, the authors investigate whether continuous learning rate schedules prevent degradation.
* Five learning rate schedules were evaluated on CIFAR-10: piecewise decay (step drop), multi-step decay, linear decay, cyclic, and cosine annealing.
* **Findings:** None of the continuous learning rate schedules matched the peak robust accuracy achieved by piecewise step decay. Step decay creates a sharp transition that achieves the lowest robust test error.
* **Conclusion:** Continuous learning rate schedules smooth out the visual appearance of the learning curves, but do not prevent robust overfitting; the underlying generalization gap remains.

---

### Validation-Based Early Stopping Mechanics (Section 3.2)

* To prevent data leakage (peeking at test labels), early stopping must rely on a separate validation split.
* The authors reserve a hold-out validation set of 1,000 images from the CIFAR-10 training set to monitor validation robust loss.
* Validation robust loss closely tracks test robust loss. Halting training when validation loss plateaus reliably identifies the optimal checkpoint.
* **Validation Performance:** Applying validation-based early stopping to a PreActResNet-18 model on CIFAR-10 achieves **46.9%** robust test error, virtually identical to the theoretical best checkpoint (**46.7%**).
* **Re-evaluating Published Models:** Continuing to train publicly released models to convergence degrades robust test error on ImageNet from **62.7% to 85.5%** ($\ell_\infty$). Early stopping recovers this lost performance.

---

### Double Descent vs. Robust Overfitting (Section 3.3, Figure 5)

* Modern deep learning theory highlights *double descent*, where expanding model capacity or training duration eventually improves generalization beyond the interpolation threshold.
* Rice et al. test whether robust overfitting is simply an instance of double descent by systematically scaling network width (using WideResNet architectures) rather than training duration.
* **Empirical Disalignment:**
  - **Best Checkpoint Curve:** Continues to improve as model width increases, adhering to double descent principles (larger models achieve better peak robustness).
  - **Final Checkpoint Curve:** Remains significantly worse across all network widths, maintaining a persistent gap between best and final performance.
* **Conclusion:** Double descent and robust overfitting are distinct phenomena. Increasing model capacity raises peak achievable robustness, but does not eliminate robust overfitting during extended training.

---

### Ablation Study: Standard Regularization vs. Robust Overfitting (Section 4)

* The authors perform an empirical evaluation comparing standard regularization and data augmentation techniques against early stopping on CIFAR-10 (PreActResNet-18, PGD-10, $\ell_\infty$ budget $\epsilon = 8/255$).

#### Explicit Regularization ($L_1$ and $L_2$ Weight Decay)
* Adding weight decay penalties to the loss function ($\tilde{\ell}(\theta) = \ell(\theta) + \lambda \Omega(\theta)$) fails to mitigate robust overfitting.
* With an optimal hyperparameter setting ($\lambda = 5 \times 10^{-3}$), $L_2$ weight decay yields a final robust test error of **55.2%**, significantly worse than validation early stopping (**46.9%**).
* *Note on Paper Text Discrepancy:* Section 4.1 text lists optimal $L_2$ decay as $\lambda = 5 \times 10^{-2}$, whereas the Figure 6 caption lists $\lambda = 5 \times 10^{-3}$ for the same 55.2% result. In either case, the conclusion stands.
* **Mechanism Failure:** Weak weight decay allows robust overfitting to persist, while strong weight decay over-regularizes the model, degrading both clean and robust accuracy.

#### Data Augmentation (Cutout & Mixup)
* **Cutout:** Randomly masking $14 \times 14$ pixel patches during training yields a final robust test error of **48.8%** (Best: **46.7%**, Gap: **2.1%**).
* **Mixup:** Convex blending of training samples yields a final robust test error of **49.1%** (Best: **46.3%**, Gap: **2.8%**).
* While Cutout and Mixup slightly narrow the generalization gap compared to unregularized training, neither eliminates robust overfitting or matches simple early stopping.

#### Semi-Supervised Learning
* Incorporating 500,000 pseudo-labeled unlabeled images from TinyImages reduces the severity of robust overfitting, but results in high variance at convergence (average final error: **47.1%**).
* Combining semi-supervised learning with early stopping yields **40.2%** robust test error—the single best result reported in the paper and the only approach that significantly outperforms standard early stopping.

---

### Table 2: Complete Method Comparison (CIFAR-10, $\ell_\infty$, $\epsilon = 8/255$)

| Method                    | Final Test Error | Best Test Error | Difference |
|---------------------------|------------------|-----------------|------------|
| Early stopping w/ val     | 46.9%            | 46.7%           | +0.2%      |
| $L_1$ regularization      | 53.0%            | 48.6%           | +4.4%      |
| $L_2$ regularization      | 55.2%            | 46.4%           | +8.8%      |
| Cutout                    | 48.8%            | 46.7%           | +2.1%      |
| Mixup                     | 49.1%            | 46.3%           | +2.8%      |
| Semi-supervised           | 47.1%            | 40.2%           | +6.9%      |

* *Note on Table 2 Math:* The original paper lists the $L_2$ difference column as 55.2%, which is a typesetting error in the publication. The true difference ($55.2\% - 46.4\%$) is **8.8%**, as reflected above.

---

### Conclusions & Recommendations for the Field

* Robust overfitting is a dominant, universal property of adversarial training across architectures, attack types, and datasets.
* Standard regularization tools (weight decay, data augmentation) fail to prevent robust overfitting without causing over-regularization.
* Early stopping guided by a validation set is the most consistent, low-cost defense against robust overfitting.
* The authors urge the machine learning community to report full validation learning curves alongside benchmark metrics. Comparing algorithms solely at final-epoch convergence risks misattributing early-stopping artifacts to algorithmic superiority.

---

### Connection to Our Project

* **Core Research Subject:** Rice et al. defined and characterized the robust overfitting phenomenon that our project analyzes.
* **Experimental Control:** The paper's findings establish that early stopping must be integrated into baseline controls to ensure fair comparison between adversarial training variants.
* **Dataset Scaling Relevance:** Table 1 demonstrates that robust overfitting intensifies on complex datasets like ImageNet, underscoring the importance of tracking generalization gaps across diverse evaluation sets.
* **Separation of Dynamics:** The distinction between double descent and robust overfitting informs our theoretical modeling, ensuring we treat capacity scaling and epoch duration as separate factors.
* **Reporting Standards:** We adopt the paper's recommendation to log complete validation learning curves for all experiments, evaluating models at both peak validation checkpoints and training completion.

---

### Works Cited

* Rice, Leslie, Eric Wong, and J. Zico Kolter. "Overfitting in Adversarially Robust Deep Learning." *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119:8093-8104, 2020. arXiv: https://arxiv.org/abs/2002.11569.