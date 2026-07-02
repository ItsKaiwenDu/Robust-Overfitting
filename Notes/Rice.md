## Reading Notes: Overfitting in Adversarially Robust Deep Learning
### Rice, Wong & Kolter (ICML 2020)

> **Summary:** When we train a deep learning model for a long time, it usually keeps getting better at handling new data, even if it has already memorized training set. This paper shows that this is not true for adversarial training, which is process of training a model to resist small, unnoticeable changes to its inputs that try to fool it. authors show that if we keep training an adversarially trained model for too long, it starts to do worse on new test data, even though it keeps improving on training data. This is Robust Overfitting, and it happens across many different datasets and types of attacks. good news is that simply stopping training early, at right checkpoint, fixes most of problem and can match results of much more complicated training methods. They also test other common fixes for overfitting, like regularization and data augmentation, but none of these work as well as early stopping on their own.

---

### Definitions

* **Adversarial Training**: A way of making a neural network tougher by training it on adversarially perturbed inputs, so it learns to handle attacks during training rather than being surprised by them at test time.
* **CIFAR-10 / CIFAR-100**: Two standard image classification benchmarks. CIFAR-10 has 10 categories and CIFAR-100 has 100 categories, both with 32x32 pixel images.
* **Cutout**: A data augmentation method that randomly masks out a square patch of input image during training, simulating occlusion.
* **Double Descent**: A pattern sometimes seen in standard deep learning where test error first goes down, then briefly goes up as model capacity grows, and then comes back down again. It is related to but different from robust overfitting.
* **Early Stopping**: A training technique where you stop training at point where model performs best on held-out data, rather than letting it run to full convergence.
* **Fast Gradient Sign Method (FGSM)**: A simpler, one-step attack that computes gradient of loss and nudges each input value in direction that increases model's error (introduced by Goodfellow et al., 2014).
* **ImageNet**: A very large image dataset with over a million images across 1,000 categories. It is one of most demanding benchmarks in computer vision.
* **l-infinity (l∞) Perturbation**: A type of attack budget that limits how much any single pixel in image can change. For CIFAR-10, a common budget is ε = 8/255, which is invisible to human eye.
* **l2 Perturbation**: A type of attack budget that limits total size of all pixel changes combined, measured as Euclidean distance between original and perturbed image.
* **Learning Rate Decay**: A training schedule trick where learning rate is reduced at certain points during training (for example, at epoch 100 and 150) to help model converge more precisely.
* **Mixup**: A data augmentation method that trains on blended pairs of images and their labels, encouraging model to behave linearly between training examples.
* **Projected Gradient Descent (PGD)**: A multi-step attack method that repeatedly takes small gradient steps to find worst-case adversarial perturbation within a allowed budget. It is generally considered a strong baseline attack for evaluating robustness.
* **Regularization**: A group of techniques that try to prevent a model from memorizing its training data too precisely. Common examples include L1 and L2 weight penalties and Dropout.
* **Robust Overfitting**: A specific failure mode in adversarial training where model keeps getting better at defending its training data, but gets worse at defending new, unseen test data longer training continues.
* **Semi-supervised Learning**: A training approach that uses both a small set of labeled examples and a large set of unlabeled examples to improve model generalization.
* **SVHN**: Street View House Numbers, a real-world image dataset of digits photographed from house numbers in Google Street View, used as a benchmark for image classification.
* **TRADES**: A method for adversarial training that tries to balance standard accuracy and robust accuracy by adding a regularization term to training objective.

---

### Core Problem: Standard Overfitting Does Not Apply in Adversarial Training

* In standard deep learning, it is widely observed that training very large models for a very long time does not hurt generalization much. This is sometimes called "double descent" phenomenon.
* This paper shows that adversarial training breaks that pattern. After a certain point, further training keeps reducing training robust loss, but actually increases test robust error.
* This mismatch between train and test robust performance is what authors call Robust Overfitting.
* It appears consistently across SVHN, CIFAR-10, CIFAR-100, and ImageNet, and under both l∞ and l2 perturbation budgets.

---

### Figure 1: Learning Curves

* key chart in this paper shows four curves plotted across 200 training epochs on CIFAR-10.
* Train robust error (orange) falls smoothly toward 0, meaning model gets nearly perfect at defending its own training data.
* Test robust error (blue) drops initially, but after first learning rate decay at epoch 100 it spikes sharply upward and keeps rising.
* At first learning rate decay, model briefly achieves 43.2% test robust error. By end of full training, that number has risen to 51.4%, a significant drop in defense quality.
* Train standard error (red) also falls toward 0, and test standard error (green) stays relatively flat and well-behaved.
* key takeaway: overfitting hurts robust test performance badly, while standard test performance is largely unaffected. This shows that robustness is a much more fragile property than standard accuracy.

---

### Early Stopping as a Fix

* Since best robust test performance occurs right after first learning rate decay, simply saving model at that checkpoint and stopping training is an effective fix.
* authors show that vanilla PGD-based adversarial training (Madry et al., 2017) with early stopping can achieve 43.2% test robust error on CIFAR-10, which matches 43.4% reported by TRADES, a much more complex method.
* same pattern holds for l2 adversarial training. Stopping a CIFAR-10 model early can reduce robust test error from 31.1% down to 28.4%.
* implication is large: many recent algorithmic improvements over PGD-based training may not reflect genuine advances. They may just be recovering performance that was lost to robust overfitting, and early stopping achieves same gains for free.

---

### Background: Adversarial Training Lineage

* first major adversarial training method used FGSM, a single gradient step, to generate attacks during training (Goodfellow et al., 2014).
* FGSM was extended to multiple steps in Basic Iterative Method (Kurakin et al., 2016), and then combined with random restarts to form PGD adversarial training (Madry et al., 2017).
* Later improvements layered on top of PGD include: adding momentum to adversary (Dong et al., 2018), logit pairing (Mosbach et al., 2018), feature denoising (Xie et al., 2019), and matrix norm estimation (Yang et al., 2019).
* TRADES (Zhang et al., 2019) is one of most prominent of these, balancing trade-off between standard error and robust error in loss function directly.
* Separate work has also tried to speed up adversarial training by reducing number of attack steps (Shafahi et al., 2019; Wong et al., 2020).

---

### Background: Defenses That Were Broken

* Several proposed defenses beyond adversarial training were later shown to fail under stronger attacks. These include distillation (Papernot et al., 2016; Carlini & Wagner, 2017) and detection methods (Metzen et al., 2017; Feinman et al., 2017).
* Adversarial examples were shown to transfer across different viewpoints in real world (Lu et al., 2017), but this was later countered by Athalye et al. (2017).
* Many defenses were revealed to rely on obfuscated gradients, making them appear robust when evaluated with weak attacks but failing against stronger ones (Athalye et al., 2018). Examples include thermometer encoding (Buckman et al., 2018) and various preprocessing techniques.

---

### Formal Adversarial Training Objective

* Adversarial training is framed as a minimax optimization problem. goal is to find model parameters θ that minimize worst-case loss across all allowed perturbations:

$$\min_{\theta} \sum_{i} \max_{\delta \in \Delta} \ell(f_{\theta}(x_i + \delta), y_i)$$

* Here, $f_\theta$ is neural network, $(x_i, y_i)$ is a training example, $\ell$ is loss function, and $\Delta$ is set of allowed perturbations.
* perturbation set $\Delta$ is usually an $\ell_p$-norm ball: $\Delta = \{\delta : ||\delta||_p \leq \epsilon\}$ for some small budget $\epsilon > 0$.
* inner maximization (finding worst perturbation) is solved approximately using PGD, which iteratively updates perturbation with rule:

$$\tilde{\delta} = \delta^{(t)} + \alpha \cdot \text{sign}(\nabla_x \ell(f(x), y))$$
$$\delta^{(t+1)} = \max(\min(\tilde{\delta}, \epsilon), -\epsilon)$$

* This inner loop runs for multiple steps with step size $\alpha$, projecting back onto $\ell_\infty$ ball after each step.
* outer minimization is solved with standard gradient descent on model parameters.

---

### Table 1: Robust Overfitting Across Datasets

* authors run adversarial training across four datasets and two perturbation norms and record two numbers: best robust test error seen at any checkpoint during training, and final robust test error at end of training.
* difference between those two numbers is cost of robust overfitting. Key results:

| Dataset    | Norm | Radius  | Best   | Final  | Difference |
|------------|------|---------|--------|--------|------------|
| SVHN       | l∞   | 8/255   | 39.0%  | 45.6%  | 6.6%       |
| SVHN       | l2   | 128/255 | 25.2%  | 26.4%  | 1.2%       |
| CIFAR-10   | l∞   | 8/255   | 43.2%  | 51.4%  | 8.2%       |
| CIFAR-10   | l2   | 128/255 | 28.4%  | 31.1%  | 2.7%       |
| CIFAR-100  | l∞   | 8/255   | 71.9%  | 78.6%  | 6.7%       |
| CIFAR-100  | l2   | 128/255 | 56.8%  | 62.5%  | 5.7%       |
| ImageNet   | l∞   | 4/255   | 62.7%  | 85.5%  | 22.8%      |
| ImageNet   | l2   | 76/255  | 63.0%  | 94.8%  | 31.8%      |

* degradation is worst on ImageNet, where full training raises robust error by over 22 percentage points under l∞ attacks. This makes intuitive sense because ImageNet is largest and most complex dataset, giving model most room to overfit.
* Robust overfitting is also not specific to PGD-based training. It affects faster training methods like FGSM adversarial training and top-performing methods like TRADES.

---

### Learning Rate Schedules and Robust Overfitting (Figure 2)

* Since overfitting seems to start exactly at first learning rate decay, a natural question is whether smoother learning rate schedules could prevent it.
* authors test five schedules on CIFAR-10: piecewise decay (default), multiple decay steps, linear decay, cyclic, and cosine.
* Figure 2 shows that none of smoother alternatives can match peak performance of piecewise decay schedule. best robust test error is achieved specifically by having a single large, discrete drop in learning rate.
* Smoother schedules produce smoother curves, but their best checkpoints are still strictly worse than best checkpoint under piecewise decay.
* Conclusion: changing learning rate schedule does not eliminate robust overfitting, it just makes curves look less jagged. fundamental problem remains.

---

### Early Stopping: How It Works in Practice (Section 3.2)

* Early stopping is a classic implicit regularization technique. It works by monitoring performance on a held-out validation set and stopping training when that performance stops improving.
* authors find it is especially important for adversarially robust training because test robust error does not decrease monotonically during training. It actively gets worse after learning rate drops.
* Key result: when early stopping is applied to vanilla PGD-based adversarial training, model reaches 42.3% robust test error on CIFAR-10. This is actually slightly better than best result reported by TRADES (Zhang et al., 2019c).
* For ImageNet, continuing to train publicly released pretrained models degrades their robust test error from 62.7% to 85.5% under l∞ (ε = 4/255) and from 63.0% to 94.8% under l2 (ε = 128/255). Early stopping recovers nearly all of that lost performance.
* TRADES itself relies on early stopping to get its headline number. authors confirm that allowing TRADES to train to convergence raises its robust test error from 44.1% to 50.6% on CIFAR-10.

---

### Validation-Based Early Stopping (Figure 4)

* A concern with early stopping is that it might "look at" test set performance to decide when to stop, which would be cheating.
* authors address this by using a true hold-out validation set of 1,000 examples withheld from CIFAR-10 training set.
* Figure 4 shows that validation robust loss curve closely tracks test robust loss curve. Stopping when validation loss stops improving is a reliable proxy for true best test checkpoint.
* Using this honest, validation-based early stopping, model achieves 46.9% robust test error on a pre-activation ResNet18. best model checkpoint during training (which does look at test labels indirectly by reporting minimum) achieves 46.7%, nearly identical.
* This confirms that early stopping does not require peeking at test data. A small held-out validation set is enough to implement it correctly.

---

### Double Descent vs. Robust Overfitting (Section 3.3, Figure 5)

* In standard deep learning, a well-known phenomenon called double descent says that increasing model complexity (bigger models, more training) eventually helps test performance even past point of interpolating all training data.
* At first glance, robust overfitting looks like a contradiction of double descent, since training longer makes test robust performance worse.
* authors show these are actually two separate effects by varying model width (using Wide ResNets) rather than training time.
* Figure 5 shows two separate curves plotted against model width factor: final model checkpoint (blue) and best checkpoint during training (orange).
* best checkpoint curve keeps improving as width increases, following double descent pattern. Larger models do help robustness when you stop at right time.
* final model curve, however, is much higher (worse) and shows a different shape. This shows that robust overfitting from training too long is a separate problem from standard bias-variance tradeoff that double descent describes.
* Practical implication: you cannot explain away robust overfitting by pointing to double descent. They are different phenomena and need separate fixes.

---

### Section 4: Alternative Methods to Prevent Robust Overfitting

* authors run a full ablation study asking whether standard tools for preventing overfitting also work against robust overfitting.
* All experiments use CIFAR-10 with a PreActResNet18 model and a 10-step PGD adversary with l∞ radius 8/255, and each method is tuned to its best hyperparameter before comparing.
* baseline throughout this section is pure early stopping using a validation set, which achieves 46.9% final robust test error and 46.7% best robust test error (difference of only 0.2%).

---

### Explicit Regularization (Section 4.1, Figure 6)

* standard approach to preventing overfitting in machine learning is to add a penalty term directly to loss function that punishes model complexity. modified loss becomes:

$$\tilde{\ell}(\theta) = \ell(\theta) + \lambda \Omega(\theta)$$

* Here $\Omega(\theta)$ is regularization penalty (such as sum of squared weights for L2, or sum of absolute weights for L1), and $\lambda$ is a hyperparameter that controls how strongly penalty is applied.
* L2 regularization (also called weight decay) is most common form used in deep learning. L1 regularization encourages model to set many weights to exactly zero, producing sparser models.
* Results: even with optimal value of $\lambda = 5 \times 10^{-3}$, best L2 regularization can achieve is 55.2% robust test error at end of training, which is worse than early stopping's 46.9%.
    * **Note:** Found λ inconsistency where in Section 4.1 reports the best ℓ2 regularization result as 55.2% robust test error, but cites the hyperparameter as $\lambda = 5 \times 10^{-2}$ in the main text and $\lambda = 5 \times 10^{-3}$ in the Figure 6 caption. Same result, two different stated λ values which seems like a typo in one of the two spots.
* Key problem shown in Figure 6 is that there is a narrow window where L2 helps at all. Too little regularization and robust overfitting continues. Too much regularization and model becomes over-regularized, hurting both train and test robust performance without fixing gap between them.
* L1 regularization shows same pattern and never matches early stopping either.
* Conclusion: explicit regularization cannot remove detrimental effects of robust overfitting without also causing over-regularization. It does not substitute for early stopping.

---

### Data Augmentation: Cutout and Mixup (Section 4.2, Figure 7)

* Cutout randomly masks out a square patch of input image during training. Mixup trains on blended pairs of images and their corresponding blended labels.
* Both are popular modern data augmentation strategies that have been shown to reduce overfitting in standard deep learning.
* authors scan across full range of hyperparameters for both methods.
* Cutout results: even with optimal patch length of 14, best robust test error at end of training is 48.8%. gap between best and final checkpoint is 2.1 percentage points, meaning robust overfitting still occurs.
* Mixup results: achieves 49.1% robust test error at convergence with a best checkpoint of 46.3%, a 2.8 point gap.
* Neither method closes gap between best and final checkpoint to level that early stopping achieves (0.2 points).
* Conclusion: cutout and mixup reduce robust overfitting slightly compared to unregularized training, but they do not solve it, and neither reaches performance of simple early stopping.

---

### Semi-Supervised Learning (Section 4.2, Figure 8)

* Semi-supervised learning augments training set with a large collection of unlabeled data. A standard classifier is used to assign pseudo-labels to these unlabeled examples, and full augmented dataset is then used for robust training.
* authors use 500K pseudo-labeled examples from TinyImages, a large dataset of internet images related to CIFAR-10 categories.
* Figure 8 shows that semi-supervised training has an unusual learning curve: robust overfitting is much less severe, but robust test error has very high variance at convergence rather than clearly plateauing.
* Because of this high variance, average final model performance is 47.1%, which is actually similar to pure early stopping (46.9%). On average, semi-supervised learning alone is not clearly better.
* However, combining early stopping with semi-supervised learning eliminates both robust overfitting and high variance at once, achieving 40.2% robust test error.
* This combination is only approach in entire paper that significantly improves upon early stopping alone.

---

### Table 2: Summary of All Regularization Methods

* full comparison of all methods on CIFAR-10 (PreActResNet18, l∞, radius 8/255):

| Method                  | Final Test Error | Best Test Error | Difference |
|-------------------------|-----------------|----------------|------------|
| Early stopping w/ val   | 46.9%           | 46.7%          | 0.2%       |
| L1 regularization       | 53.0%           | 48.6%          | 4.4%       |
| L2 regularization       | 55.2%           | 46.4%          | 8.8%       |
| Cutout                  | 48.8%           | 46.7%          | 2.1%       |
| Mixup                   | 49.1%           | 46.3%          | 2.8%       |
| Semi-supervised         | 47.1%           | 40.2%          | 6.9%       |

* **Note:** Paper's own Table 2 prints Diff for L2 regularization as 55.2, but Final (55.2) minus Best (46.4) is actually **8.8**. The table above uses corrected value of 8.8%. Looks like a copy-paste error instead of actual computed gap.
* Every method except semi-supervised produces a final model that is worse than early stopping's final model of 46.9%.
* Every method except semi-supervised produces a best checkpoint that is roughly equal to or worse than early stopping's best checkpoint of 46.7%.
* Semi-supervised learning achieves best checkpoint of any method at 40.2%, but only when early stopping is also applied. Without early stopping, its variance makes it unreliable.
* difference column shows how much robust overfitting each method still suffers. L2 regularization has largest gap at 8.8 points, meaning it actually makes robust overfitting worse on this metric even though it reduces final error somewhat.

---

### Conclusion (Section 5)

* central finding is that robust overfitting is a dominant and general property of adversarial training across datasets, architectures, attack types, and training methods.
* Increasing model size helps best achievable robust test performance (consistent with double descent), but it does not reduce effect of robust overfitting. Larger models still suffer same gap between their best and final checkpoints.
* full suite of tested techniques, including explicit regularization and data augmentation, mostly either over-regularize model or fail to close gap to early stopping.
* simplest fix, early stopping with a validation set, remains most reliable and consistent remedy across all settings.
* only method that genuinely beats early stopping alone is combining it with semi-supervised learning, but this requires access to a large pool of unlabeled data.
* authors make a broader point aimed at research community: because robust overfitting is so prevalent, researchers should always report validation-based learning curves alongside their final numbers. A model with a better final number might still be worse at its best checkpoint. field risks mistaking a better training procedure for a better adversarial defense, when in reality difference comes from where training happened to stop.

---

### Connection to Our Research

* This paper is direct source of Robust Overfitting problem our project investigates. It names phenomenon, characterizes when it appears, and proposes early stopping as primary remedy.
* finding that early stopping alone can match TRADES and other complex methods is important for our project because it changes how we should read adversarial robustness leaderboard. Better numbers may not mean better algorithms.
* FGSM and PGD background in Section 2 connects directly to Goodfellow et al. (2014): FGSM was origin, and PGD is multi-step extension that Rice et al. use as their standard attack throughout experiments.
* Understanding Figure 1 is foundation for all later analysis in this paper. shape of those four learning curves, especially spike in test robust error at epoch 100, is central empirical fact rest of paper tries to explain and fix.
* Table 1 is critical for our project because it shows that robust overfitting is not a quirk of one dataset. 22.8 percentage point gap on ImageNet is especially striking and shows that problem scales with dataset complexity.
* distinction between double descent and robust overfitting in Section 3.3 matters for how we frame our research. We cannot dismiss robust overfitting as just a case of model not being big enough. Figure 5 shows that bigger models still overfit robustly; they just overfit from a better starting point.
* validation-based early stopping result in Figure 4 is practically important. It shows that a research team does not need access to test labels to implement best fix. A 1,000-example validation split is sufficient.
* Table 2 is key result for our research direction: it shows exactly which tools fail against robust overfitting and by how much. fact that L2 regularization, most common anti-overfitting tool in deep learning, makes best-vs-final gap worse rather than better is a striking result.
* conclusion's call for community to always report validation-based learning curves is directly relevant to how we evaluate our own experiments. We should apply same standard and never report only a final-epoch number without full learning curve context.

---

### Works Cited

* Rice, Leslie, et al. "Overfitting in Adversarially Robust Deep Learning." *Proceedings of the 37th International Conference on Machine Learning*, PMLR, vol. 119, 2020. arXiv, https://arxiv.org/abs/2002.11569.