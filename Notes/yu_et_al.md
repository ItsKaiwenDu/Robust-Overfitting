# Notes: "Understanding Robust Overfitting of Adversarial Training and Beyond" (Yu, Han, Shen, Yu, Gong, Gong & Liu, 2022)

## 1. Summary

This paper investigates what actually causes robust overfitting, building directly on Rice et al.'s finding that robust overfitting is a widespread but poorly understood problem in adversarial training. The authors compare the training data distribution between adversarial training that overfits (strong adversary) and adversarial training that does not overfit (weak adversary). Through a series of data ablation experiments, they identify that small-loss data, meaning training examples the model already finds easy, is the actual cause of robust overfitting when the adversary is strong. Based on this finding, they propose minimum loss constrained adversarial training (MLCAT), a method that does not remove this small-loss data but instead artificially increases its loss during training, so the model keeps learning from it without collapsing into overfitting. Experiments show MLCAT eliminates robust overfitting and further improves robustness across multiple datasets, architectures, and threat models.

**Note on how this relates to Rice et al.:** Rice et al. showed robust overfitting exists and that early stopping fixes it, but did not explain why it happens. Meanwhile this paper, Yu et al. fills that gap, identifying small-loss data under a strong adversary as the specific cause.

## 2. Key Terms

- **Small-Loss Data vs. Large-Loss Data**: Small-loss data are training examples the model already classifies well, even after being attacked, meaning they are "easy" for the model. Large-loss data are examples the model still struggles with, meaning they are "hard" and push the model to keep learning.
- **Weak Adversary vs. Strong Adversary**: A weak adversary uses a small perturbation size (small epsilon), producing mild attacks. A strong adversary uses a large perturbation size, producing more aggressive attacks. Robust overfitting only shows up clearly under a strong adversary.
- **Data Ablation**: An experimental technique where the authors deliberately remove certain training examples (for example, only small-loss ones) to test whether that removal changes an observed outcome, here used to isolate the actual cause of robust overfitting.
- **MLCAT (Minimum Loss Constrained Adversarial Training)**: The paper's proposed training method, which keeps all training data but artificially boosts the loss of small-loss examples so the model cannot fully "solve" them and stop learning from them.
- **AWP (Adversarial Weight Perturbation)**: An earlier defense method that perturbs the model's own weights, not just the input, during training. MLCAT's weight perturbation variant borrows this general idea but uses it for a different purpose.

## 3. The Problem: What Causes Robust Overfitting?

Rice et al. established that robust overfitting is common in adversarial training, but did not explain why. Yu et al. answers that question directly.

**Starting observation.** The authors compare two training setups:

- **Weak adversary** (small perturbation size): robust overfitting does not really happen.
- **Strong adversary** (large perturbation size): robust overfitting clearly happens.

They then look at the training data itself, sorted by how much loss each example produces at a given point in training:

- Under a **weak adversary**, the training data is mostly small-loss data. Almost everything is "easy" for the model.
- Under a **strong adversary**, the training data is a mix: a meaningful chunk of small-loss data and a meaningful chunk of large-loss data.

This mismatch is the clue the rest of the paper investigates: something about that mix of small-loss and large-loss data under a strong adversary seems to be connected to robust overfitting.

## 4. Finding the Cause via Data Ablation

Based on the observation above, the authors ask two competing questions, then test each one directly by removing data and watching what happens to robust overfitting.

- **Q1: Is it the large-loss data?** Since strong-adversary training has a chunk of large-loss data that weak-adversary training doesn't, maybe removing the large-loss data would fix things.
- **Q2: Is it the small-loss data?** Alternatively, maybe the small-loss data present under a strong adversary is somehow "not worthy" of that adversary's strength, and is the actual problem.

**The experiment.** Using a fixed, strong perturbation size, the authors train while deliberately removing data from specific loss ranges, then check whether robust overfitting still happens.

**The result:**

- Removing **large-loss data** → robust overfitting still happens. Q1 is answered no.
- Removing **small-loss data** → robust overfitting disappears. Q2 is answered yes.

**Conclusion:** Small-loss data, specifically under a strong adversary, is the actual cause of robust overfitting. It is worth being precise about the type of claim this is: this conclusion comes from a controlled ablation experiment, not a mathematical proof. The authors' explanation for why this happens is that as adversarial training progresses, the model becomes more robust, so some of the adversarial examples generated during training stop being genuinely challenging. Once an adversarial example's loss drops low enough, continuing to train on it appears to actively hurt robust test performance rather than help it.

One additional detail: they note that this small-loss data comes from two different sources; some of it was small-loss from the very start of training, and some of it only became small-loss partway through training as the model improved. They find the overfitting effect is mainly driven by this second group, the data that transitioned into being small-loss partway through training, though the paper treats both groups together going forward for simplicity.

## 5. The Method: MLCAT

**The reasoning behind it.** Simply deleting small-loss data outright is not a good option: prior work (Schmidt et al.) shows that adversarial training generally benefits from having more data, so throwing data away has its own cost. The authors instead want to keep using every training example, but stop the model from treating small-loss data as "already solved."

**The core idea, in plain steps:**

1. During training, split each mini-batch of data into two groups based on their current loss: large-loss data and small-loss data, using a threshold value.
2. For large-loss data, train as normal. This data is still useful and challenging, so nothing changes.
3. For small-loss data, instead of training on it normally, apply an adjustment that artificially increases its loss before using it to update the model.
4. This prevents the model from ever getting to fully "coast" on easy examples, which is what the authors argue causes robust overfitting.

The authors describe this philosophy as turning waste into treasure: rather than discarding data that has become too easy, they force the model to keep extracting value from it.

**Two concrete ways they implement this idea:**

- **MLCAT with Loss Scaling (MLCAT_LS)**: Directly scales up the loss value of small-loss examples using a simple multiplier, so the model treats them as if they were harder than they currently are. This is roughly equivalent to training on that data with a larger effective learning rate.
- **MLCAT with Weight Perturbation (MLCAT_WP)**: Instead of touching the loss value directly, this version perturbs the model's own weights specifically to increase the loss on small-loss data, borrowing the general weight perturbation idea used in AWP, but applying it in a targeted way only to the data that needs it.

Both methods are described as orthogonal implementations of the same underlying MLCAT prototype: one adjusts the loss directly, the other adjusts the model's parameters to achieve a similar effect.

## 6. Results with MLCAT

MLCAT was tested on CIFAR-10, CIFAR-100, and SVHN, using both PreAct ResNet-18 and Wide ResNet-34-10, under both L∞ and L2 threat models (two different ways of measuring how large an adversarial perturbation is allowed to be).

**On the robustness gap (the difference between best and final robust accuracy, which is the direct measure of robust overfitting):**

- Standard adversarial training (AT) consistently showed a large gap, often around 5 to 8 percentage points, confirming the robust overfitting problem.
- Both MLCAT_LS and MLCAT_WP consistently shrank this gap dramatically, often down to less than 1 percentage point, across nearly every dataset, architecture, and threat model tested.

**On overall robustness performance, there is an important nuance:**

- Under **PGD-20** attacks, both MLCAT_LS and MLCAT_WP generally outperformed standard AT.
- Under **AutoAttack (AA)**, a stronger and more standardized attack benchmark, MLCAT_WP still generally outperformed standard AT, but **MLCAT_LS often performed worse than standard AT**. The authors attribute this specifically to the loss scaling technique making the model more sensitive to a known attack strategy called the logit-scaling attack, which AutoAttack can exploit.
- Because of this, MLCAT_WP is presented as the more reliable of the two realizations overall.

**Other supporting results:**

- MLCAT_WP achieved comparable natural (clean) accuracy to standard AT, meaning the robustness gains did not come at a large cost to clean performance.
- The MLCAT approach also generalized well when applied to TRADES, another popular adversarial training method, again narrowing its robustness gap.
- In a direct comparison, MLCAT_WP consistently outperformed AWP, the earlier weight-perturbation-based defense that inspired part of its design, across both PGD-20 and AutoAttack evaluations.
- An ablation study on the minimum loss threshold (a key hyperparameter controlling which data counts as "small-loss") showed that setting this threshold too high eventually hurts robustness and can even cause training to collapse, meaning this value needs to be tuned carefully rather than set as large as possible.

## 7. Works Cited

Yu, Chaojian, Bo Han, Li Shen, Jun Yu, Chen Gong, Mingming Gong, and Tongliang Liu. "Understanding Robust Overfitting of Adversarial Training and Beyond." *Proceedings of the 39th International Conference on Machine Learning*, PMLR 162, 2022.