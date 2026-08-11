# Notes: "Rethinking and Improving Robustness of Convolutional Neural Networks: a Shapley Value-based Approach in Frequency Domain" (Chen, Ren & Yan, 2022)

## 1. Summary

This paper introduces a new way to measure exactly how much each individual frequency component of an image contributes to a CNN's prediction. To do this, the authors borrow the Shapley value, a tool from game theory that fairly splits credit among cooperating players, and apply it to frequency components instead of players in a game. Using this tool, they find that standard-trained models barely use high-frequency information on clean images but are very vulnerable to attacks on that same high-frequency information. They also find that adversarially trained models become robust largely by learning to ignore high-frequency information altogether, which helps explain the well-known trade-off between clean accuracy and robustness. Based on these findings, they propose a new data augmentation defense called Class-wise Shapley value-guided Augmentation (CSA), which improves adversarial robustness with very little added cost.

## 2. Key Terms

- **Shapley Value**: A method from game theory that fairly measures how much each individual player contributes to a group's total result, by averaging that player's effect across every possible order players could join in.
- **Frequency Component / Discrete Fourier Transform (DFT)**: The DFT breaks an image down into a set of frequency components, each representing a different "wave pattern" in the image. Low frequencies capture broad structure; high frequencies capture fine detail and texture.
- **Positive/Negative Frequency Component (PFC/NFC)**: A PFC is a frequency component whose Shapley value is positive, meaning it pushes the model toward the correct class. An NFC has a negative Shapley value, meaning it pushes the model away from the correct class.
- **Standard Training (ST) vs. Adversarial Training (AT) model**: An ST model is trained only on clean, unperturbed data. An AT model is trained on adversarially perturbed data specifically to resist attacks.
- **Class-wise Shapley value-guided Augmentation (CSA)**: The paper's proposed defense, which augments training data using NFCs collected from other clean samples in the same class, to correct which frequency features the model relies on.

## 3. The Problem: Why a New Way to Measure Frequency Contribution?

There is a popular hypothesis in prior work: CNNs rely too much on high-frequency components (HFCs), and this is why they are vulnerable to adversarial attacks. The authors point out two limitations in how earlier work tested this hypothesis:

- **Too coarse a split.** Most prior work manually cuts the frequency spectrum into just two buckets, "low frequency" and "high frequency," rather than looking at each individual frequency component.
- **Only dataset-level analysis.** Prior work usually measures contribution averaged across the whole dataset, which hides how much this can vary from one image to the next.

The authors' goal is to fix both problems at once: measure the contribution of every individual frequency component, for every individual image, using a mathematically principled tool instead of an intuitive, manual split.

## 4. The Method: Shapley Value in the Frequency Domain

**What a Shapley value normally does.** In game theory, if a group of players cooperate to produce some total value, the Shapley value tells you how much credit each individual player deserves. It does this by imagining every possible order in which players could join the group, and averaging how much each player's presence changes the outcome.

**How the authors adapt it here.** Instead of players in a game, they treat each frequency component of an image as a "player." The "outcome" is the model's prediction score for the correct class. A frequency component's Shapley value is the average effect that component has on the model's correct-class output, across every possible combination of which other frequency components are present or masked out.

- A component with a **positive** Shapley value is called a **PFC** (positive frequency component): it helps the model predict correctly.
- A component with a **negative** Shapley value is called an **NFC** (negative frequency component): it actively hurts the correct prediction.

**Why this is a reasonable thing to do mathematically.** The authors also prove a supporting result (Remark 1): a convolution operation can be rewritten as a simple linear function of the image's frequency components. Since Shapley value is known to work especially well and give clean, intuitive results specifically for linear or additive systems, this gives a solid mathematical justification for applying Shapley value in the frequency domain, rather than just doing it because it seemed convenient.

## 5. Key Findings: What Shapley Values Reveal About AT

Using this frequency-based Shapley value, the authors run several analyses. The findings break down into three main groups.

### 5.1 Standard-trained (ST) models quietly depend on HFCs

- On **clean images**, low-frequency components (LFCs) have a strong positive Shapley value, while HFCs are close to zero. In plain terms: on normal images, the model is barely using high-frequency information at all.
- On **adversarial images**, the Shapley value of HFCs turns sharply negative. In plain terms: the model does depend on HFCs more than it looks like, and that hidden dependence is exactly what an attacker can exploit.

### 5.2 Adversarially-trained (AT) models trade away HFC information for robustness

- In AT models, HFCs stay close to zero in Shapley value on both clean and adversarial images.
- LFCs remain strongly positive in both cases.
- The authors' interpretation: adversarial training does not really teach the model to use HFCs safely. Instead, it teaches the model to mostly stop relying on HFCs at all.
- This gives a plausible explanation for the well-known clean accuracy vs. robustness trade-off: if HFCs genuinely help with clean accuracy, but the model is trained to ignore them for the sake of safety, some clean accuracy is naturally lost.

### 5.3 Robustness is not equally distributed across classes (a fairness problem)

- Even on a class-balanced dataset like CIFAR-10, the robust accuracy of an AT model differs a lot between classes, even though clean accuracy on an ST model is fairly similar across classes.
- The authors measure, for the ST model, how much each class's clean images rely on HFCs on average.
- They find a strong negative relationship between that HFC reliance and the class's robust accuracy after adversarial training (Pearson correlation of about -0.88).
- **Important distinction:** the authors are careful to call this a conjecture, not a proven cause. Their claim is: classes where HFCs matter more for the original model tend to end up less robust after adversarial training. This is a strong correlation they observed, not something they mathematically proved must be true.

## 6. The Defense: Class-wise Shapley Value-guided Augmentation (CSA)

**The idea.** If NFCs are frequency components that push a model's prediction toward the wrong class, then a model's understanding of "what a class looks like" is partly misaligned with what that class should actually look like. The authors propose correcting this directly during training.

**How CSA works, step by step:**

1. For each class, take a small set of clean training images.
2. For each of those images, extract only its NFCs (the frequency components hurting the correct prediction).
3. During training, augment other images from that same class by adding in a bit of this NFC information.
4. The intuition: this exposes the model to the specific "wrong-class-looking" frequency patterns for that class, so the model can learn to correct for them ahead of time, rather than being fooled by them later during an attack.

**Practical note:** computing a fully accurate Shapley value for every image would be expensive. The authors found that using NFCs from only a small subset of images per class, about 3 percent of the training set for CIFAR-10, was enough to see a meaningful improvement.

**Results.** CSA was tested on CIFAR-10 and CIFAR-100, combined with two existing adversarial training methods (PGD-AT and TRADES), and across three model architectures (ResNet-18, VGG16, WideResNet-28-10):

- CSA consistently improved robust accuracy under both PGD-20 and AutoAttack, two standard adversarial attack benchmarks.
- This improvement came at only a small cost to clean accuracy.
- The improvement held across all three tested architectures, suggesting CSA is not tied to one specific model design.
- The authors also compared CSA against a prior defense that works by directly suppressing (removing) high-frequency information. That method actually hurt accuracy under the ℓ∞ attack setting used in this paper, while CSA did not, suggesting CSA's more targeted, class-specific correction works better than simply discarding HFCs wholesale.

## 7. Works Cited

Chen, Yiting, Qibing Ren, and Junchi Yan. "Rethinking and Improving Robustness of Convolutional Neural Networks: a Shapley Value-based Approach in Frequency Domain." *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*, 2022.