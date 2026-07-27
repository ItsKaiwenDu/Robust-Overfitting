## Reading Notes: Explaining and Harnessing Adversarial Examples
### Goodfellow, Shlens & Szegedy (ICLR 2015)

> **Summary:** Adversarial examples can be thought of as optical illusions designed specifically for AI. This landmark paper from Google challenges the common belief that neural networks are tricked by adversarial examples because they are overly complex or excessively focused on fine details. Instead, Goodfellow et al. explain that the opposite is true: modern neural networks are intentionally designed to behave linearly to make training easier. However, in spaces with thousands of input dimensions, this linear behavior creates a massive vulnerability. Tiny, invisible adjustments across many pixels add up like thousands of small nudges to create a massive shift, forcing the model to make incorrect predictions with high confidence. The authors demonstrate this vulnerability using the Fast Gradient Sign Method (FGSM), a fast, one-step technique to generate adversarial inputs using the model's own gradient calculations. They show that training models directly on these adversarial examples helps them resist attacks and even improves accuracy on clean images, setting a new benchmark on the MNIST dataset. Finally, they show that standard defense ideas (such as model ensembling) fail, and that adversarial examples transfer across different network architectures because different models learn similar decision boundaries.

---

### Definitions

* **Backpropagation**: The standard algorithm a neural network uses to calculate how much each internal setting contributed to an error, and then adjust those settings to improve performance. It works backward through the network's layers. FGSM reuses this same backward calculation, but instead of updating model settings, it uses the gradient to find the input direction that hurts the model most.
* **Dropout**: A regularization technique where the network randomly deactivates a fraction of its neurons during each training step. This prevents the model from depending too heavily on any single neuron and improves generalization on new, unseen data.
* **Fast Gradient Sign Method (FGSM)**: A fast, one-step attack that tricks a neural network by making small changes to the input based on the direction of the model's loss gradient, causing it to misclassify the input.
* **Linear**: A system where the output changes predictably and proportionally to its input (e.g., turning a volume knob up increases sound volume at a constant rate).
* **Long Short-Term Memory (LSTM)**: A specialized neural network layer designed to process sequential data, such as words in a sentence or audio signals over time. It dynamically chooses what information to store, maintain, or discard.
* **Maxout Networks**: A neural network layer design that evaluates multiple linear functions and outputs only the maximum value. This gives the model flexibility to learn complex patterns while keeping computation efficient.
* **Model Averaging**: Combining the predictions of several distinct models (an "ensemble") rather than relying on a single model, with the expectation that individual model errors will cancel out.
* **Modified National Institute of Standards and Technology (MNIST)**: A standard benchmark dataset of 70,000 handwritten digits (0–9) widely used to evaluate machine learning and computer vision algorithms.
* **Non-Linear**: A system where the output does not change in a simple, proportional manner and can respond abruptly depending on the input (e.g., a light switch).
* **Pretraining**: Training a model on an initial dataset or task before fine-tuning its learned parameters on the target task.
* **Radial Basis Function (RBF) Network**: A neural network model that makes confident predictions only for inputs that closely resemble its training data. When presented with unfamiliar data far from its training distribution, it defaults to low confidence rather than making an overconfident guess.
* **Rectified Linear Unit (ReLU)**: An activation function defined as $f(x) = \max(0, x)$. It passes positive values through unchanged and sets negative values to zero, allowing networks to learn complex functions efficiently while maintaining near-linear gradient flow.
* **Softmax**: A function applied at the final layer of a classifier that converts raw numerical scores (logits) into a normalized probability distribution across classes, ensuring all class probabilities sum to 100%.
* **Weight Decay / Regularization ($L_1$ Penalty)**: A training penalty that encourages internal model weights toward smaller or zero values, keeping the model simpler to prevent overfitting. Goodfellow et al. specifically analyze $L_1$ penalty constraints in comparison with adversarial training.

---

### The Linearity Problem in AI

* Researchers originally hypothesized that adversarial examples resulted from deep neural networks being overly complex, highly non-linear, and prone to overfitting.
* Goodfellow et al. demonstrate that adversarial examples primarily occur because modern neural networks behave too linearly in high-dimensional spaces.
* Common architectural building blocks—including *ReLUs*, *LSTMs*, and *Maxout Networks*—are explicitly designed to act linearly because linearity makes optimization faster and easier. The authors argue that this deliberate design choice creates vulnerability to small, coordinated input perturbations.

---

### Fragility in High-Dimensional Spaces

* Digital images consist of many individual pixels. Even a small 100×100 pixel RGB image contains 30,000 input values, placing the classifier in a 30,000-dimensional input space.
* An attacker can alter every pixel value by a tiny amount $\epsilon$ that is imperceptible to human eyes.
* In high-dimensional spaces, these tiny individual changes accumulate across all dimensions. Combined, they generate a large dot-product shift in the model's activation, driving the output across a decision boundary into a confident misclassification.
* Because human intuition is bounded by three-dimensional space, high-dimensional linear aggregation is counterintuitive.

---

### Fast Gradient Sign Method (FGSM)

* The authors introduced a fast method to compute adversarial perturbations in a single step using the model's backpropagation gradients.
* The perturbation vector $\eta$ is calculated in the direction that maximizes the model's loss $J$:

$$\eta = \epsilon \operatorname{sign}(\nabla_x J(\theta, x, y))$$

* **Note:** For deep neural networks, this formula represents a linear approximation of the loss surface; it is an exact worst-case perturbation only for linear models such as logistic regression.
* Adding a tiny perturbation fraction ($\epsilon = 0.007$) of this noise vector to an image of a panda causes the model to classify it as a gibbon (Goodfellow et al., 2015, Fig. 1).
* Result: The neural network is **99.3%** confident that the image is a gibbon, even though the image remains clearly recognizable as a panda to a human observer.

---

### Adversarial Training: Formulation & Results

* Traditional regularization techniques such as *Dropout*, *Pretraining*, and *Model Averaging* provide relatively **ineffective** protection against targeted adversarial perturbations.
* The authors propose *adversarial training*: injecting adversarial examples directly into the training batch alongside clean examples, continuously generating fresh perturbations against the current parameter state.
* The modified training objective combines clean and adversarial loss with weight factor $\alpha = 0.5$:

$$\tilde{J}(\theta, x, y) = \alpha J(\theta, x, y) + (1 - \alpha) J(\theta, x + \epsilon \operatorname{sign}(\nabla_x J(\theta, x, y)))$$

* On the MNIST dataset using a Maxout network, adversarial training reduced the error rate on adversarial inputs from **89.4% down to 17.9%**.
* **Caveat:** Even at 17.9% adversarial error, the model's average confidence on incorrect predictions remained high at **81.4%**, indicating that the model still makes overconfident mistakes, albeit less frequently.
* **Generalization Gains:** Adversarial training also improved clean test accuracy. Using an expanded Maxout network (1,600 units per layer instead of 240) trained with both dropout and adversarial examples, the authors achieved a benchmark error rate of **0.782%**. This outperformed the same architecture trained without adversarial examples (1.14% error, where extra capacity caused mild overfitting), matching top-performing dropout baselines on MNIST.

---

### Adversarial Training vs. $L_1$ Weight Decay

* Adversarial training shares mathematical similarities with $L_1$ weight decay, but the two methods function differently during optimization.
* In adversarial training, the effective perturbation penalty *disappears* once the model becomes sufficiently confident and assigns large margin boundaries to correct labels, making adversarial training self-regulating.
* $L_1$ weight decay applies a constant penalty regardless of margin size, overestimating the required regularization and degrading clean accuracy.
* For practical MNIST experiments, a standard $L_1$ weight decay coefficient ($\lambda = 0.0025$) proved too severe, causing training error to exceed 5%.

---

### Weight Visualization After Adversarial Training

* Prior to adversarial training, standard neural network weight filters display diffuse, noisy patterns that respond to seemingly random pixel configurations (Goodfellow et al., 2015, Fig. 3, left panel).
* Following adversarial training, the learned weight filters become visibly localized and structured, responding to coherent edge and shape features rather than high-frequency noise (Goodfellow et al., 2015, Fig. 3, right panel).
* Forcing the network to resist worst-case perturbations encourages it to rely on robust, semantically meaningful feature representations.

---

### Model Confidence & The RBF Tradeoff

* Radial Basis Function (RBF) networks evaluate inputs based on distance to stored prototypes, making them naturally resistant to adversarial examples: they predict with high confidence only when inputs lie near training data points.
* When fooled by out-of-distribution inputs, an RBF network's prediction confidence drops to **1.2%**, meaning the model accurately reflects uncertainty ("knows what it doesn't know").
* However, RBF networks struggle to generalize because they lack invariance to standard transformations, creating a fundamental precision-recall trade-off:
  - Linear models exhibit high recall (generalize well) but low precision (overconfident in unfamiliar regions).
  - RBF models exhibit high precision (low overconfidence) but low recall (poor generalization).

---

### Transferability of Adversarial Examples

* An adversarial example generated to fool one neural network model frequently fools another model, even if the second model uses a different architecture or was trained on a separate dataset.
* Adversarial perturbations do not rely on narrow, isolated points in input space; rather, they inhabit wide, continuous regions of misclassification.
* Because different classifiers trained on the same task learn similar linear decision boundaries, they share common high-dimensional blind spots.
* **Empirical Evidence:** Adversarial examples generated against a Maxout network transferred to a Softmax classifier, which agreed with the Maxout model's incorrect class label **84.6%** of the time on shared misclassifications.

---

### Failed Defense Strategies (Section 9)

* **Generative Pretraining:** A Deep Boltzmann Machine (MP-DBM) generative model still suffered a **97.5% error rate** under adversarial attack, demonstrating that generative training alone does not confer robustness.
* **Model Ensembling:** An ensemble of 12 Maxout networks still yielded a **91.1% error rate** when evaluated against adversarial examples targeted at the ensemble average (and 87.9% when targeted at individual ensemble members). Simple model averaging offers limited defense.

---

### Rubbish Class Examples

* Neural networks can be manipulated into making highly confident classifications on inputs containing no recognizable object features (pure noise).
* **Unmodified Noise:** Feeding 10,000 unperturbed Gaussian noise samples to a standard Maxout network resulted in **98.35%** of samples being classified as real digit classes with an average confidence of **92.8%**.
* **Targeted Noise Perturbation:** Taking a Gaussian noise sample and applying a single gradient step toward a target class (e.g., "airplane") causes the network to output that label with over **50%** confidence, even though the image remains visual static to a human.
* RBF networks achieve **0% error** on rubbish class examples because out-of-distribution inputs produce low confidence scores across all categories.

---

### Connection to Our Project

* **Foundational Framework:** Goodfellow et al. introduced the adversarial training paradigm that forms the core subject of our research into robust overfitting.
* **Methodological Lineage:** The Fast Gradient Sign Method (FGSM) is the single-step precursor to Projected Gradient Descent (PGD), the multi-step attack used across our experimental benchmarks.
* **Persistent Overconfidence:** The authors' initial observation—that adversarially trained models still exhibit high confidence on residual errors—foreshadows the generalization breakdown studied in robust overfitting.
* **Early Sighting of Robust Generalization Gap:** When scaling up network capacity, Goodfellow et al. noted that clean validation error stabilized while adversarial validation error diverged, requiring early stopping tuned to adversarial validation loss. This represents an early observation of the decoupling between clean and robust validation metrics later detailed by Rice et al.
* **Theoretical Context:** The linearity hypothesis provides an analytical framework for interpreting robust overfitting: if models learn linear decision boundaries that memorize perturbation directions rather than true data geometry, robustness degrades during extended training.

---

### Works Cited

* Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and Harnessing Adversarial Examples." *International Conference on Learning Representations*, 2015. arXiv: https://arxiv.org/abs/1412.6572.