## Reading Notes: Explaining and Harnessing Adversarial Examples
### Goodfellow, Shlens & Szegedy (ICLR 2015)

> **Summary:** Adversarial examples can be thought of as optical illusions, but for AI. This academic paper from Google challenges common belief that neural networks are tricked by adversarial examples because they are too complex or over-focused on details. Instead, Goodfellow et al. explain that opposite is true: modern networks are designed to act linearly to make training easier, but in spaces with multiple dimensions, this straightness creates a massive blind spot. Because of this, tiny changes to every pixel can add up like thousands of small nudges to form a giant push, easily forcing model to make a wrong prediction with high confidence. The authors show this with Fast Gradient Sign Method (FGSM), a quick, one-step method to generate these tricky inputs using model's own feedback math. They demonstrate that training models on these tricky examples helps them resist attacks and even makes them more accurate on normal images, setting a new record on MNIST dataset. Finally, they show that common defense ideas (like combining different models) fail, and that these tricky examples can fool multiple models because different networks learn similar boundaries.

---

### Definitions

* **Backpropagation**: The method a neural network uses to figure out how much each of its internal settings contributed to a mistake, then adjust them to do better. It works backward through the network's layers, which is where the name comes from. FGSM reuses this same backward calculation, but instead of using it to learn, it uses it to find the direction that hurts the model most.
* **Dropout**: A training trick where the network randomly turns off a portion of its neurons at each training step. This stops the model from leaning too heavily on any single neuron and tends to help it perform better on new, unseen data.
* **Fast Gradient Sign Method (FGSM)**: A fast, one-step attack that tricks a neural network by making small changes to the input based on the model's error gradient, causing it to misclassify the data.
* **Linear**: Model's output changes predictably and proportionally to its input (eg. volume knob).
* **Long Short-Term Memory (LSTMs)**: Special type of AI layer designed to handle data that comes in a specific order, like words in a sentence or notes in music. It decides what information to remember for a long time, what to use right now, and what to forget.
* **Maxout Networks**: A neural network design that uses multiple linear functions and keeps only the largest output. This gives the model more flexibility to learn complex patterns while remaining efficient to compute.
* **Model Averaging**: Combining the predictions of several different models, often called an "ensemble," instead of relying on just one. The hope is that the models' individual mistakes cancel each other out.
* **Modified National Institute of Standards and Technology (MNIST)**: A widely used dataset of handwritten digits (0-9) that serves as a benchmark for training and testing machine learning and computer vision models.
* **Non-Linear**: Model's output does not change in a simple proportional way and can respond differently depending on the input (e.g., a light switch).
* **Pretraining**: Training a model on a separate, often larger or simpler task first, then reusing those learned settings as a starting point before training it on the real target task.
* **Radial Basis Function (RBF) Network**: A type of model that only makes confident predictions for inputs that closely resemble its training data. Far away from that data, it defaults to low confidence instead of guessing.
* **Rectified Linear Units (ReLUs)**: An activation function that acts like a filter in a neural network. If the input is positive, it passes through unchanged. If the input is negative, it becomes zero. This simple approach helps neural networks learn complex patterns efficiently.
* **Softmax**: A function placed at the end of a classifier that turns the model's raw output scores into probabilities for each possible class, with all the probabilities adding up to 100%.
* **Weight Decay (L¹ Regularization)**: A training penalty that pushes a model's internal weights toward smaller values. The goal is to keep the model simpler so it is less likely to overfit to the training data.

---

### Linearity Problem in AI

* Scientists originally thought adversarial examples were caused by deep networks being too complex and nonlinear.
* This paper argues that adversarial examples happen mainly because modern AI models behave too linearly in high-dimensional spaces.
* Many modern neural network components, including *ReLUs*, *LSTMs*, and *Maxout Networks*, are designed to be more linear since linear behavior makes training easier. The authors argue that this design choice makes models vulnerable to small adversarial perturbations.

---

### High Dimensions Fragility

* Images are made of many pixels. Even a small 100×100 RGB image contains 30,000 input values, meaning the model operates in a 30,000-dimensional space.
* An attacker can change every input value by a tiny, nearly invisible amount.
* In high-dimensional spaces, these tiny changes add up. They can create a large shift in the model's output together and ultimately cause a confident misclassification.
* It is hard for us as humans to imagine this phenomenon since we live in 3D space.

---

### Fast Gradient Sign Method (FGSM)

* The authors created a fast way to generate adversarial examples using model's own internal math: **Backpropagation**.
* The formula computes a perturbation in direction that most increases model's loss:

$$\eta = \epsilon \operatorname{sign}(\nabla_x J(\theta, x, y))$$

* Note: For neural networks this is an *approximation* based on linearizing cost function; it is only exact worst-case perturbation for simpler models like logistic regression.
* Adding a tiny fraction ($\epsilon = 0.007$) of this calculated noise turns a picture of a panda into a gibbon (Goodfellow et al., 2015, Fig. 1).
* Result: AI is **99.3%** confident that it sees a gibbon, despite image clearly shows a panda to human eye.

---

### Adversarial Training: Methods & Results

* Traditional regularization safeguards such as *Dropout*, *Pretraining*, and *Model Averaging* are relatively **ineffective** in reducing vulnerability to these attacks.
* Authors propose mixing adversarial examples directly into training data alongside clean examples, continually regenerating them against current version of model.
* Training objective becomes a blend, with $\alpha = 0.5$:

$$\tilde{J}(\theta, x, y) = \alpha J(\theta, x, y) + (1 - \alpha) J(\theta, x + \epsilon \operatorname{sign}(\nabla_x J(\theta, x, y)))$$

* On MNIST with a maxout network, this dropped error rate on adversarial examples from **89.4% down to 17.9%**.
* Caveat: Even at 17.9%, model's average confidence on its wrong answers remained high at **81.4%**, meaning it still fails confidently, just less often.
* Bonus result: Adversarial training also improved clean accuracy. Using a larger maxout network (1,600 units per layer instead of 240) trained with both dropout and adversarial training, the authors reached a best-reported **0.782% error rate** on permutation-invariant MNIST. This beats the same architecture trained with dropout alone (0.94% error without adversarial training), but it is statistically indistinguishable from the best existing dropout-based result on this benchmark, a fine-tuned DBM at 0.79% error (Goodfellow et al., 2015). So the accurate takeaway is that adversarial training matched the best known regularizer on this task, not that it clearly surpassed it.

---

### Adversarial Training vs. Weight Decay

* Adversarial training looks superficially similar to L¹ regularization, but they differ in an important way.
* In adversarial training, penalty term can *disappear* once model becomes confident enough, meaning that it is self-regulating.
* L¹ weight decay is more pessimistic: It keeps applying penalty even in cases where model already has good margin, overestimating damage an adversary can actually do.
* For practical training on MNIST, an L¹ coefficient that would seem reasonable (0.0025) was already too large and caused over 5% training error.

---

### How Training Cleans Up Model Weights

* Before adversarial training, a naively trained model has messy, diffuse weight filters that respond to apparently random parts of an image (Goodfellow et al., 2015, Fig. 3, left panel).
* After adversarial training, model's filters become noticeably more localized and interpretable, meaning that they respond to specific features rather than noise (Goodfellow et al., 2015, Fig. 3, right panel).
* Forcing model to handle worst-case inputs pushes it to learn genuinely meaningful features for recognition.

---

### Different Models & Confidence: RBF Tradeoff

* Humans have poor instincts for high-dimensional spaces because we only live in three dimensions.
* Radial Basis Function (RBF) networks are naturally resistant to adversarial examples: they only predict confidently near their training data, so when fooled they drop confidence to just **1.2%**, meaning that model correctly "knows what it doesn't know."
* However, RBF networks cannot generalize well because they are not invariant to significant transformations of their input, meaning that high precision comes at cost of recall.
* The paper frames this as a fundamental precision-recall tradeoff: linear models have high recall (respond to everything in right direction) but low precision (overconfident in unfamiliar regions); RBF models are reverse.

---

### Adversarial Examples Transfer Between Models

* An adversarial example built to fool one AI model will usually fool a completely different model too, even one with a different architecture trained on different data.
* Adversarial examples do not exist in tiny isolated pockets; they occupy wide, continuous regions of input space.
* Because different models trained on same task learn similar linear decision boundaries, they share same blind spots.
* Evidence: when adversarial examples from a maxout network were tested on a softmax classifier, it agreed with maxout network's wrong label **84.6%** of time on shared mistakes.

---

### Defenses That Don't Work (Section 9)

* **Generative training:** An MP-DBM generative model still achieved a **97.5% error rate** on adversarial examples, meaning that being generative alone is not sufficient.
* **Ensembling:** An ensemble of 12 maxout networks still got a **91.1% error rate** on adversarial examples targeted at whole ensemble (87.9% when targeted at a single member). Averaging over many models provides only marginal resistance.

---

### Rubbish Class Examples

* AI models can be tricked by inputs that contain no real objects at all, which are pure noise.
* In basic demonstration, authors fed 10,000 samples of random Gaussian noise directly into classifiers with no modification. A naively trained maxout network classified **98.35%** of these as real objects, with **92.8%** average confidence.
* In a targeted variant, authors took a Gaussian noise sample and applied a single gradient step toward a specific category (e.g., "airplane"). To a human image still looks like colorful static, but network classifies it as an airplane with over **50%** confidence.
* RBF networks are immune to this: they achieve **0% error** on rubbish examples because they cannot confidently predict any class far from their training data.

---

### Connection to Our Research

* This paper invented adversarial training, the exact defense our project investigates. Without Goodfellow et al., there is no adversarial training pipeline for Rice et al. to later discover robust overfitting in.
* FGSM is conceptual predecessor to PGD, which is a stronger attack our project uses. Understanding one-step logic of FGSM is necessary background before PGD's multi-step version makes sense.
* The paper's own caveat points directly to our research question: even after adversarial training, model still failed with high average confidence on wrong answers. Robust overfitting is a deeper version of this same problem.
* Early hint of the phenomenon itself: when the authors scaled up their maxout network, they noticed clean validation error stayed flat over training while adversarial validation error did not, so they had to switch to early stopping based on the adversarial validation error specifically. This is a direct, early sighting of clean and robust performance decoupling during training, which is the same gap Rice et al. later names and studies as robust overfitting.
* The linearity explanation gives us a lens for interpreting our results. If models memorize perturbation directions instead of learning truly robust boundaries, that is exactly what paper predicts linear models are biased to do.
* The paper frames robustness and trainability as a fundamental tension: "models that are easy to optimize are easy to perturb." Our overfitting point is exact epoch where that tension breaks down.

---

### Works Cited

* Goodfellow, Ian J., et al. "Explaining and Harnessing Adversarial Examples." *International Conference on Learning Representations*, 2015. arXiv, https://arxiv.org/abs/1412.6572.