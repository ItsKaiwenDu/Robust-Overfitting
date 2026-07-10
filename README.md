# Investigating Robust Overfitting in Adversarial Training

This is GitHub repository for research on robust overfitting in adversarial training.

---

## Project Overview
Deep neural networks can be easily fooled by adversarial attacks, which are small, hidden changes to inputs that cause model to make wrong predictions. Adversarial training helps fix this, but models often run into a problem known as **robust overfitting**. This means that later in training, model's performance on test attacks gets worse even though its training loss keeps improving.

Research Objectives:

* Replicate adversarial training results from Rice et al. (2020) paper using a PreActResNet-18 model on CIFAR-10.
* Find exact point where model stops learning real robustness and starts memorizing specific perturbation patterns.

---

## Research Team
* Principal Investigator: Dr. Nicholas Q. Tran (Department of Mathematics and Computer Science)
* Student Researcher: Kaiwen Du (Computer Science)

---

## Week 1 (Completed)
* **Progress Report:** During week 1, I focused on digesting the readings for our research project. The first paper I read is called “Explaining and Harnessing Adversarial Examples” by Goodfellow et al., and I learned about the Fast Gradient Sign Method (FGSM), which is a technique that tricks AI models into incorrectly and confidently seeing something else by adding tiny mathematical noise to an image input that is unnoticeable to human eyes. The second paper I read is called “Overfitting in Adversarially Robust Deep Learning” by Rice et al., and I learned about robust overfitting, which means a model's defense against attacks starts getting worse on test data even though it keeps improving on the training data. I also learned about many techniques for reducing robust overfitting, including Early Stopping, which is a method that stops the training process at the exact point where the model's test performance starts to drop, and this one happened to be the best and simplest method for preventing this overfitting problem. During the Friday meeting with my professor Dr. Tran, he suggested that I should condense my notes, which were way too detailed and long, down into presentation slides, and suggested I add some examples for better understanding, and to only focus on the Early Stopping method for training the model due to research time constraints.

* **Deliverables:** Reading notes have been written and are available in [`Notes/`](Notes/) directory:
  * [`Notes/Goodfellow.md`](Notes/Goodfellow.md): Covers linearity hypothesis, FGSM, adversarial training results on MNIST, and connections to our research.
  * [`Notes/Rice.md`](Notes/Rice.md): Covers robust overfitting phenomenon, effect of early stopping, comparison of regularization methods, and connections to our research.

---

## Week 2 (Current)
* **Progress Report:** This week, I focused on setting up the development environment and organizing the repository files. I successfully configured a Lambda Labs cloud account accessing NVIDIA A10 GPU for training the AI model in the upcoming weeks. I also listed the required Python package dependencies in [`requirements.txt`](requirements.txt) and created a script in [`verify_setup.py`](verify_setup.py) to verify that the everything works properly: version, imports, and system device capabilities. Finally, I downloaded and integrated the PreAct ResNet-18 model architecture (a standard neural network architecture used in adversarial training for its improved training stability and performance) from the Rice et al. paper into the project models folder for replicating robust overfitting experiments. During my meeting with Dr. Tran on Friday, he reviewed my presentation slides and provided feedback for major revisions. He noted that the current slides lack a logical flow and look like a mix of random ideas. He suggested adding a specific visual example of an adversarial attack, such as showing how an image of a bear turns into a truck when noise is added. He also asked me to include a clear math example for the adversarial formula and to properly define robust overfitting before discussing it. Finally, he requested that I send him a plan this weekend for next week, where my goal will be to implement the Projected Gradient Descent (PGD) adversarial training pipeline and run a small test on the cloud.

* **Deliverables:**
  * [`.gitignore`](.gitignore): Configured to ignore environment folders, datasets, cache files, and model checkpoints.
  * [`requirements.txt`](requirements.txt): Lists project dependencies (`torch`, `torchvision`, `numpy`, `matplotlib`, `tensorboard`).
  * [`Models/preact_resnet.py`](Models/preact_resnet.py): PyTorch implementation of `PreActResNet18` tailored for CIFAR-10 with standard pre-activation blocks, final BatchNorm and ReLU, and correct dimensions.
  * [`verify_setup.py`](verify_setup.py): Script to verify python imports, system device capabilities (CPU/MPS/CUDA), and run a forward pass sanity check.
  * [`cloud_setup.md`](cloud_setup.md): Guide for deploying runs to cloud instances (Lambda Labs), ssh configurations, code syncing, `tmux` sessions, and TensorBoard port forwarding.
  * [`Notes/Presentation.pdf`](Notes/Presentation.pdf): Slides summarizing notes and examples focusing on early stopping.

### Required Local Setup Instructions

1. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   ```
2. **Activate & install dependencies**:
   ```bash
   source .venv/bin/activate
   pip3 install -r requirements.txt
   ```
3. **Verify setup**:
   ```bash
   python3 verify_setup.py
   ```

---

## Week 3 (Up Next)
* **Objective:** Implement the Projected Gradient Descent (PGD) adversarial training pipeline, which is a training process where we generate slightly modified images (adversarial examples) designed to fool the neural network, and then train the model directly on them so it learns to ignore these perturbations. After that, we will verify its correctness on a diagnostic test run. This diagnostic run ensures our training script is bug-free and runs efficiently on the GPU before we launch the long, full-scale training process.

  1. Create a training script `train.py` containing a custom PyTorch training loop that generates adversarial perturbations via a 10-step PGD attack with a random initialization start ($\epsilon = 8/255$) using the [`Models/preact_resnet.py`](Models/preact_resnet.py) architecture.
  2. Implement checkpoint saving functionality in `train.py` to write model weights periodically (every 5 epochs) to a new `Checkpoints/` directory.
  3. Execute `train.py` for a diagnostic test run (1 epoch on 10% of CIFAR-10) to verify that loss decreases, gradients update without numerical issues, and GPU VRAM usage remains within limits.

* **Expectations:** A fully functional and verified training script (`train.py`) that successfully runs a short test training cycle and writes checkpoints, confirming the pipeline is ready for the full baseline training.

---

## Week 4
* **Objective:** Run the full baseline adversarial training on CIFAR-10 (a standard image dataset consisting of 60,000 everyday images across 10 categories) to replicate the robust overfitting baseline. This full run is necessary to observe and document the robust overfitting behavior where the model's defense starts decaying on test data later in training.

  1. Configure `train.py` and execute the full 100-epoch training schedule on the complete CIFAR-10 dataset using the Lambda Labs cloud instance.
  2. Ensure training parameters employ standard hyperparameters (SGD with momentum, weight decay, and a multi-step learning rate schedule). This means using standard mathematical optimization controls (e.g., speed of learning, penalizing extreme weight values, and slowing down learning at specific epochs) to ensure the network trains stably and achieves high accuracy.
  3. Save model weights every 5 epochs to the `Checkpoints/` directory to allow post-hoc analysis of the model's robustness progression.

* **Expectations:** A complete set of 20 model checkpoints saved in the `Checkpoints/` directory, representing the model's transition from clean accuracy to peak robustness and eventual robust overfitting.

---

## Week 5
* **Objective:** Evaluate training and test robust accuracy across all saved checkpoints to identify the robust overfitting point.

  1. Create a new evaluation script `evaluate.py` to run the saved model checkpoints from the `Checkpoints/` directory against both PGD-20 and FGSM attacks on the CIFAR-10 test set.
  2. Record the clean accuracy, robust accuracy, and training/test loss for each checkpoint in a CSV file or logs.
  3. Log evaluation metrics to a TensorBoard logging directory `runs/` to identify the exact epoch where test robust accuracy peaks and begins to decline.

---

## Week 6
* **Objective:** Execute verification runs with different random seeds to confirm the consistency of the identified overfitting point.

  1. Modify `train.py` to accept random seed arguments and run 3 additional training runs from scratch using different random seeds.
  2. Save and collect the training/test statistics at each epoch for each run under the TensorBoard `runs/` directory.
  3. Compare the peak robust accuracy epochs across seed runs to verify that the overfitting threshold is statistically consistent.

---

## Week 7
* **Objective:** Analyze experimental data, generate publication-quality visualizations, and document findings.

  1. Aggregate the training and evaluation metrics (mean and standard deviation of peak robust accuracy) from all seed runs in the `runs/` directory.
  2. Create a new plotting script `plot_results.py` to generate dual-axis line charts showing clean vs. robust accuracy and training vs. test loss over epochs.
  3. Draft the results section of the paper using the generated charts.

---

## Week 8
* **Objective:** Complete the final report, compile presentation materials, and prepare deliverables.

  1. Document the overall research findings by writing the final report `Report.md`.
  2. Populate the [`Notes/Presentation.pdf`](Notes/Presentation.pdf) with slides summarizing the literature, the PreAct ResNet-18 architecture, and empirical results.
  3. Refactor the repository scripts (`train.py`, `evaluate.py`, `plot_results.py`), clean up documentation, and ensure code reproducibility.

---

## References
* Goodfellow, I. J., Shlens, J., and Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
* Rice, L., Wong, E., and Kolter, J. Z. (2020). *Overfitting in adversarially robust deep learning.* ICML.