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
* **Objective:** Review and summarize foundational literature on adversarial training and robust overfitting.

  1. Read and digest "Explaining and Harnessing Adversarial Examples" by Goodfellow et al. (2014). Learn definitions of adversarial examples/attacks, how they are generated via Fast Gradient Sign Method (FGSM), analyzing adversarial examples, and learn about how to defend these adversarial attacks.
  2. Read and digest "Overfitting in Adversarially Robust Deep Learning" by Rice et al. (2020). Learn about robust overfitting during adversarial training, and learn what are some methods to stop/reduce robust overfitting.
  3. Compile detailed reading notes for ourselves, which will be saved to the [`Notes`](Notes) folder.

* **Expectations:** A solid theoretical understanding of adversarial examples and early stopping.

* **Progress Report & Deliverables:** Documented in [`Notes/Progress_report.md`](Notes/Progress_report.md#week-1-completed).

---

## Week 2 (Completed)
* **Objective:** Configure the local development environment, set up cloud compute resources, integrate the PreActResNet-18 model architecture, verify the setup with a verification script, and create the presentation slides based on feedback.

  1. Configure required Python virtual environment, set up gitignore, and specify dependencies in `requirements.txt`.
  2. Create a Lambda Labs cloud account, set up billing information and payment methods, and gain familiarity with the cloud platform for accessing high-performance GPU resources (such as NVIDIA A10G instances) needed to train deep learning models.
  3. Integrate the PreActResNet-18 model architecture in PyTorch (`Models/preact_resnet.py`), which will be downloaded from standard implementations (as adopted in the Rice et al. 2020 codebase) for the purpose of ensuring exact experimental replication.
  4. Write a setup verification script (`verify_setup.py`) to verify package imports, check hardware/device availability (such as CUDA/MPS/CPU), and run a forward pass sanity check with the model to ensure the training environment is ready to go.
  5. Create presentation slides summarizing the literature review with visual and mathematical explanations of adversarial attacks. These slides communicate our foundational understanding to the PI and will be reviewed and refined during the Friday meeting with Dr. Tran.

* **Expectations:** A fully operational local and cloud training environment, model implementation complete, and a completed presentation (with placeholders reserved for our own research findings) with a clear and logical flow.

* **Progress Report & Deliverables:** Documented in [`Notes/Progress_report.md`](Notes/Progress_report.md#week-2-completed).

### Required Local Setup Instructions

1. **Create virtual environment** *(an isolated Python workspace that keeps this project's dependencies separate from other Python projects on your system)*:
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

## Week 3 (Current)
* **Objective:** Implement the Projected Gradient Descent (PGD)-based adversarial training pipeline and verify its correctness on a diagnostic test run. Lambda Labs will not be used yet this week; all local.

  1. Create a training script `train.py` with a custom PyTorch training loop that generates adversarial examples via a 10-step PGD attack (where $\epsilon = 8/255$ caps the maximum per-pixel perturbation to keep changes imperceptible, $\alpha = 2/255$ sets the step size per iteration, and the attack initializes from a random point within the $\epsilon$-ball around each clean image) using the [`Models/preact_resnet.py`](Models/preact_resnet.py) architecture. This is the core training procedure we will replicate from Rice et al. (2020) and must match their setup exactly for our results to be comparable.
  2. Implement checkpoint saving in `train.py` to write model weights every 5 epochs to a new `Checkpoints/` directory. Periodic checkpoints let us reconstruct the model's robustness trajectory across training and are required for the epoch-level analysis in Weeks 5 and 6.
  3. Execute `train.py` for a short diagnostic run (1 epoch on 10% of CIFAR-10) to confirm that loss decreases, gradients update without numerical issues, and GPU VRAM usage stays under 8 GB. This smoke test catches implementation bugs before we commit to an expensive full training run on the cloud.

* **Expectations:** A fully functional and verified training script (`train.py`) that completes a short test cycle and writes checkpoints correctly, confirming the pipeline is ready for full baseline training.

---

## Week 4
* **Objective:** Run the full baseline adversarial training on CIFAR-10 using the Lambda Labs cloud instance and collect model checkpoints across all 100 epochs.

  1. Launch the full 100-epoch PGD adversarial training run on the complete CIFAR-10 dataset on the Lambda Labs cloud instance. This is the primary replication run that will generate the learning curves needed to observe and document robust overfitting as reported by Rice et al.
  2. Confirm that training hyperparameters match the standard setup from Rice et al.: Stochastic Gradient Descent (SGD) with momentum, weight decay, and a piecewise learning rate schedule with drops at epochs 100 and 150. Matching these exactly is necessary for our results to be directly comparable to the paper.
  3. Save model weights every 5 epochs to the `Checkpoints/` directory throughout training. These checkpoints allow us to reconstruct the model's full robustness trajectory and are required for the epoch-level evaluation in Week 5.

* **Expectations:** A complete set of 20 model checkpoints in the `Checkpoints/` directory covering the full 100-epoch run, ready for evaluation.

---

## Week 5
* **Objective:** Evaluate training and test robust accuracy across all saved checkpoints to identify the robust overfitting point.

  1. Create a new evaluation script `evaluate.py` to run each checkpoint from the `Checkpoints/` directory against both PGD-20 and FGSM attacks on the CIFAR-10 test set. Using two different attacks gives a more complete picture of robustness and checks whether our findings hold beyond the attack used during training.
  2. Record the clean accuracy, robust accuracy, and training/test loss for each checkpoint in a structured CSV file. Storing results in a structured format makes it easy to load and analyze across sessions without re-running evaluation.
  3. Log all evaluation metrics to a TensorBoard logging directory `runs/` and visualize the learning curves to identify the exact epoch where test robust accuracy peaks and begins to decline. This epoch is the overfitting point, the central empirical finding of our replication.

* **Expectations:** A completed `evaluate.py`, a populated results CSV, and TensorBoard logs with a clearly identified epoch where test robust accuracy peaks and begins to decline.

---

## Week 6
* **Objective:** Execute verification runs with different random seeds to confirm the consistency of the identified overfitting point.

  1. Modify `train.py` to accept a random seed argument and run 3 additional full training runs from scratch using different random seeds. Repeating training with different seeds rules out the possibility that the overfitting point identified in Week 5 was a statistical artifact of one particular initialization.
  2. Save training and test robust accuracy at each epoch for each run to the TensorBoard `runs/` directory. Keeping all runs in one place makes it straightforward to aggregate and compare them side by side.
  3. Compare the peak robust accuracy epochs across all seed runs to verify that the overfitting point is statistically consistent. Consistency across seeds strengthens the claim that the identified epoch is a reliable diagnostic marker rather than noise.

* **Expectations:** Four total training runs (original plus 3 seed runs) with full epoch-level accuracy logs, and a confirmed consistent overfitting point across all runs.

---

## Week 7
* **Objective:** Analyze all experimental results, produce publication-quality visualizations, and draft the findings section of the paper.

  1. Aggregate training and evaluation metrics across all seed runs in the `runs/` directory, computing mean and standard deviation of peak robust accuracy. Aggregating across seeds gives us statistically grounded numbers to report rather than a single-run result.
  2. Create a plotting script `plot_results.py` to generate dual-axis line charts showing clean vs. robust accuracy and training vs. test loss over epochs for all runs. These charts are the primary visual evidence for our findings and will appear in the final report.
  3. Draft the results section of the final report using the generated charts and aggregated statistics. Writing the results section while the data is fresh ensures our analysis stays grounded in what we actually observed.

* **Expectations:** A completed `plot_results.py` with publication-ready charts and a drafted results section ready for PI review.

---

## Week 8
* **Objective:** Complete the final report, update presentation materials with our results, and prepare the repository for publication.

  1. Write the final report `Report.md` documenting the full research methodology, experimental setup, results, and conclusions. This is the primary written deliverable of the project and the basis for the conference submission.
  2. Update [`Notes/Presentation.pdf`](Notes/Presentation.pdf) to incorporate the empirical results and findings from our experiments. Adding our own experimental results to the existing literature slides completes the presentation for the symposium and any conference submission.
  3. Refactor the repository scripts (`train.py`, `evaluate.py`, `plot_results.py`), clean up documentation, and verify full end-to-end reproducibility by running the pipeline on a clean environment. A clean, reproducible codebase is required for our GitHub publication commitment and for any external replication.

* **Expectations:** A submitted final report, an updated presentation with empirical results, and a fully reproducible public GitHub repository.

---

## References
* Goodfellow, I. J., Shlens, J., and Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
* Rice, L., Wong, E., and Kolter, J. Z. (2020). *Overfitting in adversarially robust deep learning.* ICML.