# Weekly Goals & Objectives

This document tracks the weekly goals, objectives, expectations, and deliverables for the research project on investigating robust overfitting in adversarial training.

---

## Week 1 (Completed)
* **Objective:** Review and summarize foundational literature on adversarial training and robust overfitting.

  1. Read and digest "Explaining and Harnessing Adversarial Examples" by Goodfellow et al. (2014). Learn definitions of adversarial examples/attacks, how they are generated via Fast Gradient Sign Method (FGSM), analyzing adversarial examples, and learn about how to defend these adversarial attacks.
  2. Read and digest "Overfitting in Adversarially Robust Deep Learning" by Rice et al. (2020). Learn about robust overfitting during adversarial training, and learn what are some methods to stop/reduce robust overfitting.
  3. Compile detailed reading notes for ourselves, which will be saved to the [`Notes`](Notes) folder.

* **Expectations:** A solid theoretical understanding of adversarial examples and early stopping.

* **Progress Report & Deliverables:** Documented in [`Report/progress.md`](Report/progress.md#week-1-completed).

---

## Week 2 (Completed)
* **Objective:** Configure the local development environment, set up cloud compute resources, integrate the PreActResNet-18 model architecture, verify the setup with a verification script, and create the presentation slides based on feedback.

  1. Configure required Python virtual environment, set up gitignore, and specify dependencies in `requirements.txt`.
  2. Create a Lambda Labs cloud account, set up billing information and payment methods, and gain familiarity with the cloud platform for accessing high-performance GPU resources (such as NVIDIA A10G instances) needed to train deep learning models.
  3. Integrate the PreActResNet-18 model architecture in PyTorch (`Models/preact_resnet.py`), which will be downloaded from standard implementations (as adopted in the Rice et al. 2020 codebase) for the purpose of ensuring exact experimental replication.
  4. Write a setup verification script (`scripts/verify_setup.py`) to verify package imports, check hardware/device availability (such as CUDA/MPS/CPU), and run a forward pass sanity check with the model to ensure the training environment is ready to go.
  5. Create presentation slides summarizing the literature review with visual and mathematical explanations of adversarial attacks. These slides communicate our foundational understanding to the PI and will be reviewed and refined during the Friday meeting with Dr. Tran.

* **Expectations:** A fully operational local and cloud training environment, model implementation complete, and a completed presentation (with placeholders reserved for our own research findings) with a clear and logical flow.

* **Progress Report & Deliverables:** Documented in [`Report/progress.md`](Report/progress.md#week-2-completed).

---

## Week 3 (Completed)
* **Objective:** Implement the Projected Gradient Descent (PGD)-based adversarial training pipeline and verify its correctness on a diagnostic test run. Lambda Labs will not be used yet this week; all local.

  1. Write the training script [`scripts/train.py`](scripts/train.py) using the [`PreActResNet18`](Models/preact_resnet.py#L153) architecture. Script should dynamically generate adversarial images using a 10-step PGD attack, which starts with random noise and makes pixel adjustments within a safety range (the $\epsilon$-ball of $8/255$) so the edits remain invisible to human eyes. All of these should match exactly as in Rice et al. paper.
  2. Configure and confirm that [`scripts/train.py`](scripts/train.py) saves model weights every 5 epochs to `Checkpoints/` folder. This allows us to analyze model's performance at different stages of training and pinpoint exactly where robust overfitting begins.
  3. Configure and run a diagnostic test locally (for 1 epoch on 10% of CIFAR-10 data) to ensure the training loop, PGD attack, weight updates, checkpoints, and logging all work together without any memory crashes before moving to Lambda Labs.

* **Expectations:** A fully functional and verified training script (`scripts/train.py`), ready for full training runs on Lambda Labs next week.

* **Progress Report & Deliverables:** Documented in [`Report/progress.md`](Report/progress.md#week-3-completed).

---

## Week 4 (Completed)
* **Objective:** Review and consolidate understanding of the Week 3 training pipeline, improve presentation skills and polish presentation slides, and run full training on Lambda Labs on Friday.

  1. Review [`scripts/train.py`](scripts/train.py) alongside the Week 3 progress report to make sure the full pipeline is understood and can be clearly explained.
  2. Read "How To Give Strong Technical Presentations" provided by Dr. Tran and watch public speaking YouTube videos to improve delivery for upcoming presentations.
  3. Polish presentation slides, from teleprompter to figures and key talking points, with details spoken out loud from speaking notes.
  4. Adjust upcoming weekly schedules.
  5. Deploy `scripts/train.py` to Lambda Labs and start the full 200-epoch PGD adversarial training run, monitor for any unexpected timeouts or crashes, and download all model checkpoints once training completes (expected run time: 7-10 hours).

* **Expectations:** A complete set of 40 model checkpoints in the `Checkpoints/` directory covering the full 200-epoch run (saved every 5 epochs), ready for evaluation.

* **Progress Report & Deliverables:** Documented in [`Report/progress.md`](Report/progress.md#week-4-completed).

---

## Week 5 (Completed)
* **Objective:** Evaluate model robustness across all saved checkpoints to identify the robust overfitting point.

  1. Create an evaluation script `scripts/evaluate.py` that loads each of the 40 checkpoints produced by the Week 4 training run and tests them using PGD-20 on the CIFAR-10 test set. PGD-20 uses 20 attack steps instead of 10 used during training, making it a stronger test that gives a more rigorous and honest measure of how well each checkpoint actually holds up against adversarial images.
  2. Record the clean accuracy, robust accuracy, and loss for each checkpoint in a CSV file so results can be reviewed conveniently.
  3. Plot the training and test robust accuracy curves across all 200 epochs using `matplotlib` to identify the exact epoch where test robust accuracy peaks and begins to decline.

* **Expectations:** A completed `scripts/evaluate.py`, a populated results CSV (`Report/evaluation_results.csv`), and a clear plot (`Report/robust_overfitting_curves.png`) identifying the peak robust accuracy epoch (Epoch 105 at 45.91%).

* **Progress Report & Deliverables:** Documented in [`Report/progress.md`](Report/progress.md#week-5-completed).

---

## Week 6 (Upcoming)
* **Objective:** Run 3 additional training runs with different random seeds to confirm that the overfitting point identified in Week 5 is consistent and not a one-time result.

  1. Run `scripts/train.py` 3 more times from scratch using different `--seed` values. Using different starting points rules out the possibility that the overfitting point found in Week 5 was a coincidence of one particular run.
  2. `scripts/train.py` automatically saves TensorBoard logs for each run to the `runs/` directory, so accuracy and loss curves for all runs are captured without any extra setup.
  3. Compare the peak robust accuracy epochs across all 4 runs (original plus 3 new) using `matplotlib` to confirm the overfitting point appears consistently.

* **Expectations:** Four total training runs (original plus 3 seed runs) with a confirmed consistent overfitting point across all runs.

---

## Week 7 (Upcoming)
* **Objective:** Analyze results from all runs, generate final plots, and draft the findings section of the report.

  1. Aggregate the accuracy and loss metrics from all 4 training runs to compute the average and spread of the peak robust accuracy epoch across seeds.
  2. Create a plotting script `scripts/plot_results.py` that generates line charts showing clean vs. robust accuracy and training vs. test loss across all epochs and runs.
  3. Draft the results section of the final report using the generated charts and aggregated numbers.

* **Expectations:** A completed `scripts/plot_results.py` with final charts and a drafted results section.

---

## Week 8 (Upcoming)
* **Objective:** Write the final report, update the presentation with our results, and clean up the repository.

  1. Write the final report `Report.md` covering the research methodology, experimental setup, results, and conclusions.
  2. Update [`Report/presentation.pdf`](Report/presentation.pdf) to include our experimental results and findings alongside the existing literature slides.
  3. Clean up and add comments to `scripts/train.py`, `scripts/evaluate.py`, and `scripts/plot_results.py` so the code is easy to follow, and make sure `README.md` and `requirements.txt` are easy to follow for anyone else to run the project and replicate our behavior findings.

* **Expectations:** A submitted final report, an updated presentation with our results, and a clean public GitHub repository.
