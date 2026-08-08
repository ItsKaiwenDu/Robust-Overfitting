# Weekly Goals & Objectives

This document tracks the weekly goals, objectives, expectations, and deliverables for the research project on investigating robust overfitting in adversarial training.

---

## Week 1 (Completed)
* **Objective:** Review and summarize foundational literature on adversarial training and robust overfitting.

  1. Read and digest "Explaining and Harnessing Adversarial Examples" by Goodfellow et al. (2014). Learn definitions of adversarial examples/attacks, how they are generated via Fast Gradient Sign Method (FGSM), analyzing adversarial examples, and learn about how to defend these adversarial attacks.
  2. Read and digest "Overfitting in Adversarially Robust Deep Learning" by Rice et al. (2020). Learn about robust overfitting during adversarial training, and learn what are some methods to stop/reduce robust overfitting.
  3. Compile detailed reading notes for ourselves, which will be saved to the [`Notes`](Notes) folder.

* **Expectations:** A solid theoretical understanding of adversarial examples and early stopping.

* **Progress Report & Deliverables:** Documented in [`progress.md`](progress.md#week-1-completed).

---

## Week 2 (Completed)
* **Objective:** Configure the local development environment, set up cloud compute resources, download PreActResNet-18 model architecture, verify setup with a verification script, and create the presentation slides based on feedback.

  1. Configure required Python virtual environment, set up gitignore, and specify dependencies in `requirements.txt`.
  2. Create a Lambda Labs cloud account and set up billing to access NVIDIA A10G GPU instances for the Week 4 training run.
  3. Download PreActResNet-18 model architecture in PyTorch (`Models/preact_resnet.py`) from the Rice et al. 2020 codebase so our training setup matches theirs.
  4. Write a setup verification script (`scripts/verify_setup.py`) to verify package imports, check hardware/device availability (such as CUDA/MPS/CPU), and run a forward pass sanity check with the model to ensure the training environment is ready to go.
  5. Create presentation slides summarizing the literature review with visual and mathematical explanations of adversarial attacks. These slides communicate our foundational understanding to the PI and will be reviewed and refined during the Friday meeting with Dr. Tran.

* **Expectations:** A fully operational local and cloud training environment, model implementation complete, and a completed presentation (with placeholders reserved for our own research findings) with a clear and logical flow.

* **Progress Report & Deliverables:** Documented in [`progress.md`](progress.md#week-2-completed).

---

## Week 3 (Completed)
* **Objective:** Implement the Projected Gradient Descent (PGD)-based adversarial training pipeline and verify its correctness on a diagnostic test run. Lambda Labs will not be used yet this week; all local.

  1. Write the training script [`scripts/train.py`](scripts/train.py) using the [`PreActResNet18`](Models/preact_resnet.py#L153) architecture. Script should dynamically generate adversarial images using a 10-step PGD attack, which starts with random noise and makes pixel adjustments within a safety range (the $\epsilon$-ball of $8/255$) so the edits remain invisible to human eyes. All of these should match exactly as in Rice et al. paper.
  2. Make [`scripts/train.py`](scripts/train.py) save model weights every 5 epochs to `Checkpoints/` folder. This allows us to analyze model's performance at different stages of training and pinpoint exactly where robust overfitting begins.
  3. Make and run diagnostic test locally (for 1 epoch on 10% of CIFAR-10 data) to ensure training loop, PGD attack, weight updates, checkpoints, and logging all work without complications before full training on Lambda Labs.

* **Expectations:** A fully functional and verified training script (`scripts/train.py`), ready for full training runs on Lambda Labs next week.

* **Progress Report & Deliverables:** Documented in [`progress.md`](progress.md#week-3-completed).

---

## Week 4 (Completed)
* **Objective:** Review Week 3 training pipeline, improve speaking skills and polish presentation slides, and run full training on Lambda Labs on Friday.

  1. Review [`scripts/train.py`](scripts/train.py) alongside the Week 3 progress report to make sure the full pipeline is understood.
  2. Read "How To Give Strong Technical Presentations" provided by Dr. Tran and watch YouTube videos on public speaking practices.
  3. Polish presentation slides, from teleprompter to figures and key talking points, with details spoken out loud from speaking notes.
  4. Adjust upcoming weekly schedules.
  5. Deploy `scripts/train.py` to Lambda Labs and start the full 200-epoch PGD adversarial training run, monitor for any unexpected timeouts or crashes, and download all model checkpoints once training completes (expected run time: 7-10 hours).

* **Expectations:** A complete set of 40 model checkpoints in the `Checkpoints/` directory covering the full 200-epoch run (saved every 5 epochs), ready for evaluation.

* **Progress Report & Deliverables:** Documented in [`progress.md`](progress.md#week-4-completed).

---

## Week 5 (Completed)
* **Objective:** Evaluate model robustness across all saved checkpoints to identify the robust overfitting point.

  1. Create `scripts/evaluate.py` to load each of the 40 Week 4 checkpoints and test them using PGD-20 on the CIFAR-10 test set. PGD-20 uses 20 attack steps instead of 10 used during training, making it a harder and more reliable measure of adversarial robustness.
  2. Record clean accuracy, robust accuracy, and loss for each checkpoint to a CSV file.
  3. Plot the training and test robust accuracy curves across all 200 epochs using `matplotlib` to identify the exact epoch where test robust accuracy peaks and begins to decline.

* **Expectations:** A completed test evaluation, a populated results CSV, and a clear plot showing peak robust accuracy epoch.

* **Progress Report & Deliverables:** Documented in [`progress.md`](progress.md#week-5-completed).

---

## Week 6 (Completed)
* **Objective:** Assess promising directions for the project by studying Cross-Epoch Transfer Matrices, reviewing the novelty of our research question, and considering a possible case-study approach.

  1. Learn how a Cross-Epoch Transfer Matrix can compare adversarial perturbations across model checkpoints, and identify what this analysis can and cannot explain about robust overfitting.
  2. Revisit our research question and hypothesis, then review related published work to determine whether the main question has already been answered and whether our proposed contribution is sufficiently specific and novel.
  3. Explore Dr. Tran's proposal to recast the project as a case study on the benefits and limitations of using AI tools to conduct research about AI.

* **Expectations:** A clearer understanding of the limits of Cross-Epoch Transfer Matrix analysis, an informed assessment of the project's research novelty, and a set of discussion points for selecting the next direction with Dr. Tran.

* **Progress Report & Deliverables:** Documented in [`progress.md`](progress.md#week-6-completed).

---

## Week 7 (Upcoming)
* **Objective:** Build the background and research plan for studying robust overfitting under frequency-restricted adversarial perturbations.

  1. Learn the basics of the Discrete Cosine Transform (DCT), Fast Fourier Transform (FFT), image-frequency bands, and JPEG compression. Distinguish between spatial-domain perturbations, which change pixel values directly, and frequency-domain perturbations, which change frequency coefficients.
  2. Find and study research papers on frequency-domain adversarial attacks and adversarial training. Create reading notes that identify what each paper has already studied and how it relates to our project.
  3. Determine whether prior work has studied robust overfitting separately for low-, middle-, and high-frequency perturbations. Use this review to define the project's exact research gap, revise the research question and hypothesis, and choose an appropriate transform for implementation.

* **Expectations:** A clearly defined and properly cited research direction, including a revised question and hypothesis, a documented research gap, and a decision about the frequency-domain representation to use.

---

## Week 8 (Upcoming)
* **Objective:** Implement frequency-restricted adversarial training and verify that the new pipeline works correctly before full experiments.

  1. Modify the training and evaluation scripts so that the adversarial attack changes only selected frequency coefficients instead of directly changing pixel values. Support separate low-, middle-, and high-frequency bands.
  2. Define a reproducible experimental configuration for each band, including the frequency-band masks, perturbation budget, random seed, number of training epochs, saved checkpoints, and evaluation settings. Keep the existing pixel-space PGD result as the baseline for comparison.
  3. Run a short local diagnostic to verify the forward and inverse transforms, confirm that only the intended frequency band is changed, and ensure that training and evaluation finish without errors.
  4. Run a short diagnostic experiment on Lambda Labs to check runtime, GPU behavior, checkpoint saving, and result logging before starting the full experiments.

* **Expectations:** Verified frequency-domain training and evaluation scripts, documented experiment settings, and successful local and Lambda Labs diagnostic runs.

---

## Week 9 (Upcoming)
* **Objective:** Run the full frequency-band experiments and organize the resulting data for analysis.

  1. Run full adversarial-training experiments on Lambda Labs using low-, middle-, and high-frequency perturbations, following the configuration established in Week 8.
  2. Monitor each run for errors or unexpected behavior, and save checkpoints, training logs, evaluation CSV files, and plots in organized locations.
  3. Evaluate all saved checkpoints using the matching frequency-restricted attack so robust accuracy can be compared consistently across training epochs and frequency bands.

* **Expectations:** A complete, organized set of experimental results for the low-, middle-, and high-frequency conditions, ready for analysis.

---

## Week 10 (Upcoming)
* **Objective:** Analyze whether robust overfitting differs by frequency band and begin communicating the findings.

  1. Compare clean and robust accuracy curves across the pixel-space baseline and the low-, middle-, and high-frequency experiments. Identify each condition's peak robust accuracy and determine whether robustness declines with further training.
  2. Investigate unclear, inconsistent, or unexpected results. Refine frequency-band definitions or rerun selected experiments only when needed to support a reliable conclusion.
  3. Begin drafting the final report and consortium presentation slides, including the research question, experimental method, results, and the project's relationship to prior work.

* **Expectations:** A defensible interpretation of the results, supported by plots and metrics, plus initial drafts of the report and consortium slides.

---

## Week 11 (Upcoming)
* **Objective:** Complete and verify the final report and consortium presentation.

  1. Finish the final report, clearly describing the motivation, related work, frequency-domain methodology, experimental results, limitations, and conclusions.
  2. Complete the consortium presentation slides, making sure they communicate the research gap, method, results, and contribution in a clear sequence.
  3. Review the report, code, figures, and slides for accuracy and reproducibility. Incorporate feedback and rehearse the presentation.

* **Expectations:** A polished, verified final report, reproducible supporting materials, and presentation-ready consortium slides.

*Note: Because the research schedule has been extended through November, providing roughly two additional months for the project, three weeks are intentionally left as buffer time in case an experiment needs to be rerun, unexpected problems arise, or additional time is needed to complete the work.*
