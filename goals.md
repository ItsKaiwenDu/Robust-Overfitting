# Weekly Goals & Objectives

This document tracks weekly goals, objectives, expectations, and deliverables for research project on investigating robust overfitting in adversarial training.

---

## Week 1 (Completed)
* **Objective:** Review and summarize foundational literature on adversarial training and robust overfitting.

  1. Read and digest "Explaining and Harnessing Adversarial Examples" by Goodfellow et al. (2014). Learn definitions of adversarial examples/attacks, how they are generated via Fast Gradient Sign Method (FGSM), analyzing adversarial examples, and learn about how to defend these adversarial attacks.
  2. Read and digest "Overfitting in Adversarially Robust Deep Learning" by Rice et al. (2020). Learn about robust overfitting during adversarial training, and learn what are some methods to stop/reduce robust overfitting.
  3. Compile detailed reading notes for ourselves, which will be saved to [`Notes`](Notes) folder.

* **Expectations:** A solid theoretical understanding of adversarial examples and early stopping.

* **Progress Report & Deliverables:** Documented in [`progress.md`](progress.md#week-1-completed).

---

## Week 2 (Completed)
* **Objective:** Configure local development environment, set up cloud compute resources, download PreActResNet-18 model architecture, verify setup with a verification script, and create presentation slides based on feedback.

  1. Configure required Python virtual environment, set up gitignore, and specify dependencies in `requirements.txt`.
  2. Create a Lambda Labs cloud account and set up billing to access NVIDIA A10G GPU instances for Week 4 training run.
  3. Download PreActResNet-18 model architecture in PyTorch (`models/preact_resnet.py`) from Rice et al. 2020 codebase so our training setup matches theirs.
  4. Write a setup verification script (`scripts/verify_setup.py`) to verify package imports, check hardware/device availability (such as CUDA/MPS/CPU), and run a forward pass sanity check with model to ensure training environment is ready to go.
  5. Create presentation slides summarizing literature review with visual and mathematical explanations of adversarial attacks. These slides communicate our foundational understanding to PI and will be reviewed and refined during Friday meeting with Dr. Tran.

* **Expectations:** A fully operational local and cloud training environment, model implementation complete, and a completed presentation (with placeholders reserved for our own research findings) with a clear and logical flow.

* **Progress Report & Deliverables:** Documented in [`progress.md`](progress.md#week-2-completed).

---

## Week 3 (Completed)
* **Objective:** Implement Projected Gradient Descent (PGD)-based adversarial training pipeline and verify its correctness on a diagnostic test run. Lambda Labs will not be used yet this week; all local.

  1. Write training script [`scripts/train.py`](scripts/train.py) using [`PreActResNet18`](models/preact_resnet.py#L153) architecture. Script should dynamically generate adversarial images using a 10-step PGD attack, which starts with random noise and makes pixel adjustments within a safety range (the $\epsilon$-ball of $8/255$) so edits remain invisible to human eyes. All of these should match exactly as in Rice et al. paper.
  2. Make [`scripts/train.py`](scripts/train.py) save model weights every 5 epochs to `checkpoints/` folder. This allows us to analyze model's performance at different stages of training and pinpoint exactly where robust overfitting begins.
  3. Make and run diagnostic test locally (for 1 epoch on 10% of CIFAR-10 data) to ensure training loop, PGD attack, weight updates, checkpoints, and logging all work without complications before full training on Lambda Labs.

* **Expectations:** A fully functional and verified training script (`scripts/train.py`), ready for full training runs on Lambda Labs next week.

* **Progress Report & Deliverables:** Documented in [`progress.md`](progress.md#week-3-completed).

---

## Week 4 (Completed)
* **Objective:** Review Week 3 training pipeline, improve speaking skills and polish presentation slides, and run full training on Lambda Labs on Friday.

  1. Review [`scripts/train.py`](scripts/train.py) alongside Week 3 progress report to make sure full pipeline is understood.
  2. Read "How To Give Strong Technical Presentations" provided by Dr. Tran and watch YouTube videos on public speaking practices.
  3. Polish presentation slides, from teleprompter to figures and key talking points, with details spoken out loud from speaking notes.
  4. Adjust upcoming weekly schedules.
  5. Deploy `scripts/train.py` to Lambda Labs and start full 200-epoch PGD adversarial training run, monitor for any unexpected timeouts or crashes, and download all model checkpoints once training completes (expected run time: 7-10 hours).

* **Expectations:** A complete set of 40 model checkpoints in `checkpoints/` directory covering full 200-epoch run (saved every 5 epochs), ready for evaluation.

* **Progress Report & Deliverables:** Documented in [`progress.md`](progress.md#week-4-completed).

---

## Week 5 (Completed)
* **Objective:** Evaluate model robustness across all saved checkpoints to identify robust overfitting point.

  1. Create `scripts/evaluate.py` to load each of 40 Week 4 checkpoints and test them using PGD-20 on CIFAR-10 test set. PGD-20 uses 20 attack steps instead of 10 used during training, making it a harder and more reliable measure of adversarial robustness.
  2. Record clean accuracy, robust accuracy, and loss for each checkpoint to a CSV file.
  3. Plot training and test robust accuracy curves across all 200 epochs using `matplotlib` to identify exact epoch where test robust accuracy peaks and begins to decline.

* **Expectations:** A completed test evaluation, a populated results CSV, and a clear plot showing peak robust accuracy epoch.

* **Progress Report & Deliverables:** Documented in [`progress.md`](progress.md#week-5-completed).

---

## Week 6 (Completed)
* **Objective:** Assess promising directions for project by studying Cross-Epoch Transfer Matrices, reviewing novelty of our research question, and considering a possible case-study approach.

  1. Learn how a Cross-Epoch Transfer Matrix can compare adversarial perturbations across model checkpoints, and identify what this analysis can and cannot explain about robust overfitting.
  2. Revisit our research question and hypothesis, then review related published work to determine whether main question has already been answered and whether our proposed contribution is sufficiently specific and novel.
  3. Explore Dr. Tran's proposal to recast project as a case study on benefits and limitations of using AI tools to conduct research about AI.

* **Expectations:** A clearer understanding of limits of Cross-Epoch Transfer Matrix analysis, an informed assessment of project's research novelty, and a set of discussion points for selecting next direction with Dr. Tran.

* **Progress Report & Deliverables:** Documented in [`progress.md`](progress.md#week-6-completed).

---

## Week 7 (Completed)
* **Objective:** Build background and research plan for studying robust overfitting under frequency-restricted adversarial perturbations.

  1. Learn basics of Discrete Cosine Transform (DCT), Fast Fourier Transform (FFT), image-frequency bands, and JPEG compression. Distinguish between spatial-domain perturbations, which change pixel values directly, and frequency-domain perturbations, which change frequency coefficients.
  2. Find and study research papers on frequency-domain adversarial attacks and adversarial training. Create reading notes that identify what each paper has already studied and how it relates to our project.
  3. Determine whether prior work has studied robust overfitting separately for low-, middle-, and high-frequency perturbations. Use this review to define project's exact research gap, revise research question and hypothesis, and choose an appropriate transform for implementation.

* **Expectations:** A clearly defined and properly cited research direction, including a revised question and hypothesis, a documented research gap, and a decision about frequency-domain representation to use.

---

## Week 8 (In Progress)
* **Objective:** Study multi-perturbation literature and implement low-frequency DCT-PGD attack.

  1. Read and digest Tramèr and Boneh (2019), Maini, Wong, and Kolter (2020), and Xie, He, and Fang (2026). Create notes that explain their methods, results, and relevance to this project.
  2. Distinguish project's randomized epoch-level attack schedule from AVG, MAX, MSD, and TaFD methods in literature.
  3. Finalize and document research question, hypothesis, training conditions, checkpoint evaluation protocol, and reproducibility requirements for pixel-only, low-frequency-only, and mixed-domain conditions.
  4. Implement low-frequency DCT-constrained PGD attack, including forward transform, low-frequency mask, inverse transform, and image-space perturbation constraint. 

* **Expectations:** Complete literature notes and citations, a clear experimental specification, and a working low-frequency DCT-PGD attack ready to integrate into training pipeline.

---

## Week 9 (Upcoming)
* **Objective:** Integrate DCT-PGD attack into pipeline and validate revised mixed-domain setup.

  1. Add three reproducible training modes: pixel-only, low-frequency-only, and mixed-domain. In mixed-domain mode, choose pixel-space or low-frequency PGD once at start of each epoch using a seeded fair random choice.
  2. Extend checkpoint evaluation to report clean accuracy, pixel-PGD robust accuracy, low-frequency-PGD robust accuracy, and a per-example worst-case union summary.
  3. Run short local and Lambda Labs diagnostics to verify DCT mask, epoch-level attack selection, checkpoint saving, evaluation metrics, runtime, GPU behavior, and result logging.

* **Expectations:** Verified training and evaluation scripts, documented run configuration, and successful local and cloud diagnostics.

---

## Week 10 (Upcoming)
* **Objective:** Run and analyze low-frequency-only baseline.

  1. Run full low-frequency-only adversarial-training experiment on Lambda Labs using Week 9 configuration.
  2. Save checkpoints, training logs, evaluation CSV files, attack-domain schedule metadata, and plots in `report/`.
  3. Evaluate every checkpoint under clean, pixel-PGD, low-frequency-PGD, and union conditions.
  4. Identify peak epoch and peak-to-final decline for every robust metric. Compare low-frequency-only curves with completed pixel-only baseline.

* **Expectations:** A complete low-frequency-only baseline with paired evaluation curves and a documented comparison against pixel-only training.

---

## Week 11 (Upcoming)
* **Objective:** Run and analyze mixed-domain experiment.

  1. Run full mixed-domain adversarial-training experiment on Lambda Labs using seeded epoch-level attack schedule.
  2. Save checkpoints, training logs, evaluation CSV files, chosen attack domain for every epoch, and plots in `report/`.
  3. Evaluate every checkpoint under clean, pixel-PGD, low-frequency-PGD, and union conditions.
  4. Compare mixed-domain curves with pixel-only and low-frequency-only baselines. Determine whether robust-accuracy peaks shift, flatten, or decline differently.

* **Expectations:** A complete mixed-domain result with reproducible attack scheduling and a cross-condition robust-overfitting comparison.

---

## Week 12 (Upcoming)
* **Objective:** Verify and interpret multi-threat results.

  1. Check that two attack evaluations are sufficiently strong and that union calculation uses worst result for each test image.
  2. Rerun selected diagnostics or full conditions only if a configuration, evaluation, or reproducibility problem is identified.
  3. Analyze trade-offs between pixel and low-frequency robustness, clean accuracy, peak epoch, and post-peak decline.
  4. Consider adversarial training-loss distributions by attack domain if robust-overfitting curves differ substantially.

* **Expectations:** Verified result quality and a defensible explanation of observed robust-overfitting behavior.

---

## Week 13 (Upcoming)
* **Objective:** Synthesize pixel-only, low-frequency-only, and mixed-domain findings and set up final report skeleton.

  1. Compare clean, pixel-PGD, low-frequency-PGD, and union robust-accuracy curves across all three training conditions.
  2. Summarize each condition's peak epoch, peak accuracy, and peak-to-final decline. Identify whether mixed-domain training shifted, flattened, or removed any robust-overfitting peak.
  3. Create final report skeleton, including motivation, related work, methodology, results, limitations, and conclusions.

* **Expectations:** A defensible interpretation of mixed-domain robust-overfitting study and a report structure ready for writing.

---

## Week 14 (Upcoming)
* **Objective:** Make progress on final report and consortium slides in parallel.

  1. Write sections of final report (motivation, related work, methodology, results, limitations, conclusions) and incorporate figures, tables, and citations.
  2. Build consortium presentation slides and write speaker notes, covering research gap, method, results, and contribution.

* **Expectations:** Meaningful progress on both report and slides.

---

## Week 15 (Upcoming)
* **Objective:** Continue and complete final report and consortium slides in parallel.

  1. Continue writing and finishing remaining sections of final report.
  2. Continue building and polishing consortium slides and speaker notes.

* **Expectations:** A complete written draft of final report and a complete set of consortium slides with speaker notes.

---

## Week 16 (Upcoming)
* **Objective:** Verify all materials and prepare for submission.

  1. Review repository and final report for accuracy and clarity. Fix any typos, unclear phrasing, or inconsistencies.
  2. Review consortium slides to ensure content matches final report, potential audience questions are addressed, and presentation is clear and easy to follow. Fix if needed.
  3. Confirm everything is submission-ready.

* **Expectations:** A verified, polished final report, clean repository, and presentation-ready consortium slides ready for submission.

*Note: Because research schedule has been extended through November, we intentionally left several unassigned weeks as a flexible buffer; if any week overruns, needs a rerun, or encounters unexpected problems, we can simply expand schedule into those weeks without disrupting overall timeline.*
