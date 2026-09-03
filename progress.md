# Research Progress Reports

## Week 1 (Completed)
* **Progress Report:** During week 1, I focused on digesting foundational readings for our research project. The first paper I read is "Explaining and Harnessing Adversarial Examples" by Goodfellow et al. I learned that adversarial examples are inputs with tiny, carefully calculated perturbations that are invisible to human eye yet cause neural networks to misclassify with high confidence. Contrary to common belief that complex models are tricked by being overly sensitive to fine details, paper argues opposite: adversarial examples exist because modern networks are designed to behave linearly, and in high-dimensional input spaces (such as images with tens of thousands of pixels), these linear responses let small per-pixel changes accumulate into a large shift in model's output. The Fast Gradient Sign Method (FGSM) formalizes this by computing gradient of model's loss with respect to input and shifting each pixel by $\epsilon$ in direction that increases model's error most. As a defense, paper proposes adversarial training, which mixes adversarially perturbed examples directly into training so model learns to handle attacks during training, dropping adversarial error on MNIST from 89.4% down to 17.9%. The paper also showed that common defenses such as dropout, pretraining, and model averaging are largely ineffective against these attacks. The second paper I read is "Overfitting in Adversarially Robust Deep Learning" by Rice et al. I learned that robust overfitting is a specific failure mode in adversarial training where a model's test robustness degrades over continued training even as its training loss keeps improving, and that this phenomenon appears consistently across multiple datasets including SVHN, CIFAR-10, CIFAR-100, and ImageNet. The paper tests a full range of methods to fix this: explicit regularization (L1 and L2 weight penalties), data augmentation (cutout and mixup), and semi-supervised learning on unlabeled data. None of these fully solved problem on their own. The simplest and most effective fix is early stopping, which saves model checkpoint at point of best validation robustness before test performance starts to degrade. PGD adversarial training with early stopping matches or exceeds TRADES, a far more complex method. The only approach that genuinely outperformed early stopping alone was combining it with semi-supervised learning. During Friday meeting with Dr. Tran, he suggested I condense my notes, which were very detailed, into presentation slides for better communication, and to focus specifically on early stopping as our chosen method given research time constraints.

* **Deliverables:**
  * [`goodfellow.md`](../notes/goodfellow.md): Covers linearity hypothesis, FGSM, adversarial training results on MNIST, and connections to our research.
  * [`rice.md`](../notes/rice.md): Covers robust overfitting phenomenon, effect of early stopping, comparison of regularization methods, and connections to our research.

---

## Week 2 (Completed)
* **Progress Report:** This week, I focused on setting up development environment and organizing repository files. I successfully configured a Lambda Labs cloud account accessing NVIDIA A10 GPU for training AI model in upcoming weeks. I also listed required Python package dependencies in [`requirements.txt`](../requirements.txt) and created a script in [`verify_setup.py`](../scripts/verify_setup.py) to verify that everything works properly: version, imports, and system device capabilities. Finally, I downloaded and integrated PreAct ResNet-18 model architecture (a standard neural network architecture used in adversarial training for its improved training stability and performance) from Rice et al. paper into project models folder for replicating robust overfitting experiments. I also created an initial set of presentation slides summarizing literature. During Friday meeting with Dr. Tran, he reviewed slides and said they were essentially a collection of random ideas with no clear storyline or logical flow. After meeting, I completely overhauled presentation from scratch, restructuring it around a natural storyline: foundational definitions and adversarial problem, how adversarial examples are generated via FGSM, analyzing why they exist, how we can defend against them, new problem of robust overfitting that emerges during adversarial training, evidence showing it is real and consequential, early stopping as best solution, its effectiveness relative to other regularization methods, our research proposal, our summer findings (placeholder for now; TBD after research findings are complete), and a summary with key takeaways and references. We also laid out plan for next week in [`README.md`](../README.md), which covers writing a training script that generates adversarial examples using PGD and trains model to resist them, then running a short local test to verify everything works correctly before moving to cloud.

* **Deliverables:**
  * [`.gitignore`](../.gitignore): Configured to ignore environment folders, datasets, cache files, and model checkpoints.
  * [`requirements.txt`](../requirements.txt): Lists project dependencies (`torch`, `torchvision`, `numpy`, `matplotlib`, `tensorboard`).
  * [`models/preact_resnet.py`](../models/preact_resnet.py): PyTorch implementation of `PreActResNet18` sourced from Rice et al. 2020 codebase, tailored for CIFAR-10, to replicate their training setup.
  * [`scripts/verify_setup.py`](../scripts/verify_setup.py): Script to verify python imports, system device capabilities (CPU/MPS/CUDA), and run a forward pass sanity check.
  * [`setup_lambda_labs.md`](../setup_lambda_labs.md): Guide for deploying runs to cloud instances (Lambda Labs), ssh configurations, code syncing, `tmux` sessions, and TensorBoard port forwarding.
  * [`presentation.pdf`](presentation.pdf): Presentation slides covering foundational definitions and FGSM, adversarial examples, adversarial training, robust overfitting, early stopping, and research proposal.

---

## Week 3 (Completed)
* **Progress Report:** This week, I built training script [`scripts/train.py`](../scripts/train.py) based on Rice et al., which uses CIFAR-10 dataset and contains a PGD adversarial image generator (takes a clean image, adds a small random noise, then runs 10 steps of edits that change pixels in direction that will increase model's error most, and outputs final adversarial image), a training loop that runs those adversarial images through PreActResNet-18, slows learning rate down at epochs 100 and 150 (with `MultiStepLR`) so model fine-tunes gradually, and saves model weights checkpoint every 5 epochs. We also added a `Normalizer` step that rescales pixel brightness values to a consistent range before every forward pass, since Rice et al.'s original code uses CUDA software and it only supports NVIDIA GPU, which would be incompatible on a MacBook. Finally, we added a diagnostic mode (`--diagnostic`) that runs 1 epoch on 10% of CIFAR-10 data locally to verify that full pipeline works before starting a full training run on Lambda Labs.

* **Diagnostic Output Log:**
  ```
  Mode: Diagnostic
  Diagnostic training subset size: 5000
  Diagnostic test subset size: 1000
  Starting training pipeline: 1 epochs.
  Hyperparameters: LR=0.1, Decay Epochs=[100, 150], Weight Decay=0.0005, Momentum=0.9
  PGD Attack config: epsilon=0.0314, alpha=0.0078, steps=10
  Epoch 001 | Train Loss: 2.2761 | Train Clean Acc: 20.46% | Train Robust Acc: 16.24% | Time: 41.66s
  --> Evaluation: Test Clean Acc: 16.50% | Test Robust Acc: 10.70% | Time: 28.94s
  Checkpoint saved: checkpoints/diagnostic/epoch_1.pt
  Training pipeline finished.
  ```

* **Note on FGSM vs. PGD:** Both methods share same core idea: use model's own gradient on input image to find which pixel changes maximize model's error. The difference is how many times they do it. FGSM does it once, shifting every pixel by full epsilon in one shot. PGD does it 10 times in small steps (as in Rice et al.), re-checking gradient from its new position each time to find a stronger worst-case image within epsilon ball (which is allowed perturbation region that keeps all pixel changes within +-8/255, keeping noise invisible to human eyes). PGD is better than FGSM because it explores many directions within epsilon ball across multiple steps, finding a stronger worst-case adversarial image that builds genuine, broad robustness in model, whereas FGSM only ever attacks from one fixed gradient direction, so a model trained against it only learns to handle that one specific perturbation pattern and stays vulnerable to stronger attacks. Additionally, Rice et al. uses PGD because robust overfitting is most clearly documented and visible in PGD-trained networks, making it right baseline for observing and studying phenomenon.

* **Deliverables:**
  * [`scripts/train.py`](../scripts/train.py): Main PyTorch training script implementing PGD attack, training loop, scheduler, and diagnostic setup.
  * [`checkpoints/diagnostic/epoch_1.pt`](../checkpoints/diagnostic/epoch_1.pt): Model checkpoint saved from local diagnostic test to verify weight saving.
  * [`.gitignore`](../.gitignore): Configured to ignore raw dataset folder to keep GitHub commits clean.

---

## Week 4 (Completed)
* **Progress Report:** This week, I focused on better understanding Week 3 work with [`scripts/train.py`](../scripts/train.py) on CIFAR-10, and then I reworked progress report into a single, concise paragraph. I also digested information and tips on "How To Give Strong Technical Presentations" and also watched popular YouTube videos on public speaking to prepare for presentation. Based on that, I reworked and polished presentation slides, shifting from slides that were mostly teleprompters into slides that use figures and key talking points, and details are instead spoken out loud from speaking notes I made. Finally, I also adjusted my upcoming weekly schedules in [README.md](../README.md). This Friday, after one-on-one meeting with Dr. Tran, I plan to start full CIFAR-10 training on Lambda Labs to stay on schedule and avoid time pressure at end. The run is expected to take 7-10 hours and requires minimal human work: deploying `scripts/train.py`, monitoring for any unexpected timeouts or crashes, and downloading all model checkpoints once training completes.

* **Deliverables:**
  * [`presentation.pdf`](presentation.pdf): Polished presentation slides using figures and key talking points, with details spoken out loud from speaking notes.

---

## Week 5 (Completed)
* **Progress Report:** Last Friday, I launched full 200-epoch PGD-10 adversarial training run on a 1x NVIDIA A10 GPU instance on Lambda Labs (~2.5 mins per epoch, ~8 hours total). The run completed all 200 epochs with clear robust overfitting observed. All 40 checkpoints (~3.57 GB) were downloaded locally, excluded from version control via `.gitignore`, and uploaded to HuggingFace for public access. This week, I created [`scripts/evaluate.py`](../scripts/evaluate.py) to run PGD-20 adversarial evaluation across all 40 checkpoints on CIFAR-10 test set and launched it on Lambda Labs again, confirming robust overfitting as described by Rice et al. (2020). Evaluation results were saved to [`evaluation_results.csv`](evaluation_results.csv) and visualized in [`robust_overfitting_curves.png`](robust_overfitting_curves.png) using [`scripts/plot_results.py`](../scripts/plot_results.py). Finally, I have polished slides for clarifications, organization, and added robust overfitting curve graph to there as well. I am also getting familiar with Overleaf, including LaTeX formats, which will be used to write our final report during Week 8.

  **Training results (PGD-10):**
  * Test robust accuracy: peaked at **47.85%** at Epoch 105, declined to **41.26%** by Epoch 200
  * Train robust accuracy: **89.71%** at Epoch 200

  **Evaluation results (PGD-20):**
  * Test robust accuracy: peaked at **45.91%** at Epoch 105, dropped to **36.18%** by Epoch 200 (−**9.73 percentage points**)
  * Clean accuracy at Epoch 105: **81.94%**
  * Robust loss: **1.4761** at Epoch 105 → **3.7487** at Epoch 200

* **Deliverables:**
  * [`.gitignore`](../.gitignore): Excludes large files (datasets, checkpoints) from version control.
  * [Model checkpoints](https://huggingface.co/KaiwenDu/robust-overfitting-checkpoints/tree/main): All 40 checkpoints (~3.57 GB) hosted on HuggingFace.
  * [`scripts/evaluate.py`](../scripts/evaluate.py): Evaluates all 40 checkpoints using PGD-20.
  * [`evaluation_results.csv`](evaluation_results.csv): Accuracy and loss metrics for all 40 checkpoints.
  * [`scripts/plot_results.py`](../scripts/plot_results.py): Plots clean and robust accuracy curves.
  * [`robust_overfitting_curves.png`](report/robust_overfitting_curves.png): Chart showing robust overfitting with peak at Epoch 105.
  * [`training_results_curves.png`](report/training_results_curves.png): Detailed training and evaluation accuracy/loss curves.

---

## Week 6 (Completed)
* **Progress Report:** This week, I learned more about Cross-Epoch Transfer Matrices and found that they can show whether neural networks retain or transfer adversarial perturbations between training checkpoints. However, this technique would mainly demonstrate that network memorizes perturbations at different points in training; it would not directly explain why this memorization occurs. I also revisited our research question and hypothesis and found that central ideas have already been addressed in several published papers, including Dong et al. (ICLR 2022), Yu et al. (ICML 2022), and Liu et al. (JMLR 2024). This suggests that our current question may be too broad and that a direct replication would not be sufficiently novel. Finally, I considered Dr. Tran's proposal to turn project into a case study on advantages and limitations of using AI tools to conduct research about AI, and I prepared key points to discuss during Friday's meeting so we can decide on project's next direction.

* **Deliverables:**
  * Discuss project's next steps with Dr. Tran during Friday's meeting.

---

## Week 7 (Completed)
* **Progress Report:** This week, I studied Discrete Cosine Transform (DCT). Then I found and reviewed six relevant research papers on frequency-domain adversarial attacks and training, and wrote notes in [`notes/`](notes/) covering frequency-domain attacks, frequency-aware adversarial training, frequency bias, and proposed mechanisms for robust overfitting. Three papers were more directly useful for our project: Chen et al. support frequency-masked PGD perturbations, Guo et al. provide DCT transform-mask-inverse-transform pattern, and Yu et al. offer a possible explanation for robust overfitting. The other three provide supporting background on how frequency can affect robust learning. After that, I refined project's research question and hypothesis to test whether selected perturbation band changes timing (peak robust-accuracy epoch) and severity (post-peak decline) of robust overfitting during PGD adversarial training on CIFAR-10. Lastly, [`goals.md`](goals.md) was updated to align with this refined direction.

* **Deliverables:**
  * [`README.md`](README.md): Updated research question and hypothesis for frequency-band study.
  * [`goals.md`](goals.md): Updated upcoming weekly goals.
  * [`chen_et_al.md`](notes/chen_et_al.md): Frequency-masked PGD perturbations and frequency contributions to predictions.
  * [`guo_et_al.md`](notes/guo_et_al.md): Low-frequency DCT perturbations; basis for frequency-restricted attacks.
  * [`yu_et_al.md`](notes/yu_et_al.md): Small-loss adversarial examples as a possible cause of robust overfitting.
  * [`bu_et_al.md`](notes/bu_et_al.md) *(background info)*: Low-frequency feature bias for adversarial robustness.
  * [`kim_et_al.md`](notes/kim_et_al.md) *(background info)*: Frequency principle and learning behavior in adversarial training.
  * [`li_et_al.md`](notes/li_et_al.md) *(background info)*: Fourier amplitude and phase in adversarial robustness.

---

## Week 8 (Completed)
* **Progress Report:** This week, I read three papers Dr. Tran provided. Tramèr and Boneh (2019) show that robustness to one attack type can reduce robustness to another and compare AVG and MAX multi-attack training strategies. Maini, Wong, and Kolter (2020) extend this idea with Multi Steepest Descent (MSD), which finds the worst-case adversarial example across multiple attack types at each step and emphasizes evaluating a mixed model against each component attack separately. Xie, He, and Fang (2026) show that attacks can concentrate energy in different frequency ranges and propose an architecture that routes examples to specialized components based on the predicted attack type. After reading these papers, I refined our research question and hypothesis and outlined the three training conditions and evaluation plan. Finally, I implemented the low-frequency DCT-masked PGD attack in [`scripts/dct_pgd.py`](scripts/dct_pgd.py), which restricts adversarial perturbations to low-frequency image content by operating in the frequency domain rather than directly on raw pixels. Next week, I will integrate this attack into the training pipeline, add evaluation support, and run diagnostics locally and on Lambda Labs.

* **Deliverables:**
  * [`tramer_et_al.md`](notes/tramer_et_al.md): Multi-perturbation robustness trade-offs; AVG/MAX training; union vs. average evaluation.
  * [`maini_et_al.md`](notes/maini_et_al.md): Multi Steepest Descent (MSD); per-checkpoint tracking of clean, pixel-space, low-frequency, and union robust accuracy.
  * [`xie_et_al.md`](notes/xie_et_al.md): Threat-Aware Frequency Decoupling (TaFD); separable attack frequency spectra; negative transfer in joint adversarial training.
  * [`scripts/dct_pgd.py`](scripts/dct_pgd.py): Reusable DCT, low-frequency mask/projection, DCT-masked PGD, and leakage-metadata implementation.

---

## Week 9 (Completed)
* **Progress Report:** This week, I connected low-frequency DCT-masked PGD attack to our training ([`scripts/train.py`](scripts/train.py)) and evaluation ([`scripts/evaluate.py`](scripts/evaluate.py)) pipeline. The training script now supports pixel-only, low-frequency-only, and mixed-domain training. In mixed-domain mode, we make a seeded 50/50 choice between pixel-space and low-frequency attacks at start of each epoch, then save selected attack domain, complete schedule, seed, and DCT cutoff with every checkpoint. I also extended evaluation script to test each checkpoint on clean images, pixel-space PGD-20, low-frequency DCT-masked PGD-20, and union robustness (where an image must resist both domain attacks to count as correct). When finished, everything will still be saved in [`report`](report/), but organized into separate appropriate folders inside. Finally, I ran short diagnostics for all 3 modes (pixel-only, low-frequency-only, mixed-domains) locally and on Lambda Labs; all successful with no unexpected behavior.

* **Deliverables:**
  * [`scripts/train.py`](scripts/train.py): Supports pixel-only, low-frequency-only, and seeded mixed-domain adversarial training with condition-and-seed-specific output directories.
  * [`scripts/evaluate.py`](scripts/evaluate.py): Reports clean, pixel-PGD, low-frequency-PGD, and union robustness for every checkpoint.
  * [`scripts/plot_results.py`](scripts/plot_results.py): Reads the expanded evaluation CSV format and saves plots in the matching condition-and-seed directory.
  * [`README.md`](README.md): Documents the three modes, output-directory layout, commands, and updated project structure.
  * [`goals.md`](goals.md): Updated schedule for 10 low-frequency-only runs, 10 pixel-only runs, 10 mixed-domain runs, and the later full comparison.
  * Local and Lambda Labs diagnostic checkpoints, TensorBoard logs, and four-metric evaluation CSV files for all three training modes.

---

## Week 10 (Completed)
* **Progress Report:** This week, I ran 5 full 200-epoch low-frequency-only adversarial-training experiments on Lambda Labs using seeds 42 through 46, saved 40 checkpoints and TensorBoard logs for each run, and evaluated every checkpoint on clean images, pixel-space PGD-20, low-frequency DCT-masked PGD-20, and their per-image union. The results show that the models were consistently robust to the low-frequency attacks they were trained against: peak low-frequency robust accuracy ranged from 84.22% to 85.32% across the 5 seeds, and the five-seed average at epoch 200 was 84.50%. In contrast, pixel-space and union robust accuracy never exceeded 0.02%, so the models remained almost completely vulnerable to unrestricted pixel-space attacks; low-frequency robustness also remained largely stable late in training, with only small end-of-training declines in two seeds. This matches my prediction: training against the restricted, smooth low-frequency perturbations produced strong low-frequency robustness but did not transfer to the broader pixel-space threat model, which also caused union robustness to remain near zero.

* **Deliverables:**
  * [`checkpoints/low-frequency-only/`](checkpoints/low-frequency-only/): 5 seed-specific folders containing 40 saved model checkpoints each, from epochs 5 through 200.
  * [`runs/low-frequency-only/`](runs/low-frequency-only/): TensorBoard event logs for the 5 full low-frequency-only training runs.
  * [`report/low-frequency-only/`](report/low-frequency-only/): Seed-specific evaluation CSVs, evaluation figures (`robust_overfitting_curves.png`), and training dynamics figures (`training_results_curves.png`) across all 5 seeds.
  * [`report/low-frequency-only/overall/`](report/low-frequency-only/overall/): 5-seed overall evaluation curves, training dynamics curves with mean $\pm 1\text{ SD}$ error bands, and summary CSV (`evaluation_results.csv`).
  * [`scripts/evaluate.py`](scripts/evaluate.py): Used to evaluate all saved checkpoints under clean, pixel-PGD-20, low-frequency-PGD-20, and union conditions.
  * [`scripts/plot_results.py`](scripts/plot_results.py): Unified plotting script generating per-seed and aggregated evaluation and training curves.
  * [`goals.md`](goals.md): Week 10 status updated to Completed after all 5 runs and evaluations finished.
