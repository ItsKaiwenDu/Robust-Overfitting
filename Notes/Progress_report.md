# Research Progress Reports

## Week 1 (Completed)
* **Progress Report:** During week 1, I focused on digesting foundational readings for our research project. The first paper I read is "Explaining and Harnessing Adversarial Examples" by Goodfellow et al. I learned that adversarial examples are inputs with tiny, carefully calculated perturbations that are invisible to human eye yet cause neural networks to misclassify with high confidence. Contrary to common belief that complex models are tricked by being overly sensitive to fine details, paper argues opposite: adversarial examples exist because modern networks are designed to behave linearly, and in high-dimensional input spaces (such as images with tens of thousands of pixels), these linear responses let tiny per-pixel changes accumulate into a large shift in model's output. The Fast Gradient Sign Method (FGSM) formalizes this by computing gradient of model's loss with respect to input and shifting each pixel by $\epsilon$ in direction that increases model's error most. As a defense, paper proposes adversarial training, which mixes adversarially perturbed examples directly into training so model learns to handle attacks during training, dropping adversarial error on MNIST from 89.4% down to 17.9%. The paper also showed that common defenses such as dropout, pretraining, and model averaging are largely ineffective against these attacks. The second paper I read is "Overfitting in Adversarially Robust Deep Learning" by Rice et al. I learned that robust overfitting is a specific failure mode in adversarial training where a model's test robustness degrades over continued training even as its training loss keeps improving, and that this phenomenon appears consistently across multiple datasets including SVHN, CIFAR-10, CIFAR-100, and ImageNet. The paper tests a full range of methods to fix this: explicit regularization (L1 and L2 weight penalties), data augmentation (cutout and mixup), and semi-supervised learning on unlabeled data. None of these fully solved problem on their own. The simplest and most effective fix is early stopping, which saves model checkpoint at point of best validation robustness before test performance starts to degrade. PGD adversarial training with early stopping matches or exceeds TRADES, a far more complex method. The only approach that genuinely outperformed early stopping alone was combining it with semi-supervised learning. During Friday meeting with Dr. Tran, he suggested I condense my notes, which were very detailed, into presentation slides for better communication, and to focus specifically on early stopping as our chosen method given research time constraints.

* **Deliverables:**
  * [`Goodfellow.md`](Goodfellow.md): Covers linearity hypothesis, FGSM, adversarial training results on MNIST, and connections to our research.
  * [`Rice.md`](Rice.md): Covers robust overfitting phenomenon, effect of early stopping, comparison of regularization methods, and connections to our research.

---

## Week 2 (Completed)
* **Progress Report:** This week, I focused on setting up development environment and organizing repository files. I successfully configured a Lambda Labs cloud account accessing NVIDIA A10 GPU for training AI model in upcoming weeks. I also listed required Python package dependencies in [`requirements.txt`](../requirements.txt) and created a script in [`verify_setup.py`](../verify_setup.py) to verify that everything works properly: version, imports, and system device capabilities. Finally, I downloaded and integrated PreAct ResNet-18 model architecture (a standard neural network architecture used in adversarial training for its improved training stability and performance) from Rice et al. paper into project models folder for replicating robust overfitting experiments. I also created an initial set of presentation slides summarizing literature. During Friday meeting with Dr. Tran, he reviewed slides and said they were essentially a collection of random ideas with no clear storyline or logical flow. After meeting, I completely overhauled presentation from scratch, restructuring it around a natural storyline: foundational definitions and adversarial problem, how adversarial examples are generated via FGSM, analyzing why they exist, how we can defend against them, new problem of robust overfitting that emerges during adversarial training, evidence showing it is real and consequential, early stopping as best solution, its effectiveness relative to other regularization methods, our research proposal, our summer findings (placeholder for now; TBD after research findings are complete), and a summary with key takeaways and references. We also laid out plan for next week in [`README.md`](../README.md), which covers writing a training script that generates adversarial examples using PGD and trains model to resist them, then running a short local test to verify everything works correctly before moving to cloud.

* **Deliverables:**
  * [`.gitignore`](../.gitignore): Configured to ignore environment folders, datasets, cache files, and model checkpoints.
  * [`requirements.txt`](../requirements.txt): Lists project dependencies (`torch`, `torchvision`, `numpy`, `matplotlib`, `tensorboard`).
  * [`Models/preact_resnet.py`](../Models/preact_resnet.py): PyTorch implementation of `PreActResNet18` sourced from Rice et al. 2020 codebase, tailored for CIFAR-10, for purpose of exact experimental replication.
  * [`verify_setup.py`](../verify_setup.py): Script to verify python imports, system device capabilities (CPU/MPS/CUDA), and run a forward pass sanity check.
  * [`cloud_setup.md`](../cloud_setup.md): Guide for deploying runs to cloud instances (Lambda Labs), ssh configurations, code syncing, `tmux` sessions, and TensorBoard port forwarding.
  * [`Presentation.pdf`](Presentation.pdf): Presentation slides covering foundational definitions and FGSM, adversarial examples, adversarial training, robust overfitting, early stopping, and research proposal.

---

## Week 3 (Completed)
* **Progress Report:** This week, I built the training script [`train.py`](../train.py) based on Rice et al., which contains a PGD adversarial image generator (takes a clean image, adds a small random noise, then runs 10 steps of edits that change pixels in the direction that will increase the model's error the most, and outputs final adversarial image), a training loop that runs those adversarial images through PreActResNet-18, slows the learning rate down at epochs 100 and 150 (with `MultiStepLR`) so the model fine-tunes gradually, and saves model weights checkpoint every 5 epochs. We also added a `Normalizer` step that rescales pixel brightness values to a consistent range before every forward pass, since Rice et al.'s original code uses CUDA software and it only supports NVIDIA GPU, which would be incompatible on a MacBook. Finally, we added a diagnostic mode (`--diagnostic`) that runs 1 epoch on 10% of the data locally to verify that the full pipeline works before starting a full training run on Lambda Labs.

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
  Checkpoint saved: Checkpoints/diagnostic/epoch_1.pt
  Training pipeline finished.
  ```

* **Note on FGSM vs. PGD:** Both methods share the same core idea: use the model's own gradient on the input image to find which pixel changes maximize the model's error. The difference is how many times they do it. FGSM does it once, shifting every pixel by the full epsilon in one shot. PGD does it 10 times in small steps (as in Rice et al.), re-checking the gradient from its new position each time to find a stronger worst-case image within the epsilon ball (which is the allowed perturbation region that keeps all pixel changes within +-8/255, keeping the noise invisible to human eyes). PGD is better than FGSM because it explores many directions within the epsilon ball across multiple steps, finding a stronger worst-case adversarial image that builds genuine, broad robustness in the model, whereas FGSM only ever attacks from one fixed gradient direction, so a model trained against it only learns to handle that one specific perturbation pattern and stays vulnerable to stronger attacks. Additionally, Rice et al. uses PGD because robust overfitting is most clearly documented and visible in PGD-trained networks, making it the right baseline for observing and studying the phenomenon.

* **Deliverables:**
  * [`train.py`](../train.py): Main PyTorch training script implementing the PGD attack, training loop, scheduler, and diagnostic setup.
  * [`Checkpoints/diagnostic/epoch_1.pt`](../Checkpoints/diagnostic/epoch_1.pt): Model checkpoint saved from the local diagnostic test to verify weight saving.
  * [`.gitignore`](../.gitignore): Configured to ignore the raw dataset folder to keep GitHub commits clean.

---

## Week 4 (In Progress)
* **Progress Report:** This week, I focused on better understanding the Week 3 work with [`train.py`](../train.py), and then I reworked the progress report into a single, concise paragraph. I also digested information and tips on "How To Give Strong Technical Presentations" and also watched popular YouTube videos on public speaking to prepare for presentation. Based on that, I reworked and polished the presentation slides, shifting from slides that were mostly teleprompters into slides that use figures and key talking points, and details are instead spoken out loud from speaking notes I made. Finally, I also adjusted my upcoming weekly schedules in [README.md](../README.md). This Friday, after one-on-one meeting with Dr. Tran, I plan to start full training on Lambda Labs to stay on schedule and avoid time pressure at the end. The run is expected to take 7-10 hours and requires minimal human work: deploying `train.py`, monitoring for any unexpected timeouts or crashes, and downloading all model checkpoints once training completes.

* **Deliverables:**
  * [`Presentation.pdf`](Presentation.pdf): Polished presentation slides using figures and key talking points, with details spoken out loud from speaking notes.