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
* **Progress Report:** This week, I focused on building, debugging, and locally verifying core Projected Gradient Descent (PGD)-based adversarial training pipeline. The work was divided into following days:
  * **Monday:** I started by implementing PGD adversarial perturbation generator. I wrote a PyTorch function to generate adversarial examples using a 10-step PGD attack with random initialization within $L_\infty$ $\epsilon$-ball (where $\epsilon = 8/255$ and $\alpha = 2/255$ are standard bounds). In other words: I wrote code to automatically generate adversarial images to test model. It takes a clean image, adds a tiny bit of random noise to start, and then makes 10 small, calculated adjustments to pixels to try and trick model. We cap total pixel change at 8/255 so edits remain invisible to human eyes, as in Rice et al. paper.
  * **Tuesday:** I integrated `PreActResNet18` model from our models folder and constructed core training loop. I implemented a custom `Normalizer` layer to normalize image batches *after* PGD perturbation is added, ensuring attack constraints are applied correctly to raw $[0, 1]$ pixels. In other words: I set up actual training loop where our PreActResNet-18 neural network is trained on these adversarial images so it learns to handle them. I made sure we generate adversarial edits on raw images first before adjusting contrast/brightness for network, ensuring size of our edits matches paper's math exactly.
  * **Wednesday:** I added key pipeline features to `train.py`: a MultiStepLR scheduler (with decays at epochs 100 and 150 to match 200-epoch schedule), automatic checkpoint saving to `Checkpoints/` directory every 5 epochs, and TensorBoard (a graphing tool for visualizing training progress) logging via `SummaryWriter` to track clean and robust train/test statistics. In other words: I added utility features to keep training organized. I set up a schedule to slow down model's learning speed at epochs 100 and 150 to help it fine-tune. I also set it to save model weights every 5 epochs so we can analyze them later, and hooked up TensorBoard to graph model's accuracy and defense strength in real-time.
  * **Thursday:** I finalized command-line argument configuration and implemented a `--diagnostic` mode to allow for quick testing. I ran this diagnostic run locally on my MacBook Pro using Apple Silicon (MPS acceleration) for 1 epoch on 10% of CIFAR-10 data. The smoke test verified that training loss decreases, weights update correctly, and checkpoints/TensorBoard logs are saved successfully without any memory errors. In other words: I made it easy to run different training setups and created a "quick test" mode. Since a full training run takes hours on a cloud GPU, I ran a tiny version of it locally on my MacBook's GPU. The test completed successfully, proving training loop, checkpoint saving, and logging all work together without crashing.
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

* **Note on FGSM vs. PGD:** While Goodfellow et al. introduced Fast Gradient Sign Method (FGSM), it is a simple **single-step** attack that causes models to learn to become robust only to that specific one-step direction while failing against stronger attacks. Projected Gradient Descent (PGD) is a **multi-step** version of FGSM that takes multiple small steps to find a much stronger, worst-case adversarial perturbation. Replicating Rice et al. requires PGD training because it builds actual robustness, and robust overfitting problem is most clearly documented and visible under PGD-trained networks.

* **Deliverables:**
  * [`train.py`](../train.py): PyTorch training script with a custom PGD training loop, learning rate scheduling, evaluation loop, TensorBoard integration, checkpoint saving, and diagnostic test support. In other words: This is our main code file. Its purpose is to run entire training process, generate adversarial images, slow down learning speed at specific epochs, save checkpoints, and log progress stats.
  * [`Checkpoints/diagnostic/epoch_1.pt`](../Checkpoints/diagnostic/epoch_1.pt): Saved checkpoint file from local diagnostic smoke test. In other words: This is a saved file containing model's weights (its learned patterns) after running our 1-epoch quick test. Its purpose is to prove that weight saving works properly and can be reloaded later.
  * [`.gitignore`](../.gitignore): Updated to ignore the `data/` directory to prevent large dataset archives (like `cifar-10-python.tar.gz` which is ~170MB and exceeds GitHub's 100MB limit) from being tracked, ensuring clean commits while relying on `train.py`'s automatic download feature.

---

## Week 4 (In Progress)
* **Progress Report:** 

* **Deliverables:**

