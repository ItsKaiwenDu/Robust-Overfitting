# Notes: "DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in Frequency Domain" (Li, Li, Wu, Tian & Zhou, NeurIPS 2024)

## 1. Summary

This paper looks at why adversarial examples (AEs) fool image classifiers, using the frequency-domain view of images (amplitude and phase). The authors find that adversarial attacks damage the **phase** spectrum of an image, which carries most of the semantic content, much more than the **amplitude** spectrum, which carries stylistic information. Building on this, they propose **Dual Adversarial Training (DAT)**: a training method that mixes in an adversarially generated amplitude spectrum to force the model to rely more on phase patterns, since those are less corrupted by attacks. DAT uses a small generator network, the Adversarial Amplitude Generator (AAG), to create this adversarial amplitude, plus a faster way to generate adversarial examples so training doesn't take twice as long. Across CIFAR-10, CIFAR-100, and Tiny ImageNet, DAT improves robust accuracy by about 2.1 to 2.3% on average over prior state-of-the-art methods, while also slightly improving accuracy on clean, non-adversarial images.

**Note on how this relates to Chen et al. (Amplitude-Phase Recombination, 2021):** Chen et al. established that recombining amplitude and phase from different images can probe and improve model robustness, but relied on real distractor images to source the amplitude. This paper, Li et al., fills the gap of picking a good distractor by replacing it with a trained generator, and adds a formal proof for why focusing on phase patterns helps.

## 2. Key Terms

- **Amplitude Spectrum**: The part of an image's frequency-domain representation that captures style-like information, such as brightness or texture intensity, rather than object identity.
- **Phase Spectrum**: The part of an image's frequency-domain representation that captures most of the semantic and structural information, meaning what the object actually is.
- **Adversarial Amplitude Generator (AAG)**: A small neural network the authors design to generate a synthetic "adversarial" amplitude spectrum, used to force the model to depend less on amplitude and more on phase.
- **Robust Accuracy**: A model's classification accuracy specifically on adversarial examples, as opposed to natural accuracy, which is accuracy on clean images.
- **Split Batch Normalization (Split BN)**: Using two separate sets of batch normalization parameters, one for original images and one for recombined images, because their statistics differ too much to share one set.

## 3. The Problem: Why Does Focusing on Phase Help, and How Do You Get There?

**Starting observation.** Adversarial attacks damage phase patterns far more severely than amplitude patterns. A standard model trained without adversarial training already performs worse on samples with adversarial phase patterns than on samples with adversarial amplitude patterns. Adversarially trained (robust) models handle phase-level perturbations better too, more so than amplitude-level ones, unlike standard models. This is worth being precise about: this is an **empirically observed pattern** from the authors' exploration experiments (Sec. 2 of the paper), not a formal proof.

**The existing fix, and its problem.** Prior work showed that mixing a training image's amplitude with that of a randomly selected distractor image pushes a model to focus more on phase patterns, improving robustness. But picking a good distractor has a tradeoff:

- If the distractor is too different from the original image, it damages the phase patterns too, hindering the model from predicting the adversarial example correctly.
- If the distractor is too similar to the original image, the model isn't pushed to rely on phase patterns at all.

There is no reliable, automatic way to pick a distractor that lands in the right spot. This paper's core move is to stop picking a distractor and instead generate one.

## 4. Designing the Fix: The Adversarial Amplitude Generator

Based on the problem above, the authors lay out three conditions a good recombined image should satisfy, then design the AAG specifically to meet them.

**The three conditions (C1-C3):**

- **C1**: The recombined image should keep the same phase-spectrum semantics as the original.
- **C2**: The recombined image should still be classified the same as the original by the model.
- **C3**: The recombined image's amplitude should differ enough from the original to actually force the model to stop relying on amplitude.

**Why a random distractor fails these conditions.** A random distractor's amplitude easily satisfies C1 when the distractor happens to be close to the original, but then it fails C3, since the amplitude isn't different enough to push the model away from it. Pushed the other way, a very different distractor satisfies C3 but breaks C1 and C2 by damaging the phase.

**The AAG's design.** Instead of a random distractor, the authors train a generator, Gψ, to output a synthetic amplitude conditioned on random noise and the model's own logits for that image. This generated amplitude is:

1. Trained adversarially, meaning it's optimized to maximize the model's loss, so it satisfies C3 by construction.
2. **Mixed** with the image's real amplitude, rather than fully replacing it, using a random mixing weight between 0 and 1. This mix-up is what actually keeps C1 and C2 intact, since it preserves some of the true amplitude information and keeps the phase spectrum from the original image untouched.

**The rest of the training pipeline, briefly:**

- **Efficient AE generation.** Since both the original and recombined image need adversarial examples, which normally doubles computation, the authors add a loss term that increases how much each adversarial step moves the prediction, so fewer steps are needed to reach a similarly strong adversarial example.
- **Joint optimization.** The classifier and the AAG are trained together in a min-max setup: the classifier minimizes the loss, the AAG maximizes it. A consistency loss encourages matching predictions between the original and recombined image, since they share the same phase.
- **Split BN.** The recombined images have very different amplitude statistics than the originals, so sharing one set of batch normalization parameters hurt convergence. The fix is separate BN layers for original versus recombined data.

## 5. Results

**Main comparison (DAT vs. prior methods, ResNet-18):**

- On CIFAR-10, DAT improves robust accuracy by about 2.9% on average against FGSM, PGD-20, and PGD-100, with smaller but positive gains against the stronger C&W and AutoAttack (AA) benchmarks.
- On CIFAR-100, the average improvement is about 2.7% against the weaker attacks, with 1.3 to 1.5% gains against C&W and AA.
- On Tiny ImageNet, improvements are roughly 2.9% against weaker attacks and 1.2 to 1.6% against stronger ones.
- These gains hold when DAT is combined with existing robustness techniques (AWP and SWA), and also hold on a larger architecture (WRN-34-10).
- Unlike many robustness methods that trade away clean accuracy for adversarial robustness, DAT slightly **improves** clean accuracy too.

**Ablation studies, showing each component matters:**

- Removing the consistency loss: robust accuracy drops modestly, around 0.3 to 0.5%.
- Removing the mix-up step, using only the generated amplitude instead of blending it with the real one: robust accuracy drops more, around 1.6 to 1.9%, showing the model still needs some real amplitude information.
- Removing split BN: the largest drop of the ablations, around 3.5 to 4.6%, confirming this was an important fix rather than a minor tweak.
- Removing the AAG entirely, falling back to a random distractor: drops robust accuracy by 1.6 to 2.2%, confirming the generator approach outperforms random distractors.

## 6. Theoretical Analysis

The authors provide a **formally proven theorem** (Theorem 3.2, proved in Appendix G), under a stated assumption (Assumption 3.1: that augmentation increases the discrepancy between original and augmented feature representations more for amplitude-derived features than for phase-derived ones).

Under this assumption, they prove that as the model minimizes its training loss, the weights connected to amplitude-derived features shrink toward zero. This gives a mathematical explanation for why the model ends up relying more on phase-derived features. It's a separate line of support from the empirical results above, not a restatement of them.

## 7. Works Cited

Li, Fengpeng, Kemou Li, Haiwei Wu, Jinyu Tian, and Jiantao Zhou. "DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in Frequency Domain." *38th Conference on Neural Information Processing Systems (NeurIPS 2024)*, 2024.