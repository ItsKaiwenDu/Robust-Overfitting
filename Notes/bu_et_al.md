# Notes: "Towards Building More Robust Models with Frequency Bias" (Bu, Huang & Cui, ICCV 2023)

## 1. Summary

This paper asks a structural question about adversarial robustness: instead of filtering the frequency content of input images directly, what if a model's internal, intermediate features were reshaped in the frequency domain instead? Prior work established that adversarially robust models tend to lean on low-frequency information more than standard models do, but existing attempts to exploit this applied low-pass filtering directly to input images, which permanently throws away high-frequency detail and hurts clean accuracy by 4-5%. The authors propose the **Frequency Preference Control Module (FPCM)**, a small, plug-and-play component inserted between layers of a network that reweights the low- and high-frequency components of feature maps, rather than the input image itself. Since this reweighting happens on internal features and is trained jointly with the model, it avoids the irreversible information loss of input-level filtering. Tested across CIFAR-10, CIFAR-100, and Imagenette, and combined with several existing adversarial training methods (PGD-AT, TRADES, AWP, LAS-AT), FPCM consistently improves robust accuracy with little to no cost to clean accuracy.

**Note on how this relates to the DAT paper (Li et al., NeurIPS 2024):** DAT works in the *input* frequency domain, mixing amplitude spectra of whole images to steer a model toward phase-based features. FPCM instead operates on *intermediate feature maps* inside the network, reweighting their low- and high-frequency components directly. Both start from the same broad finding, that robust models rely more on certain frequency content, but DAT manipulates what the model sees at the input, while FPCM manipulates what the model computes internally.

## 2. Key Terms

- **Low-Frequency vs. High-Frequency Components**: In an image or feature map, low-frequency components capture coarse, smooth, large-scale structure, while high-frequency components capture fine detail and texture. Prior work found that CNNs tend to lean on high-frequency (texture) information, while adversarially robust models lean more on low-frequency information.
- **Frequency Preference Control Module (FPCM)**: The paper's proposed plug-and-play module. It transforms an intermediate feature map into the frequency domain, suppresses its high-frequency content with a low-pass filter, then blends the filtered (low-frequency) and unfiltered (original) versions back together using a learned per-channel weight.
- **Gaussian Low-Pass Filter (GLPF)**: The specific type of low-pass filter FPCM uses to suppress high-frequency signal in the frequency domain. It's a common choice in image processing because it doesn't introduce ringing artifacts.
- **Cutoff Frequency (β)**: A hyperparameter controlling how aggressively the low-pass filter suppresses high-frequency content. A smaller β keeps less high-frequency information.
- **F-Principle (Frequency Principle)**: An empirically and theoretically supported idea from prior work (Xu et al.) that deep neural networks tend to learn low-frequency patterns in the data before high-frequency ones during training.

## 3. The Problem: Filtering Inputs Throws Away Information

**Starting observation.** Multiple prior studies found that CNNs are naturally biased toward high-frequency texture information, while adversarially trained models shift toward relying on low-frequency information, which is linked to their improved robustness. This connection between frequency bias and robustness motivated several follow-up attempts to directly exploit it.

**The existing fix, and its problem.** Prior methods tried to apply low-pass filtering directly to the input images, either clean or adversarial, to suppress high-frequency content before the model ever sees it. This has two problems:

- Filtering the input directly causes an **irreversible loss** of high-frequency information that can still be useful for correctly classifying an image, leading to a real drop in clean accuracy (the paper cites a 4-5% clean accuracy drop for one such method).
- The right amount of filtering (the cutoff frequency) needs to be tuned differently for different datasets, since datasets have different natural frequency characteristics, adding a fragile hyperparameter tuning burden.

There's a separate, related observation feeding into the design: Vision Transformers (ViTs) and CNNs behave oppositely in the frequency domain. ViTs tend to reduce high-frequency signal as information passes through the network, while CNNs amplify it. Since ViTs' low-frequency bias is linked to better robustness, the authors reasoned it would help to give CNNs a similar mechanism, but at the level of internal features rather than by filtering the input.

## 4. Designing the Fix: The FPCM Module

The core design choice is to move frequency reweighting from the *input image* to *intermediate feature maps*, and to make the reweighting learnable rather than a fixed, hand-tuned filter.

**How FPCM processes a feature map, step by step:**

1. Take an intermediate feature map from inside the network and transform it into the frequency domain using the Fast Fourier Transform (FFT).
2. Apply a Gaussian low-pass filter to suppress high-frequency components, producing a filtered frequency representation.
3. Apply the inverse FFT to bring the filtered result back into a low-frequency-only version of the feature map.
4. Compute per-channel weights (between 0.5 and 1 by default) from the *original* input feature map, using a small learned layer followed by a sigmoid function. These weights represent how much each channel should favor low-frequency content.
5. Combine the low-frequency version and the original (unfiltered) version using these weights as a weighted sum, so the final output is a blend rather than a hard replacement.

Because the weights are pushed toward the [0.5, 1] range by default, the module is biased toward preserving more low-frequency content, consistent with the robustness-related motivation, but it never fully discards high-frequency information the way input-level filtering does.

**Design principles the authors aimed for:**

- **Efficiency**: the module adds only a marginal number of parameters and little compute overhead.
- **Differentiability**: since FFT and the low-pass filtering step are linear operations, gradients pass through cleanly, so FPCM can be trained jointly with the rest of the network rather than as a separate preprocessing step.
- **Adaptability**: because the reweighting is learned per-channel and per-layer, the module can adapt to the different frequency characteristics found at different depths of a network, without needing per-dataset hyperparameter tuning.

**Where to insert it.** The authors found that convolutional layers tend to accumulate more high-frequency signal as depth increases, peaking at the end of each stage of a network (this is an **empirically observed pattern**, shown in Figure 3 of the paper, not a formal proof). Based on this, they insert one FPCM module at the end of each stage of a ResNet-style architecture, which they describe as a "painless" insertion that doesn't require restructuring the base model. The same approach is designed to generalize to other architectures like WideResNets.

**A secondary refinement: scheduling the cutoff frequency during training.** Motivated by the F-Principle, the idea that models naturally learn low-frequency patterns before high-frequency ones, the authors also linearly *decrease* the cutoff frequency over the course of training, starting broad (retaining more frequency information early on) and narrowing toward mostly low-frequency by the end. The stated goal is to let the model converge on the full frequency spectrum early, then increasingly emphasize low-frequency content later in training, which they argue benefits robustness.

## 5. Results

**Main robustness comparison (CIFAR-10, ResNet-18 and WRN-34-10):**

- Combined with PGD-AT, TRADES, or AWP, FPCM ("Ours-AT", "Ours-TRADES", "Ours-AWP") outperformed the corresponding baseline under most attack types (PGD-10/20/50, C&W, AutoAttack), typically without a meaningful drop in clean accuracy.
- With WRN-34-10 built on AWP, FPCM improved robustness under PGD-50 by 0.95% and under AutoAttack by 1.28%. Adding LAS on top pushed this further, by 0.66% (PGD-50) and 0.29% (AutoAttack).
- With ResNet-18 built on PGD-AT, FPCM improved PGD-50 by 2.47%, C&W by 2.47%, and AutoAttack by 1.84%.
- Across both architectures, the AWP-based version of FPCM achieved the best results under all attacks tested.

**Scaling to a larger model without extra training data.** Using a WRN-70-16 architecture, FPCM built on AWP outperformed prior state-of-the-art robust models (Gowal et al., LAS-AWP) on AutoAttack, while matching or slightly beating LAS in terms of overall performance.

**Generalization to other datasets:**

- On Imagenette (higher resolution, 160×160 images), FPCM improved AutoAttack performance by 1.2% over PGD-AT and 0.5% over TRADES.
- On CIFAR-100 (more classes than CIFAR-10) with ResNet-18, FPCM improved robustness under PGD-20 and AutoAttack by 2.39% and 2.24% respectively.

**Comparison specifically against other frequency-based defenses.** Compared against FR (frequency regularization, the prior state-of-the-art frequency-based method, which filters inputs), FPCM showed a smaller AutoAttack improvement (+0.83%) but a much larger clean accuracy advantage (+4.06%). The authors attribute FR's clean accuracy cost to its direct filtering of input images, which discards signal the model needs for correct classification on clean data. This is presented as the central practical advantage of operating on intermediate features instead of inputs.

## 6. Ablation Studies

**On the cutoff frequency (β):**

- With fixed β values, natural (clean) accuracy dropped and robust accuracy rose as β got smaller, consistent with the intuition that retaining less high-frequency information trades clean accuracy for robustness.
- Making β itself learnable per-channel didn't outperform good fixed values, which the authors suggest may be because it's redundant with the already-learnable α weights.
- Their proposed linear scheduling of the cutoff frequency during training (starting broad, narrowing to low-frequency by the end) gave the best overall trade-off, as measured by their combined "W-Robust" metric (an average of clean and robust accuracy).

**On the low-frequency reweighting (α):**

- α could not be fixed at 0 (fully discarding high-frequency information at this stage), since the model failed to converge without it, most information sits in the DC (zero-frequency) component.
- Pushing α above 0.5, meaning favoring low-frequency features, consistently improved the W-Robust metric by roughly 0.37-1.79% over lower settings.
- Making α fully learnable (rather than fixed) improved results further by about 0.39% W-Robust.
- Using a more expressive learnable component (an MLP instead of a single convolutional layer) to compute α gave only a marginal further improvement (+0.03% W-Robust), suggesting the simpler design was already close to sufficient.

**Examining the learned α weights.** Deeper stages of the network consistently learned higher α values (more weight on low-frequency features) than earlier stages; the average α was 0.6879 in stage 1 versus 0.7697 in stage 3, and never dropped below 0.74 in stage 3 across channels. The authors interpret this as the model actively counteracting the tendency of high-frequency signal to accumulate at greater depth (the pattern shown in Figure 3). They also found the learned weights had very low sample-wise variance (usually under 1e-4), suggesting FPCM adapts to the general frequency characteristics of a robust model rather than to specific input samples.

## 7. Additional Analysis: Frequency Bias of Robust Learning

Beyond the main robustness results, the authors ran additional experiments specifically to characterize *why* low-frequency bias helps:

- **Noise robustness by frequency.** They tested models against random noise injected at different frequency bands. Standard (vanilla) models were far more vulnerable to high-frequency noise than low-frequency noise, and this vulnerability grew sharply as the injected noise concentrated more in high frequencies. PGD-AT was already more resistant to this than the vanilla model, and FPCM (Ours-AT) improved resistance to high-frequency noise further still. Robustness to very low-frequency noise, by contrast, did not meaningfully change between the vanilla model, PGD-AT, and Ours-AT.
- **Fixed-α training curves.** When α (the low-frequency weighting) was fixed at a low value (0.1) throughout training, both robust accuracy and the loss landscape were worse than at higher fixed α values throughout training, not just at the end. The authors read this as further evidence that low-frequency features specifically help the *process* of robust learning, not just the final outcome.

## 8. Works Cited

Bu, Qingwen, Dong Huang, and Heming Cui. "Towards Building More Robust Models with Frequency Bias." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023.