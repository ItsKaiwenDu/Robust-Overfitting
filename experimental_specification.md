# Experimental Specification: Pixel and Low-Frequency Adversarial Training

**Status:** Finalized Week 8 specification, 17 August 2026  
**Purpose:** Define the experiment before implementation so that the three
training conditions differ only in their training attack domain.

## Research question and estimands

**Research question.** During PGD adversarial training of PreActResNet-18 on
CIFAR-10, how does selecting pixel-space PGD or low-frequency DCT-masked PGD
once per epoch affect the timing and severity of robust overfitting, relative
to the corresponding single-domain training conditions?

For every training condition and evaluation threat, report:

- **peak epoch:** the saved-checkpoint epoch with the highest robust accuracy;
- **peak robust accuracy:** the accuracy at that checkpoint; and
- **post-peak decline:** peak robust accuracy minus robust accuracy at epoch
  200.

Ties are resolved by selecting the earliest checkpoint. Because checkpoints
are saved every five epochs, the peak epoch is measured to that resolution.

This is a controlled trajectory study. It does **not** claim to implement
AVG/MAX training, Multi Steepest Descent (MSD), or TaFD; those methods combine
or select threats at the example or PGD-step level, or change the model
architecture. Here, the chosen attack is fixed for every batch of one epoch.

## Shared experimental settings

| Item | Fixed setting |
| --- | --- |
| Dataset | CIFAR-10: all 50,000 training and 10,000 test images |
| Model | PreActResNet-18 with the existing CIFAR-10 normalizer |
| Input domain | Raw, unnormalized RGB tensors in `[0, 1]`; normalization is applied only immediately before the model |
| Data augmentation | Random crop to 32x32 with 4-pixel padding and random horizontal flip for training; no test augmentation |
| Optimizer | SGD, learning rate 0.1, momentum 0.9, weight decay `5e-4` |
| Learning-rate schedule | `MultiStepLR`, multiplied by 0.1 after epochs 100 and 150 |
| Training duration | 200 epochs |
| Training batch size | 128 |
| Training pixel budget | `epsilon = 8/255` in raw image space |
| Training steps | 10 PGD steps |
| Training step size | `alpha = 2/255` in raw image space |
| Checkpoints | Epochs 5, 10, ..., 200 (40 total) |
| Primary run seed | 42 for Python, NumPy, PyTorch CPU/CUDA, model initialization, augmentation, data order, and PGD random starts |

The already-completed pixel-only run remains the historical baseline. It will
be re-evaluated under the new paired evaluation protocol; new low-frequency
and mixed-domain runs use the exact settings above.

## Training conditions

1. **Pixel-only:** Use the existing pixel-space PGD-10 attack in every epoch.
2. **Low-frequency-only:** Use low-frequency DCT-masked PGD-10 in every
   epoch.
3. **Mixed-domain:** At the beginning of each epoch, select one attack domain
   with an independent fair Bernoulli draw from a dedicated `random.Random`
   generator seeded with 42. The selected domain is used for *all* batches in
   that epoch. This generator is not used for data ordering, augmentation, or
   PGD random starts.

The implementation will write one `attack_schedule.csv` per mixed run with
`epoch`, `attack_domain`, and `schedule_seed`. A low-frequency-only run writes
the same schema with `attack_domain=low_frequency` for every epoch. The
pixel-only baseline is represented analogously with `attack_domain=pixel`.

## Low-frequency DCT-masked PGD

### Transform and mask

For each image and RGB channel independently, use the orthonormal two-
dimensional DCT-II and its inverse:

\[
P_M(\delta) = \operatorname{IDCT}_2\!\left(M \odot
\operatorname{DCT}_2(\delta)\right).
\]

For CIFAR-10's 32x32 images, the fixed mask is

\[
M_{u,v}=1 \quad \text{when } 0 \leq u < 8 \text{ and } 0 \leq v < 8,
\]

and zero otherwise. Thus, each channel retains the top-left 8x8 DCT
coefficients, including its DC coefficient: 64 of 1,024 coefficients (6.25%).
The mask is identical for every image, epoch, checkpoint, and condition. It is
stored in the run configuration and included in the code as a deterministic
function, rather than learned or tuned per image.

The 8x8 cutoff is a precommitted definition of *low frequency* for this
study—not a hyperparameter selected after observing test performance. Any
future cutoff comparison must be a separately named experiment.

### Attack update

Let `X` be a raw image batch, `y` its labels, and `delta_0` a random pixel
perturbation sampled uniformly from `[-epsilon, epsilon]` and then
low-pass-projected with `P_M`. At every PGD step, compute the image-space loss
gradient, mask it in DCT space, and take an image-scale normalized ascent step:

\[
g_t = P_M\!\left(\nabla_{\delta} \mathcal{L}(f(X+\delta_t),y)\right),
\qquad
\delta_{t+1} = \Pi_{\infty, X}\!\left(\delta_t + \alpha
\frac{g_t}{\lVert g_t \rVert_\infty + 10^{-12}}\right).
\]

`Pi_{infinity, X}` first clips the perturbation coordinatewise to
`[-epsilon, epsilon]`, then clips `X + delta` to `[0, 1]` and subtracts `X`.
This keeps the *final evaluated image* valid and guarantees
`max(abs(X_adv - X)) <= 8/255`. Pixel PGD retains the existing random-start,
sign-gradient update, and identical image-space projection.

### Clipping and spectral-leakage rule

Pixel-value clipping is not a DCT-subspace projection: when a proposed pixel
value is clipped at 0 or 1, the resulting final perturbation can contain a
small amount of frequency content outside `M`. Therefore this experiment uses
the precise name **low-frequency DCT-masked PGD**, not a claim of a mathematically
exact DCT-subspace threat set after clipping.

For every low-frequency attack invocation, the implementation will record
the batch mean and maximum *out-of-mask energy fraction* of the final
perturbation,

\[
\frac{\lVert (1-M)\odot\operatorname{DCT}_2(\delta) \rVert_2^2}
{\lVert \operatorname{DCT}_2(\delta) \rVert_2^2 + 10^{-12}}.
\]

The diagnostic must show no implementation error and report this value. If
clipping creates material leakage (mean above 1%), the result will be reported
as a limitation and the later strict-subspace alternative will require a
coefficient-space constrained optimizer; it will not be silently described as
an exact constrained attack.

## Checkpoint evaluation

Every saved checkpoint is evaluated over the complete 10,000-image CIFAR-10
test set, using the following fixed suite:

| Metric | Attack |
| --- | --- |
| `clean_acc`, `clean_loss` | No attack |
| `pixel_pgd20_acc`, `pixel_pgd20_loss` | Pixel-space PGD-20, `epsilon=8/255`, `alpha=2/255`, random start |
| `low_frequency_pgd20_acc`, `low_frequency_pgd20_loss` | DCT-masked PGD-20 with the fixed 8x8 mask and the same image-space budget/step size |
| `union_acc` | Per-example intersection of the two attack-correct indicators: correct only when both attacks leave the example correctly classified |

The union metric is computed per example, not as the minimum of the two
aggregate accuracies. Its loss is intentionally not reported: the union
quantity is a worst-case success indicator, whereas a loss would require an
additional defined rule for selecting one adversarial image per example.

Evaluation uses a separate fixed seed, 4242, reset before each checkpoint.
The evaluation script must preserve each example's pixel and low-frequency
attack correctness indicators until `union_acc` is computed. It will output
one row per checkpoint in `evaluation_results.csv`.

## Required run artifacts

Each condition is stored under a distinct directory (for example,
`Report/low_frequency_seed42/`) and must contain:

- `config.json`: all command-line arguments, git commit SHA, Python/PyTorch/
  torchvision versions, CUDA version, GPU model, device, DCT version, 8x8
  mask definition, seed values, and dataset sizes;
- `checkpoints/epoch_<N>.pt` for all 40 checkpoint epochs;
- `training_log.csv` with epoch, learning rate, training loss, clean accuracy,
  adversarial accuracy, attack domain, elapsed time, and leakage statistics;
- `attack_schedule.csv` as described above;
- `evaluation_results.csv` with all four metrics in the evaluation table;
- plots of clean, pixel-PGD, low-frequency-PGD, and union accuracy versus
  epoch; and
- the exact launch command in `command.txt`.

No results from a failed, restarted, or diagnostically different run may be
merged into the primary run directory. Such runs receive their own directory
and a clear suffix.

## Diagnostics and acceptance criteria

Before any 200-epoch cloud run, execute a one-epoch local diagnostic over the
existing 10% data subsets for each attack domain. The diagnostic passes only
when all of the following are true:

1. Pixel and low-frequency attacks return finite tensors on the selected
   device and training completes one optimizer step.
2. All returned adversarial images lie in `[0, 1]`, and every final
   perturbation has `L-infinity <= 8/255 + 1e-6`.
3. The DCT mask has shape 32x32, keeps exactly 64 coefficients per channel,
   and is identical across the batch.
4. A low-frequency attack's pre-clipping perturbation has zero out-of-mask
   DCT energy to numerical tolerance (`<= 1e-6` fraction).
5. The final post-clipping leakage statistic is written to the training log.
6. The mixed diagnostic logs exactly one selected domain for its epoch and
   re-running it with the same seed produces the same schedule.
7. Checkpoint save/load works and evaluation emits clean, pixel, low-frequency,
   and per-example union metrics.

## Scope and interpretation

The primary comparison is the shape of robust-overfitting curves, not a
leaderboard claim of state-of-the-art union robustness. A single seed gives a
reproducible case study; it does not establish uncertainty across training
runs. If time permits after the primary conditions, rerun all three conditions
with seeds 43 and 44 and report mean and range without changing any other
setting.

The frequency mask follows the transform-mask-inverse-transform approach used
to restrict low-frequency adversarial searches by Guo, Frank, and Weinberger.
The per-example union metric follows the multi-perturbation evaluation
principle emphasized by Tramèr and Boneh and by Maini, Wong, and Kolter.

## References

- Guo, C., Frank, J. S., & Weinberger, K. Q. (2020). *Low Frequency
  Adversarial Perturbation.* UAI 2020. https://proceedings.mlr.press/v115/guo20a.html
- Tramèr, F., & Boneh, D. (2019). *Adversarial Training and Robustness for
  Multiple Perturbations.* NeurIPS 2019. https://arxiv.org/abs/1904.13000
- Maini, P., Wong, E., & Kolter, J. Z. (2020). *Adversarial Robustness Against
  the Union of Multiple Perturbation Models.* ICML 2020.
  https://proceedings.mlr.press/v119/maini20a.html
