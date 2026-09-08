# Mixed-Domain: Overall Results

## Numbers

- Five runs (seeds 42-46), each with 40 checkpoints evaluated every five epochs from epoch 5 through epoch 200.
- Epoch 200 mean +/- sample SD: clean **90.33% +/- 3.45 pp**, pixel-PGD-20 **19.33% +/- 14.28 pp**, low-frequency-PGD-20 **85.20% +/- 4.29 pp**, and joint robustness **19.33% +/- 14.27 pp**.
- Mean peaks: clean **91.69%** at epoch 195; pixel **40.80%** at epoch 85; low-frequency **86.92%** at epoch 195; joint **40.79%** at epoch 85.
- Mean peak-to-final declines: pixel **21.47 pp**, low-frequency **1.72 pp**, and joint **21.46 pp**.
- Mean minimum losses: clean **0.268 +/- 0.014** at epoch 195; pixel **1.560 +/- 0.017** at epoch 85; low-frequency **0.415 +/- 0.195** at epoch 110.

## Trajectory

- At the aggregate joint peak (epoch 85), 5/5 runs had just used a pixel epoch; at epoch 200, only 1/5 had.
- Across all 200 seed-checkpoint observations, pixel robustness averages **42.36%** after pixel epochs versus **3.53%** after low-frequency epochs.
- The point-biserial correlation between a preceding pixel epoch and pixel robustness is **r = 0.971**.
- Clean and low-frequency performance are generally higher immediately after low-frequency epochs, revealing the opposite side of the same recency effect.

## Interpretation

- The apparent 21.46-point aggregate joint decline is strongly confounded by the once-per-epoch attack schedule; it must not be interpreted as pure robust overfitting.
- The mixed condition alternates between domain-specialized states rather than converging to stable joint robustness. Batch-level mixing or a combined inner maximization would be a better test of simultaneous robustness.
- The large epoch-200 standard deviations reflect which domain each seed encountered most recently, not merely ordinary seed-to-seed noise.

## Visualizations

- [Five-seed evaluation curves](mdo_eval_results_curves.pdf)
- [Five-seed training dynamics](mdo_train_results_curves.pdf)

Shaded bands show sample standard deviation across seeds. The TensorBoard test-robust curve uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
