# Mixed-Domain: Seed 42

## Numbers

- Evaluated 40 checkpoints at five-epoch intervals from epoch 5 through epoch 200.
- Epoch 200: clean **84.21%**, pixel-PGD-20 **43.71%**, low-frequency-PGD-20 **77.64%**, and joint robustness **43.70%**.
- Peaks: clean **92.28%** at epoch 180; pixel **46.16%** at epoch 160; low-frequency **88.38%** at epoch 110; joint **46.16%** at epoch 160.
- Peak-to-final declines: pixel **2.45 pp**, low-frequency **10.74 pp**, and joint **2.46 pp**.
- Minimum losses occur at epoch 180 for clean (0.248), epoch 130 for pixel (1.428), and epoch 110 for low-frequency (0.334).

## Trajectory

- Of the 40 evaluated checkpoints, 26 immediately follow a pixel-training epoch and 14 follow a low-frequency-training epoch; epoch 200 uses **pixel** training.
- Mean pixel robustness is **42.48%** after pixel epochs versus **4.60%** after low-frequency epochs.
- Joint robustness peaks at epoch 160 and ends 2.46 pp lower, while low-frequency robustness ends 10.74 pp below its peak.

## Interpretation

- Mixed training shows strong short-term specialization to the attack domain used in the immediately preceding epoch.
- Because the domain alternates once per epoch, the peak-to-final joint decline is schedule-confounded and cannot be attributed to robust overfitting alone.
- The run does not maintain stable joint robustness to both attacks; a finer-grained mixing strategy would be needed to test whether this is avoidable.

## Visualizations

- [Evaluation curves](mdo_eval_results_curves.pdf)
- [Training dynamics](mdo_train_results_curves.pdf)

The TensorBoard test-robust curve in the training-dynamics figure uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
