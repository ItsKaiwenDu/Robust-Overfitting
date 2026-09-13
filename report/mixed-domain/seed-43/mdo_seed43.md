# Mixed-Domain: Seed 43

## Numbers

- Evaluated 40 checkpoints at five-epoch intervals from epoch 5 through epoch 200.
- Epoch 200: clean **91.78%**, pixel-PGD-20 **15.93%**, low-frequency-PGD-20 **87.19%**, and joint robustness **15.93%**.
- Peaks: clean **92.83%** at epoch 165; pixel **47.79%** at epoch 115; low-frequency **89.17%** at epoch 165; joint **47.79%** at epoch 115.
- Peak-to-final declines: pixel **31.86 pp**, low-frequency **1.98 pp**, and joint **31.86 pp**.
- Minimum losses occur at epoch 165 for clean (0.223), epoch 115 for pixel (1.368), and epoch 165 for low-frequency (0.348).

## Trajectory

- Of the 40 evaluated checkpoints, 20 immediately follow a pixel-training epoch and 20 follow a low-frequency-training epoch; epoch 200 uses **low-frequency** training.
- Mean pixel robustness is **43.93%** after pixel epochs versus **5.00%** after low-frequency epochs.
- Joint robustness peaks at epoch 115 and ends 31.86 pp lower, while low-frequency robustness ends 1.98 pp below its peak.

## Interpretation

- Mixed training shows strong short-term specialization to the attack domain used in the immediately preceding epoch.
- Because the domain alternates once per epoch, the peak-to-final joint decline is schedule-confounded and cannot be attributed to robust overfitting alone.
- The run does not maintain stable joint robustness to both attacks; a finer-grained mixing strategy would be needed to test whether this is avoidable.

## Visualizations

- [Evaluation curves](mdo_eval.pdf)
- [Training dynamics](mdo_train.pdf)

The TensorBoard test-robust curve in the training-dynamics figure uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
