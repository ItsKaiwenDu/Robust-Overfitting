# Mixed-Domain: Seed 46

## Numbers

- Evaluated 40 checkpoints at five-epoch intervals from epoch 5 through epoch 200.
- Epoch 200: clean **91.69%**, pixel-PGD-20 **16.48%**, low-frequency-PGD-20 **86.51%**, and joint robustness **16.48%**.
- Peaks: clean **92.84%** at epoch 155; pixel **45.58%** at epoch 130; low-frequency **89.84%** at epoch 155; joint **45.58%** at epoch 130.
- Peak-to-final declines: pixel **29.10 pp**, low-frequency **3.33 pp**, and joint **29.10 pp**.
- Minimum losses occur at epoch 155 for clean (0.228), epoch 125 for pixel (1.418), and epoch 110 for low-frequency (0.329).

## Trajectory

- Of the 40 evaluated checkpoints, 18 immediately follow a pixel-training epoch and 22 follow a low-frequency-training epoch; epoch 200 uses **low-frequency** training.
- Mean pixel robustness is **41.81%** after pixel epochs versus **2.79%** after low-frequency epochs.
- Joint robustness peaks at epoch 130 and ends 29.10 pp lower, while low-frequency robustness ends 3.33 pp below its peak.

## Interpretation

- Mixed training shows strong short-term specialization to the attack domain used in the immediately preceding epoch.
- Because the domain alternates once per epoch, the peak-to-final joint decline is schedule-confounded and cannot be attributed to robust overfitting alone.
- The run does not maintain stable joint robustness to both attacks; a finer-grained mixing strategy would be needed to test whether this is avoidable.

## Visualizations

- [Evaluation curves](mdo_eval.pdf)
- [Training dynamics](mdo_train.pdf)

The TensorBoard test-robust curve in the training-dynamics figure uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
