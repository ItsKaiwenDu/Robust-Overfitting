# Mixed-Domain: Seed 45

## Numbers

- Evaluated 40 checkpoints at five-epoch intervals from epoch 5 through epoch 200.
- Epoch 200: clean **91.34%**, pixel-PGD-20 **14.58%**, low-frequency-PGD-20 **86.36%**, and joint robustness **14.58%**.
- Peaks: clean **92.45%** at epoch 180; pixel **46.27%** at epoch 125; low-frequency **88.85%** at epoch 180; joint **46.27%** at epoch 125.
- Peak-to-final declines: pixel **31.69 pp**, low-frequency **2.49 pp**, and joint **31.69 pp**.
- Minimum losses occur at epoch 165 for clean (0.236), epoch 125 for pixel (1.404), and epoch 110 for low-frequency (0.328).

## Trajectory

- Of the 40 evaluated checkpoints, 19 immediately follow a pixel-training epoch and 21 follow a low-frequency-training epoch; epoch 200 uses **low-frequency** training.
- Mean pixel robustness is **41.55%** after pixel epochs versus **2.99%** after low-frequency epochs.
- Joint robustness peaks at epoch 125 and ends 31.69 pp lower, while low-frequency robustness ends 2.49 pp below its peak.

## Interpretation

- Mixed training shows strong short-term specialization to the attack domain used in the immediately preceding epoch.
- Because the domain alternates once per epoch, the peak-to-final joint decline is schedule-confounded and cannot be attributed to robust overfitting alone.
- The run does not maintain stable joint robustness to both attacks; a finer-grained mixing strategy would be needed to test whether this is avoidable.

## Visualizations

- [Evaluation curves](mdo_eval.pdf)
- [Training dynamics](mdo_train.pdf)

The TensorBoard test-robust curve in the training-dynamics figure uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
