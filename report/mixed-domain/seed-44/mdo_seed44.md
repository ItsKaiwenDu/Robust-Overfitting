# Mixed-Domain: Seed 44

## Numbers

- Evaluated 40 checkpoints at five-epoch intervals from epoch 5 through epoch 200.
- Epoch 200: clean **92.63%**, pixel-PGD-20 **5.96%**, low-frequency-PGD-20 **88.30%**, and joint robustness **5.96%**.
- Peaks: clean **92.63%** at epoch 200; pixel **46.83%** at epoch 155; low-frequency **88.94%** at epoch 110; joint **46.83%** at epoch 155.
- Peak-to-final declines: pixel **40.87 pp**, low-frequency **0.64 pp**, and joint **40.87 pp**.
- Minimum losses occur at epoch 165 for clean (0.225), epoch 155 for pixel (1.421), and epoch 110 for low-frequency (0.321).

## Trajectory

- Of the 40 evaluated checkpoints, 19 immediately follow a pixel-training epoch and 21 follow a low-frequency-training epoch; epoch 200 uses **low-frequency** training.
- Mean pixel robustness is **41.89%** after pixel epochs versus **2.72%** after low-frequency epochs.
- Joint robustness peaks at epoch 155 and ends 40.87 pp lower, while low-frequency robustness ends 0.64 pp below its peak.

## Interpretation

- Mixed training shows strong short-term specialization to the attack domain used in the immediately preceding epoch.
- Because the domain alternates once per epoch, the peak-to-final joint decline is schedule-confounded and cannot be attributed to robust overfitting alone.
- The run does not maintain stable joint robustness to both attacks; a finer-grained mixing strategy would be needed to test whether this is avoidable.

## Visualizations

- [Evaluation curves](mdo_eval_results_curves.pdf)
- [Training dynamics](mdo_train_results_curves.pdf)

The TensorBoard test-robust curve in the training-dynamics figure uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
