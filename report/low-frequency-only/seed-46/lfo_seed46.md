# Low-Frequency-Only: Seed 46

## Numbers

- Evaluated 40 checkpoints at five-epoch intervals from epoch 5 through epoch 200.
- Epoch 200: clean **94.07%**, pixel-PGD-20 **0.00%**, low-frequency-PGD-20 **92.53%**, and joint robustness **0.00%**.
- Peaks: clean **94.20%** at epoch 190; pixel **0.00%** at epoch 5; low-frequency **92.65%** at epoch 170; joint **0.00%** at epoch 5.
- Peak-to-final declines: pixel **0.00 pp**, low-frequency **0.12 pp**, and joint **0.00 pp**.
- Minimum losses occur at epoch 105 for clean (0.205), epoch 5 for pixel (17.994), and epoch 105 for low-frequency (0.258).

## Trajectory

- Low-frequency robustness reaches 92.65% and changes by only 0.12 pp by epoch 200.
- Pixel and joint robustness remain effectively zero, despite strong clean and low-frequency accuracy.
- Low-frequency robust loss is minimized at epoch 105 (0.258) and ends at 0.320.

## Interpretation

- This seed learns strong matched-domain robustness with little or no late decline in low-frequency accuracy.
- The result does not transfer to unrestricted pixel-PGD; specialization to the low-frequency threat model is pronounced.

## Visualizations

- [Evaluation curves](lfo_eval_results_curves.pdf)
- [Training dynamics](lfo_train_results_curves.pdf)

The TensorBoard test-robust curve in the training-dynamics figure uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
