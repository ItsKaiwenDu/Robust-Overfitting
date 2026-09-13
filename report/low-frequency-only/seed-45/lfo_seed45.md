# Low-Frequency-Only: Seed 45

## Numbers

- Evaluated 40 checkpoints at five-epoch intervals from epoch 5 through epoch 200.
- Epoch 200: clean **94.03%**, pixel-PGD-20 **0.00%**, low-frequency-PGD-20 **92.33%**, and joint robustness **0.00%**.
- Peaks: clean **94.08%** at epoch 195; pixel **0.00%** at epoch 5; low-frequency **92.51%** at epoch 195; joint **0.00%** at epoch 5.
- Peak-to-final declines: pixel **0.00 pp**, low-frequency **0.18 pp**, and joint **0.00 pp**.
- Minimum losses occur at epoch 105 for clean (0.214), epoch 5 for pixel (17.669), and epoch 105 for low-frequency (0.270).

## Trajectory

- Low-frequency robustness reaches 92.51% and changes by only 0.18 pp by epoch 200.
- Pixel and joint robustness remain effectively zero, despite strong clean and low-frequency accuracy.
- Low-frequency robust loss is minimized at epoch 105 (0.270) and ends at 0.325.

## Interpretation

- This seed learns strong matched-domain robustness with little or no late decline in low-frequency accuracy.
- The result does not transfer to unrestricted pixel-PGD; specialization to the low-frequency threat model is pronounced.

## Visualizations

- [Evaluation curves](lfo_eval.pdf)
- [Training dynamics](lfo_train.pdf)

The TensorBoard test-robust curve in the training-dynamics figure uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
