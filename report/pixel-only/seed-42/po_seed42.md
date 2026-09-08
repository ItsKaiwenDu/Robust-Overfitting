# Pixel-Only: Seed 42

## Numbers

- Evaluated 40 checkpoints at five-epoch intervals from epoch 5 through epoch 200.
- Epoch 200: clean **84.48%**, pixel-PGD-20 **42.57%**, low-frequency-PGD-20 **76.59%**, and joint robustness **42.57%**.
- Peaks: clean **85.31%** at epoch 155; pixel **51.30%** at epoch 105; low-frequency **78.41%** at epoch 155; joint **51.30%** at epoch 105.
- Peak-to-final declines: pixel **8.73 pp**, low-frequency **1.82 pp**, and joint **8.73 pp**.
- Minimum losses occur at epoch 155 for clean (0.467), epoch 105 for pixel (1.290), and epoch 115 for low-frequency (0.648).

## Trajectory

- Pixel and joint robustness peak at epochs 105 and 105, then finish 8.73 pp and 8.73 pp below their peaks.
- Joint robustness nearly equals pixel robustness throughout, so pixel-PGD is the limiting attack for this model.
- Pixel robust loss falls to 1.290, then rises to 3.219 at epoch 200.

## Interpretation

- This run shows the classic robust-overfitting pattern: adversarial test accuracy deteriorates after its best checkpoint even as training continues.
- Checkpoint selection matters: the best pixel-robust checkpoint is preferable to the final checkpoint for the pixel threat model.

## Visualizations

- [Evaluation curves](po_eval_results_curves.pdf)
- [Training dynamics](po_train_results_curves.pdf)

The TensorBoard test-robust curve in the training-dynamics figure uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
