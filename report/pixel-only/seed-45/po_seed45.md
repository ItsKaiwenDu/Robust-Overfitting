# Pixel-Only: Seed 45

## Numbers

- Evaluated 40 checkpoints at five-epoch intervals from epoch 5 through epoch 200.
- Epoch 200: clean **84.35%**, pixel-PGD-20 **42.89%**, low-frequency-PGD-20 **76.83%**, and joint robustness **42.89%**.
- Peaks: clean **85.05%** at epoch 160; pixel **51.41%** at epoch 105; low-frequency **78.58%** at epoch 155; joint **51.40%** at epoch 105.
- Peak-to-final declines: pixel **8.52 pp**, low-frequency **1.75 pp**, and joint **8.51 pp**.
- Minimum losses occur at epoch 155 for clean (0.470), epoch 105 for pixel (1.289), and epoch 120 for low-frequency (0.645).

## Trajectory

- Pixel and joint robustness peak at epochs 105 and 105, then finish 8.52 pp and 8.51 pp below their peaks.
- Joint robustness nearly equals pixel robustness throughout, so pixel-PGD is the limiting attack for this model.
- Pixel robust loss falls to 1.289, then rises to 3.151 at epoch 200.

## Interpretation

- This run shows the classic robust-overfitting pattern: adversarial test accuracy deteriorates after its best checkpoint even as training continues.
- Checkpoint selection matters: the best pixel-robust checkpoint is preferable to the final checkpoint for the pixel threat model.

## Visualizations

- [Evaluation curves](po_eval_results_curves.pdf)
- [Training dynamics](po_train_results_curves.pdf)

The TensorBoard test-robust curve in the training-dynamics figure uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
