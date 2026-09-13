# Pixel-Only: Seed 44

## Numbers

- Evaluated 40 checkpoints at five-epoch intervals from epoch 5 through epoch 200.
- Epoch 200: clean **84.48%**, pixel-PGD-20 **42.88%**, low-frequency-PGD-20 **76.46%**, and joint robustness **42.88%**.
- Peaks: clean **85.01%** at epoch 155; pixel **51.18%** at epoch 105; low-frequency **78.42%** at epoch 155; joint **51.18%** at epoch 105.
- Peak-to-final declines: pixel **8.30 pp**, low-frequency **1.96 pp**, and joint **8.30 pp**.
- Minimum losses occur at epoch 155 for clean (0.468), epoch 105 for pixel (1.299), and epoch 115 for low-frequency (0.654).

## Trajectory

- Pixel and joint robustness peak at epochs 105 and 105, then finish 8.30 pp and 8.30 pp below their peaks.
- Joint robustness nearly equals pixel robustness throughout, so pixel-PGD is the limiting attack for this model.
- Pixel robust loss falls to 1.299, then rises to 3.277 at epoch 200.

## Interpretation

- This run shows the classic robust-overfitting pattern: adversarial test accuracy deteriorates after its best checkpoint even as training continues.
- Checkpoint selection matters: the best pixel-robust checkpoint is preferable to the final checkpoint for the pixel threat model.

## Visualizations

- [Evaluation curves](po_eval.pdf)
- [Training dynamics](po_train.pdf)

The TensorBoard test-robust curve in the training-dynamics figure uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
