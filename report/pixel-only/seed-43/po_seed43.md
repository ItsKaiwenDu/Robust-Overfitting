# Pixel-Only: Seed 43

## Numbers

- Evaluated 40 checkpoints at five-epoch intervals from epoch 5 through epoch 200.
- Epoch 200: clean **84.43%**, pixel-PGD-20 **42.56%**, low-frequency-PGD-20 **76.40%**, and joint robustness **42.56%**.
- Peaks: clean **85.14%** at epoch 160; pixel **51.45%** at epoch 105; low-frequency **78.58%** at epoch 110; joint **51.45%** at epoch 105.
- Peak-to-final declines: pixel **8.89 pp**, low-frequency **2.18 pp**, and joint **8.89 pp**.
- Minimum losses occur at epoch 155 for clean (0.465), epoch 105 for pixel (1.287), and epoch 110 for low-frequency (0.648).

## Trajectory

- Pixel and joint robustness peak at epochs 105 and 105, then finish 8.89 pp and 8.89 pp below their peaks.
- Joint robustness nearly equals pixel robustness throughout, so pixel-PGD is the limiting attack for this model.
- Pixel robust loss falls to 1.287, then rises to 3.186 at epoch 200.

## Interpretation

- This run shows the classic robust-overfitting pattern: adversarial test accuracy deteriorates after its best checkpoint even as training continues.
- Checkpoint selection matters: the best pixel-robust checkpoint is preferable to the final checkpoint for the pixel threat model.

## Visualizations

- [Evaluation curves](po_eval_results_curves.pdf)
- [Training dynamics](po_train_results_curves.pdf)

The TensorBoard test-robust curve in the training-dynamics figure uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
