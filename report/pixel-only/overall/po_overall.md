# Pixel-Only: Overall Results

## Numbers

- Five runs (seeds 42-46), each with 40 checkpoints evaluated every five epochs from epoch 5 through epoch 200.
- Epoch 200 mean +/- sample SD: clean **84.40% +/- 0.09 pp**, pixel-PGD-20 **42.66% +/- 0.22 pp**, low-frequency-PGD-20 **76.48% +/- 0.26 pp**, and joint robustness **42.66% +/- 0.22 pp**.
- Mean peaks: clean **85.07%** at epoch 155; pixel **51.22%** at epoch 105; low-frequency **78.36%** at epoch 155; joint **51.22%** at epoch 105.
- Mean peak-to-final declines: pixel **8.56 pp**, low-frequency **1.88 pp**, and joint **8.56 pp**.
- Mean minimum losses: clean **0.469 +/- 0.003** at epoch 155; pixel **1.291 +/- 0.005** at epoch 105; low-frequency **0.652 +/- 0.004** at epoch 115.

## Trajectory

- Every seed reaches its pixel-robust peak at epoch 105; the aggregate falls 8.56 pp afterward.
- Mean pixel robust loss rises from 1.291 at epoch 105 to 3.223 at epoch 200.
- Joint and pixel robustness are effectively identical, establishing pixel-PGD as the binding threat in this condition.

## Interpretation

- The five seeds reproduce the qualitative robust-overfitting result: robust test performance peaks well before training ends and then declines consistently.
- Reporting both the best and final checkpoints is necessary; selecting only the final model understates the attainable pixel robustness by about 8.56 percentage points.

## Visualizations

- [Five-seed evaluation curves](po_eval.pdf)
- [Five-seed training dynamics](po_train.pdf)

Shaded bands show sample standard deviation across seeds. The TensorBoard test-robust curve uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
