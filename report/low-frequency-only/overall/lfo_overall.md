# Low-Frequency-Only: Overall Results

## Numbers

- Five runs (seeds 42-46), each with 40 checkpoints evaluated every five epochs from epoch 5 through epoch 200.
- Epoch 200 mean +/- sample SD: clean **94.28% +/- 0.22 pp**, pixel-PGD-20 **0.00% +/- 0.00 pp**, low-frequency-PGD-20 **92.70% +/- 0.28 pp**, and joint robustness **0.00% +/- 0.00 pp**.
- Mean peaks: clean **94.28%** at epoch 200; pixel **0.00%** at epoch 5; low-frequency **92.74%** at epoch 195; joint **0.00%** at epoch 5.
- Mean peak-to-final declines: pixel **0.00 pp**, low-frequency **0.05 pp**, and joint **0.00 pp**.
- Mean minimum losses: clean **0.210 +/- 0.003** at epoch 105; pixel **18.491 +/- 1.301** at epoch 5; low-frequency **0.266 +/- 0.005** at epoch 105.

## Trajectory

- Mean low-frequency robustness peaks at epoch 195 and declines only 0.05 pp by epoch 200.
- Pixel and joint robustness are 0.00% at every evaluated checkpoint in every seed.
- Mean low-frequency robust loss reaches 0.266 at epoch 105 and ends at 0.318.

## Interpretation

- Low-frequency-only training produces strong, stable matched-domain robustness and no practically meaningful robust-overfitting decline in accuracy.
- Its complete lack of pixel-PGD and joint robustness demonstrates that robustness does not transfer automatically across these threat domains.

## Visualizations

- [Five-seed evaluation curves](lfo_eval.pdf)
- [Five-seed training dynamics](lfo_train.pdf)

Shaded bands show sample standard deviation across seeds. The TensorBoard test-robust curve uses pixel-PGD-10 in every mode; the training-robust curve uses the mode's active training attack.
