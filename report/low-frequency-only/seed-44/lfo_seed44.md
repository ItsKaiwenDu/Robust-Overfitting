# Low-Frequency-Only: Seed 44

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **87.60%** and low-frequency-PGD-20 robust accuracy is **83.88%**.
- Peak clean accuracy is 87.96% and peak low-frequency robust accuracy is 84.26%, both at epoch 160; low-frequency robustness declines **0.38 pp** by epoch 200.
- Pixel-PGD-20 and union robust accuracy both finish at **0.00%**; their maximum is only **0.01%** at epoch 35.

## Lines

- Clean accuracy rises from 65.78% at epoch 5 to its high point near epoch 160, then ends slightly lower at 87.60%.
- Low-frequency robust accuracy rises from 61.18% to 69.37% at epoch 100 and 80.42% at epoch 150 before small late decline.
- Pixel and union robustness remain on zero line apart from isolated 0.01% measurement.

## What this indicates

- This seed achieves high low-frequency robustness, with only mild late-training degradation rather than a severe robust-overfitting drop.
- It remains almost entirely vulnerable to pixel-space PGD, so its strong low-frequency result does not produce joint robustness.
