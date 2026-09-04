# Pixel-Only: Seed 42

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **82.22%**, pixel-PGD-20 robust accuracy is **36.28%**, low-frequency-PGD-20 robust accuracy is **73.01%**, and union robust accuracy is **36.28%**.
- Peak clean accuracy is 82.98% at epoch 170; peak pixel and union robustness is 45.98% at epoch 105; peak low-frequency robustness is 76.23% at epoch 105.
- From peak to epoch 200, pixel and union robustness each decline **9.70 pp**, while low-frequency robustness declines **3.22 pp**.

## Lines

- Clean accuracy rises from 58.04% and reaches its high point later than every robustness metric.
- Pixel and union robustness peak at epoch 105, then decline to 36.28% by epoch 200.
- Low-frequency robustness also peaks at epoch 105, but remains substantially higher than pixel robustness through final checkpoint.

## What this indicates

- This seed shows expected pixel-space robust-overfitting pattern: pixel and joint robustness peak early and decline while clean accuracy continues improving.
- The shared pixel and union values show that pixel-PGD is limiting attack for joint robustness in this run.
