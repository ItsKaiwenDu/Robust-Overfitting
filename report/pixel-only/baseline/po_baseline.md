# Pixel-Only: Rice et al. Reproduction Baseline

## Numbers

- This independent baseline run evaluates 40 checkpoints from epoch 5 through epoch 200. It is separate from, and not included in, five-seed pixel-only overall results.
- At epoch 200, clean accuracy is **82.22%** and pixel-PGD-20 robust accuracy is **36.18%**.
- Clean accuracy peaks at 82.98% at epoch 170, while pixel-PGD-20 robust accuracy peaks earlier at **45.91%** at epoch 105.
- Pixel robustness declines **9.73 pp** from its epoch-105 peak to final checkpoint.

## Lines

- Clean accuracy rises from 58.04% at epoch 5 to its peak at epoch 170, then ends slightly lower at 82.22%.
- Pixel-PGD-20 robust accuracy rises from 34.24% to 45.91% at epoch 105, falls to 39.14% at epoch 150, and ends at 36.18%.
- This older baseline CSV records clean and pixel-PGD-20 results only; it does not contain low-frequency or union metrics.

## What this indicates

- The run reproduces robust-overfitting pattern reported by Rice et al.: pixel-PGD test robustness drops substantially after its early peak while clean accuracy continues improving to a later epoch.
- The result serves as an independent reference point for later five-seed pixel-only control, rather than an additional seed in that overall summary.
