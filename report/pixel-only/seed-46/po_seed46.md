# Pixel-Only: Seed 46

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **81.54%**, pixel-PGD-20 robust accuracy is **34.74%**, low-frequency-PGD-20 robust accuracy is **72.43%**, and union robust accuracy is **34.74%**.
- Peak clean accuracy is 82.66% at epoch 155; peak pixel and union robustness is 44.29% at epoch 105; peak low-frequency robustness is 75.83% at epoch 115.
- From peak to epoch 200, pixel and union robustness each decline **9.55 pp**, while low-frequency robustness declines **3.40 pp**.

## Lines

- Clean accuracy rises from 59.47% to epoch 155, then declines modestly at end.
- Pixel and union robustness peak at epoch 105 and gradually fall to 34.74% by epoch 200.
- Low-frequency robustness peaks at epoch 115 and remains much higher than pixel robustness through final checkpoint.

## What this indicates

- This seed shows a clear, nearly 10-point post-peak drop in pixel and joint robustness.
- Its robust-overfitting timing matches other seeded pixel-only runs, supporting consistency of five-seed control.
