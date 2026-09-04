# Pixel-Only: Seed 44

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **82.32%**, pixel-PGD-20 robust accuracy is **35.35%**, low-frequency-PGD-20 robust accuracy is **72.97%**, and union robust accuracy is **35.35%**.
- Peak clean accuracy is 82.77% at epoch 160; peak pixel and union robustness is 44.43% at epoch 105; peak low-frequency robustness is 75.19% at epoch 105.
- From peak to epoch 200, pixel and union robustness each decline **9.08 pp**, while low-frequency robustness declines **2.22 pp**.

## Lines

- Clean accuracy improves from 56.27% to epoch 160, then remains close to that level through final checkpoint.
- Pixel and union robustness peak at epoch 105 and gradually decline to 35.35%.
- Low-frequency robustness peaks at same epoch and shows smallest low-frequency post-peak decline among 5 seeded pixel-only runs.

## What this indicates

- This run reproduces robust overfitting under pixel PGD, with a much larger pixel/union decline than low-frequency decline.
- It still has strong final low-frequency robustness, but joint robustness remains constrained by pixel-PGD performance.
