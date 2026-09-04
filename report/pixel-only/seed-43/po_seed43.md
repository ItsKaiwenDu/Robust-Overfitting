# Pixel-Only: Seed 43

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **82.00%**, pixel-PGD-20 robust accuracy is **35.16%**, low-frequency-PGD-20 robust accuracy is **72.73%**, and union robust accuracy is **35.16%**.
- Peak clean accuracy is 82.58% at epoch 155; peak pixel and union robustness is 44.86% at epoch 105; peak low-frequency robustness is 75.67% at epoch 105.
- From peak to epoch 200, pixel and union robustness each decline **9.70 pp**, while low-frequency robustness declines **2.94 pp**.

## Lines

- Clean accuracy rises from 58.17% to its epoch-155 high point, then ends slightly lower.
- Pixel and union robustness rise early, peak at epoch 105, and decline steadily to 35.16% at epoch 200.
- Low-frequency robustness follows same timing but has a much smaller late decline.

## What this indicates

- Pixel robustness shows clear robust overfitting, whereas low-frequency robustness remains comparatively stable.
- The final union result is limited by weaker pixel-PGD robustness.
