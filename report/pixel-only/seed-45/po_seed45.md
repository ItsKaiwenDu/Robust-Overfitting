# Pixel-Only: Seed 45

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **81.95%**, pixel-PGD-20 robust accuracy is **34.57%**, low-frequency-PGD-20 robust accuracy is **72.56%**, and union robust accuracy is **34.57%**.
- Peak clean accuracy is 82.99% at epoch 155; peak pixel and union robustness is 44.08% at epoch 105; peak low-frequency robustness is 75.93% at epoch 120.
- From peak to epoch 200, pixel and union robustness each decline **9.51 pp**, while low-frequency robustness declines **3.37 pp**.

## Lines

- Clean accuracy rises from 56.56% to epoch 155, then falls modestly by end.
- Pixel and union robustness peak at epoch 105 and decline to 34.57% at epoch 200.
- Low-frequency robustness peaks later, at epoch 120, and declines more gradually than pixel robustness.

## What this indicates

- The early pixel/union peak and subsequent decline are consistent with robust overfitting under pixel-space threat model.
- The later and smaller low-frequency decline shows that severity and timing of effect differ by evaluation attack.
