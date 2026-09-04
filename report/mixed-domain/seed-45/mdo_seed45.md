# Mixed-Domain: Seed 45

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **89.29%**, pixel-PGD-20 robust accuracy is **12.40%**, low-frequency-PGD-20 robust accuracy is **84.43%**, and union robust accuracy is **12.40%**.
- Peak clean accuracy is 90.40% and peak low-frequency robustness is 86.10%, both at epoch 165; their final declines are **1.11 pp** and **1.67 pp**.
- Pixel and union robustness peak at 41.36% at epoch 125, then decline **28.96 pp** by epoch 200.

## Lines

- Clean and low-frequency robustness increase steadily and remain close to their peaks at end.
- Pixel and union robustness are nearly zero at epochs 100 and 150, despite reaching 41.36% at epoch 125.
- The final pixel and union results remain well below their earlier peak.

## What this indicates

- This seed keeps high low-frequency robustness but shows strong robust overfitting for pixel and joint robustness.
- The contrasting curves reinforce that late-training behavior depends on evaluation attack domain.
