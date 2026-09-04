# Mixed-Domain: Seed 43

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **89.87%**, pixel-PGD-20 robust accuracy is **16.48%**, low-frequency-PGD-20 robust accuracy is **84.34%**, and union robust accuracy is **16.48%**.
- Peak clean accuracy is 90.71% and peak low-frequency robustness is 86.16%, both at epoch 165; their final declines are **0.84 pp** and **1.82 pp**.
- Pixel and union robustness peak at 43.32% at epoch 110, then decline **26.84 pp** by epoch 200.

## Lines

- Clean and low-frequency robustness rise to epoch 165 and remain near their peaks at end.
- Pixel and union robustness reach 37.81% at epoch 100, fall near zero at epoch 150, and end at 16.48%.
- The broad swings in pixel and union robustness contrast with stable late low-frequency line.

## What this indicates

- This seed preserves clean and low-frequency performance well, but its pixel and joint robustness exhibit strong late-training instability and decline.
- Its best checkpoint for low-frequency robustness is not same as its best checkpoint for pixel or joint robustness.
