# Mixed-Domain: Seed 44

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **90.08%**, pixel-PGD-20 robust accuracy is **6.48%**, low-frequency-PGD-20 robust accuracy is **85.48%**, and union robust accuracy is **6.48%**.
- Peak clean accuracy is 90.17% at epoch 180 and peak low-frequency robustness is 85.57% at epoch 115; each is within **0.09 pp** of its final value.
- Pixel robustness peaks at 42.15% and union robustness at 42.14%, both at epoch 155; they decline **35.67 pp** and **35.66 pp** by epoch 200.

## Lines

- Clean and low-frequency robustness remain essentially flat after their high points.
- Pixel and union robustness are high around epochs 100–155, then fall sharply to 6.48% at final checkpoint.
- This is largest pixel and union post-peak decline among mixed-domain seeds.

## What this indicates

- The run has stable clean and low-frequency performance but severe robust overfitting under pixel and joint threat models.
- An epoch near 155 would be far more suitable than final checkpoint for pixel-space or joint robustness.
