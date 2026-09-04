# Mixed-Domain: Seed 46

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **90.06%**, pixel-PGD-20 robust accuracy is **13.94%**, low-frequency-PGD-20 robust accuracy is **84.61%**, and union robust accuracy is **13.94%**.
- Peak clean accuracy is 90.82% and peak low-frequency robustness is 86.78%, both at epoch 155; their final declines are **0.76 pp** and **2.17 pp**.
- Pixel and union robustness peak at 40.97% at epoch 125, then decline **27.03 pp** by epoch 200.

## Lines

- Clean and low-frequency robustness peak at epoch 155 and remain high at final checkpoint.
- Pixel and union robustness are 34.05% at epoch 100, nearly zero at epoch 150, and end at 13.94%.
- The low-frequency curve is much more stable than pixel and union curves late in training.

## What this indicates

- This seed achieves strong final clean and low-frequency results, but not stable pixel or joint robustness.
- Its large pixel/union decline is another case where early stopping would preserve substantially more joint robustness.
