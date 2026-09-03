# Low-Frequency-Only: Seed 43

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **88.38%** and low-frequency-PGD-20 robust accuracy is **85.32%**.
- The final checkpoint is peak low-frequency robust result for this seed; peak clean accuracy is 88.51% at epoch 170.
- Pixel-PGD-20 and union robust accuracy both finish at **0.00%**; their maximum is only **0.02%** at epoch 100.

## Lines

- Clean accuracy rises from 66.69% at epoch 5 to 88.38% at epoch 200.
- Low-frequency robust accuracy rises from 63.18% to 85.32%, passing 68.94% at epoch 100 and 82.69% at epoch 150.
- Low-frequency robustness continues improving through final checkpoint, while clean accuracy stays close to its peak.
- Pixel and union robustness remain essentially flat at zero throughout run.

## What this indicates

- This is strongest final low-frequency robust result among 5 seeds.
- The near-zero pixel and union curves again show that robustness is specific to restricted low-frequency attack, not broader pixel-space threat model.
