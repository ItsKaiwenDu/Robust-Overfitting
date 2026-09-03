# Low-Frequency-Only: Seed 46

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **87.94%** and low-frequency-PGD-20 robust accuracy is **84.22%**.
- The final checkpoint is peak clean and low-frequency robust result for this seed.
- Pixel-PGD-20 and union robust accuracy both finish at **0.00%**; their maximum is only **0.01%** at epoch 15.

## Lines

- Clean accuracy rises from 68.78% at epoch 5 to 87.94% at epoch 200.
- Low-frequency robust accuracy rises from 64.27% to 84.22%, passing 67.69% at epoch 100 and 78.78% at epoch 150.
- The low-frequency robustness line is high late in training and ends at its maximum.
- Pixel and union robustness stay essentially at zero throughout run.

## What this indicates

- This seed shows sustained improvement under low-frequency threat model through final checkpoint.
- As with other seeds, absence of pixel and union robustness demonstrates no useful transfer to unrestricted pixel-space attacks.
