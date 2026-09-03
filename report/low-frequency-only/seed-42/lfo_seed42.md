# Low-Frequency-Only: Seed 42

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **88.35%** and low-frequency-PGD-20 robust accuracy is **84.80%**.
- The final checkpoint is peak clean and low-frequency robust result for this seed; low-frequency robustness peaks at epoch 170 and remains 84.80% through epoch 200.
- Pixel-PGD-20 and union robust accuracy both finish at **0.00%**; their maximum is only **0.01%** at epoch 85.

## Lines

- Clean accuracy rises from 63.23% at epoch 5 to 88.35% at epoch 200.
- Low-frequency robust accuracy rises from 58.67% to 84.80%, passing 68.04% at epoch 100 and 77.05% at epoch 150.
- The late low-frequency robustness line is stable: there is no decline from its peak to final checkpoint.
- Pixel and union robustness remain on zero line aside from isolated 0.01% measurement.

## What this indicates

- This seed learns strong robustness to low-frequency threat model without late-training degradation.
- Its lack of pixel and union robustness shows that low-frequency robustness does not extend to unrestricted pixel-space attack.
