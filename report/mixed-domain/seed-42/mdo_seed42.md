# Mixed-Domain: Seed 42

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **82.38%**, pixel-PGD-20 robust accuracy is **39.72%**, low-frequency-PGD-20 robust accuracy is **75.98%**, and union robust accuracy is **39.72%**.
- Peak clean accuracy is 89.97% at epoch 180 and peak low-frequency robustness is 85.51% at epoch 180; both decline by **7.59 pp** and **9.53 pp**, respectively, by epoch 200.
- Pixel and union robustness peak at 41.74% at epoch 135 and decline only **2.02 pp** by final checkpoint.

## Lines

- Clean and low-frequency robustness improve through epoch 180, then show a noticeable late decline.
- Pixel and union robustness are unstable: they are near zero at epoch 100, rise to their peak at epoch 135, and finish at 39.72%.
- The final model retains strongest pixel and union robustness among mixed-domain seeds, despite decline in clean and low-frequency accuracy.

## What this indicates

- This run trades a sizable late loss in clean and low-frequency accuracy for relatively sustained pixel and joint robustness.
- Its sharply changing pixel/union line shows why selecting a checkpoint based on target threat model matters in mixed-domain training.
