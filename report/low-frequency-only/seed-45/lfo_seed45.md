# Low-Frequency-Only: Seed 45

## Numbers

- The run evaluates 40 checkpoints from epoch 5 through epoch 200.
- At epoch 200, clean accuracy is **88.28%** and low-frequency-PGD-20 robust accuracy is **84.28%**.
- Peak clean accuracy is 88.45% and peak low-frequency robust accuracy is 84.60%, both at epoch 185; low-frequency robustness declines **0.32 pp** by epoch 200.
- Pixel-PGD-20 and union robust accuracy both finish at **0.00%**; their maximum is only **0.01%** at epoch 10.

## Lines

- Clean accuracy rises from 66.27% at epoch 5 to a late peak, then stays near that level through epoch 200.
- Low-frequency robust accuracy rises from 61.71% to 68.49% at epoch 100 and 78.89% at epoch 150 before its late peak at epoch 185.
- Pixel and union robustness remain essentially zero for entire run.

## What this indicates

- This seed retains high low-frequency robustness at end of training, with only a small late decline.
- The result reinforces that low-frequency-only training protects trained-against domain but provides no meaningful pixel-PGD or joint robustness.
