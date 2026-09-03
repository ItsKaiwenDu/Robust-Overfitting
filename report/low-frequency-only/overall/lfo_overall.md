# Low-Frequency-Only: Overall Results

## Numbers

- This summary collects overall results from 5 runs (seeds 42–46) across 40 checkpoints, evaluated every 5 epochs from epoch 5 through 200.
- At epoch 200, mean clean accuracy is **88.11% ± 0.33 pp** and mean low-frequency-PGD-20 robust accuracy is **84.50% ± 0.56 pp**.
- The epoch-200 means are also highest overall clean and low-frequency robust accuracies observed.
- Pixel-PGD-20 and union robust accuracy finish at **0.00%**; their overall maximum is only **0.01%** at epoch 100.

## Lines

- Mean clean accuracy rises from 66.15% at epoch 5 to 88.11% at epoch 200.
- Mean low-frequency robust accuracy rises from 61.80% to 84.50%, including 68.51% at epoch 100 and 79.57% at epoch 150.
- The low-frequency robustness curve remains high late in training and reaches its overall maximum at final checkpoint rather than falling off.
- The pixel-PGD-20 and union-robustness curves stay essentially on zero line throughout training.

## What this indicates

- Low-frequency-only adversarial training provides strong, consistent protection against low-frequency attack used for training and evaluation.
- That protection does not transfer to unrestricted pixel-space PGD; because pixel robustness is near zero, robustness to both attacks simultaneously is also near zero.
- The overall curve does not show robust overfitting under low-frequency threat model by epoch 200: mean low-frequency robustness is highest at final checkpoint.
