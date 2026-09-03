# Low-Frequency-Only: Overall Results

## Numbers

- This summary collects overall results from 5 runs (seeds 42–46) across 40 checkpoints, evaluated every 5 epochs from epoch 5 through 200.
- At epoch 200, mean clean accuracy is **88.11% ± 0.33 pp** and mean low-frequency-PGD-20 robust accuracy is **84.50% ± 0.56 pp**.
- The epoch-200 means are also highest overall clean and low-frequency robust accuracies observed.
- Pixel-PGD-20 and union robust accuracy finish at **0.00%**; their overall maximum is only **0.01%** at epoch 100.
- At epoch 200, mean clean test loss finishes at **0.37 ± 0.01** and mean low-frequency robust test loss reaches its minimum of **0.51 ± 0.01**, while pixel-PGD-20 test loss finishes at **12.39 ± 0.36**.

## Lines

- Mean clean accuracy rises from **66.15%** at epoch 5 to **88.11%** at epoch 200.
- Mean low-frequency robust accuracy rises from **61.80%** to **84.50%**, including **68.51%** at epoch 100 and **79.57%** at epoch 150.
- The low-frequency robustness curve remains high late in training and reaches its overall maximum at final checkpoint rather than falling off.
- The pixel-PGD-20 and union-robustness curves stay essentially on zero line throughout training.
- Mean clean test loss and low-frequency robust test loss drop steadily throughout training (clean loss: 1.07 → 0.37; low-frequency loss: 1.13 → 0.51) without late-stage loss explosion or divergence.

## What this indicates

- Low-frequency-only adversarial training provides strong, consistent protection against low-frequency attack used for training and evaluation.
- That protection does not transfer to unrestricted pixel-space PGD; because pixel robustness is near zero, robustness to both attacks simultaneously is also near zero.
- The overall curve does not show robust overfitting under low-frequency threat model by epoch 200: mean low-frequency robustness is highest at final checkpoint.
- The absence of test loss divergence corroborates the accuracy findings: low-frequency-only adversarial training does not exhibit robust overfitting under the low-frequency threat model through epoch 200.

## Visualizations

- Evaluation Curves: [`lfo_eval_results_curves.png`](lfo_eval_results_curves.png)
- Training Dynamics: [`lfo_train_results_curves.png`](lfo_train_results_curves.png)

