# Pixel-Only: Rice et al. Reproduction Baseline

## Numbers

- This independent baseline run evaluates 40 checkpoints from epoch 5 through epoch 200. It is separate from, and not included in, five-seed pixel-only overall results.
- At epoch 200, clean accuracy is **82.22%** and pixel-PGD-20 robust accuracy is **36.18%**.
- Clean accuracy peaks at 82.98% at epoch 170, while pixel-PGD-20 robust accuracy peaks earlier at **45.91%** at epoch 105.
- Pixel robustness declines **9.73 pp** from its epoch-105 peak to final checkpoint.
- Pixel-PGD-20 test loss reaches its minimum of **1.48** at epoch 105 (matching peak robust accuracy), then surges to **3.75** by epoch 200 (+2.27 increase, +153%), while clean test loss remains low at **0.62** (down from 0.80 at epoch 100).

## Lines

- Clean accuracy rises from 58.04% at epoch 5 to its peak at epoch 170, then ends slightly lower at 82.22%.
- Pixel-PGD-20 robust accuracy rises from 34.24% to 45.91% at epoch 105, falls to 39.14% at epoch 150, and ends at 36.18%.
- Robust test loss drops to 1.48 at epoch 105 before rising monotonically to 3.75 at epoch 200, displaying the classic U-shaped robust overfitting trajectory while clean loss continues downward.
- This older baseline CSV records clean and pixel-PGD-20 results only; it does not contain low-frequency or union metrics.

## What this indicates

- The run reproduces robust-overfitting pattern reported by Rice et al.: pixel-PGD test robustness drops substantially after its early peak while clean accuracy continues improving to a later epoch.
- The robust test loss explosion confirms that robust overfitting is not just an accuracy metric artifact, but reflects growing cross-entropy error on adversarial examples late in training.
- The result serves as an independent reference point for later five-seed pixel-only control, rather than an additional seed in that overall summary.

## Visualizations

- Evaluation Curves: [`po_eval_results_curves.png`](po_eval_results_curves.png)
- Training Dynamics: [`po_train_results_curves.png`](po_train_results_curves.png)

