# Pixel-Only: Overall Results

## Numbers

- This summary collects overall results from 5 runs (seeds 42–46) across 40 checkpoints, evaluated every 5 epochs from epoch 5 through 200. The separate Rice et al. reproduction is documented in `baseline/po_baseline.md` and is not included here.
- At epoch 200, mean clean accuracy is **82.01% ± 0.30 pp**, pixel-PGD-20 robust accuracy is **35.22% ± 0.67 pp**, low-frequency-PGD-20 robust accuracy is **72.74% ± 0.25 pp**, and union robust accuracy is **35.22% ± 0.67 pp**.
- Mean pixel and union robustness peak at **44.73%** at epoch 105, then decline **9.51 pp** by epoch 200.
- Mean low-frequency robustness peaks at **75.44%** at epoch 105 and declines **2.70 pp** by final checkpoint.
- Mean pixel robust test loss reaches its minimum of **1.52 ± 0.03** at epoch 105 (matching peak robust accuracy), then surges to **3.84 ± 0.07** by epoch 200 (+2.32 increase, +153%), while clean test loss remains low at **0.63 ± 0.01** (down from 0.81 at epoch 100).

## Lines

- Mean clean accuracy rises from 57.70% at epoch 5, peaks at 82.73% at epoch 155, and ends at 82.01%.
- Pixel and union robustness rise early, peak at epoch 105, then gradually decline to 35.22%.
- Low-frequency robustness rises from 53.20% to 75.44% at epoch 105, then declines more mildly to 72.74%.
- The small final standard deviations across all metrics show that 5 seeded runs have closely matched late-training results.
- Pixel-PGD-20 test loss shows clear divergence after epoch 105: it drops to 1.52 at the first learning-rate decay, then explodes steadily up to 3.84 at epoch 200, mirroring the decline in robust accuracy.

## What this indicates

- Pixel-only training produces reproducible robust overfitting under pixel-PGD-20: test robustness peaks well before final checkpoint and loses 9.51 pp by epoch 200.
- Low-frequency robustness is higher and declines much less, while union robustness is determined by weaker pixel-PGD result.
- An early checkpoint near epoch 105 would preserve substantially more pixel and joint robustness than final checkpoint.
- The sharp divergence between decreasing clean loss and exploding robust loss reproduces the hallmark signature of robust overfitting described by Rice et al. (2020).

## Visualizations

- Evaluation Curves: [`po_eval_results_curves.png`](po_eval_results_curves.png)
- Training Dynamics: [`po_train_results_curves.png`](po_train_results_curves.png)

