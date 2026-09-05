# Mixed-Domain: Overall Results

## Numbers

- This summary collects overall results from 5 runs (seeds 42–46) across 40 checkpoints, evaluated every 5 epochs from epoch 5 through 200.
- At epoch 200, mean clean accuracy is **88.34% ± 3.34 pp**, pixel-PGD-20 robust accuracy is **17.80% ± 12.79 pp**, low-frequency-PGD-20 robust accuracy is **82.97% ± 3.93 pp**, and union robust accuracy is **17.80% ± 12.79 pp**.
- Mean pixel and union robustness peak at **37.93%** at epoch 85, then decline **20.13 pp** by epoch 200.
- Mean low-frequency robustness peaks at **84.39%** at epoch 195 and declines only **1.42 pp** by final checkpoint.
- Mean pixel robust test loss reaches an early minimum of **1.64 ± 0.02** at epoch 85 (matching peak union robustness), then diverges to **5.27 ± 2.72** by epoch 200, whereas mean low-frequency robust loss drops steadily to **0.53 ± 0.11** (and clean loss to **0.37 ± 0.11**).

## Lines

- Mean clean accuracy rises from 58.94% at epoch 5 and peaks at 89.94% at epoch 195 before ending at 88.34%.
- The low-frequency robustness line increases from 54.51% to 82.97% and remains high late in training.
- Pixel and union robustness rise early, but their overall lines fall from their epoch-85 peaks to 17.80% at epoch 200.
- The large final variation in pixel and union robustness across seeds (±12.79 pp) shows that their late-training behavior differs substantially by run.
- Pixel robust test loss exhibits severe instability and spikes after epoch 85 across runs, while low-frequency robust loss and clean test loss decline smoothly and remain tightly controlled through epoch 200.

## What this indicates

- Mixed-domain training maintains strong low-frequency robustness while also producing meaningful pixel robustness early in training.
- Pixel and joint robustness show pronounced robust overfitting overall, so an early checkpoint near pixel-robustness peak would be preferable if protection against both attacks is goal.
- The much smaller low-frequency decline shows that late-training effect is substantially stronger under pixel-space threat model.
- The divergence in pixel test loss alongside steady low-frequency loss confirms that robust overfitting in mixed-domain training is domain-specific, occurring heavily in the pixel perturbation space while frequency robustness remains well-behaved.

## Visualizations

- Evaluation Curves: [`mdo_eval_results_curves.png`](mdo_eval_results_curves.png)
- Training Dynamics: [`mdo_train_results_curves.png`](mdo_train_results_curves.png)
- Training Dynamics: [`mixed_train_results_curves.png`](mixed_train_results_curves.png)
