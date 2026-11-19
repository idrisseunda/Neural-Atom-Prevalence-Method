# Neural Atom Prevalence (NAP)

Code and experiments for **"Neural Feature Governance: Extending Atom Prevalence"** (Seunda Ekwe, Shako, Fokoué), arXiv:2607.21671 — [paper](https://arxiv.org/html/2607.21671v1).

NAP is a Bayesian framework for neuron-level model selection in feedforward neural networks. It produces sparse networks with calibrated uncertainty, separating epistemic from aleatoric uncertainty.

## Method

NAP is a four-phase pipeline:

1. **Bayesian Lottery Ticket (BLT):** Iterative Magnitude Pruning on the dense network identifies a candidate sub-network.
2. **Soft variational training:** a Spike-and-Slab Independent Gaussian (SS-IG) model is trained on the ticket, with soft inclusion variables so no neuron is pruned during training.
3. **Poisson-Binomial layer-size selection:** the layer size is the mode of the layer-size distribution, and the neurons with the highest inclusion probabilities are kept.
4. **Bayesian fine-tuning** of the selected sparse network.

## Experiments

| Folder | Content |
|---|---|
| `Regression_experiment/` | Simulated nonlinear regression, UCI Concrete, UCI YearPredictionMSD |
| `Classification_experiment/` | MNIST classification |

Baselines: SS-IG (Jantre et al., 2023) and a dense variational BNN (VBNN).

## Selected results (from the paper)

| Dataset | Metric | NAP | SS-IG | VBNN |
|---|---|---|---|---|
| Concrete | RMSE | 6.32 ± 0.44 | 7.92 ± 0.68 | 7.34 ± 0.62 |
| Concrete | Nodes removed | 75% | 59% | none |
| MNIST | Accuracy | 97% | 97% | 99% |

- On simulated data, 95% prediction intervals reach 93.4% empirical coverage, and epistemic uncertainty is about 3–4% of predictive variance.
- On simulated data, NAP's RMSE is slightly higher than SS-IG and VBNN.

See the paper for the full tables and the remaining datasets.

## Limitations

- Mean-field variational posteriors tend to underestimate variance (overconfidence), visible in MNIST calibration.
- Out-of-distribution behaviour is not evaluated.
- Only feedforward networks are covered (no CNNs, transformers, RNNs or residual connections).
- The four-phase pipeline costs more than single-phase variational training.

## Requirements

Python 3 with PyTorch, NumPy and Matplotlib.

## Citation

```
I. K. Seunda Ekwe, P. T. Shako, E. P. Fokoué.
Neural Feature Governance: Extending Atom Prevalence.
arXiv:2607.21671, 2026.
```

## Contact

Idris Karel Seunda Ekwe — AIMS Cameroon
