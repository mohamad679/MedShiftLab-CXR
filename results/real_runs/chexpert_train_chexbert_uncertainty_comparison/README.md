# CheXpert CheXbert Uncertainty Strategy Metadata Analysis

This directory contains a real metadata analysis, not model inference. The input was the local CheXpert `train_cheXbert.csv` file. Raw dataset files and images are not committed to this repository.

## Dataset summary

- `n_records = 223414`
- `n_patients = 64540`
- `n_records_without_patient_id = 0`

## Uncertainty strategies

- `U-ignore`: exclude uncertain labels from label-level target calculations.
- `U-zero`: map uncertain labels to 0.
- `U-one`: map uncertain labels to 1.
- `U-soft`: map uncertain labels to a soft target of 0.5.

## Outputs

- `chexpert_uncertainty_strategy_label_summary.csv`
- `chexpert_uncertainty_strategy_dataset_summary.csv`

Comparison figures are stored under `figures/chexpert_train_chexbert_uncertainty_comparison/`:

- `mean_target_by_uncertainty_strategy.png`
- `positive_prevalence_by_uncertainty_strategy.png`
- `soft_counts_by_label.png`

## Interpretation

Uncertainty handling can strongly change target distributions. For Pneumonia, `mean_target` is approximately 0.176 with `U-zero`, 0.880 with `U-one`, and 0.528 with `U-soft`.

## Non-claims

These outputs are not a model benchmark and include no image inference, clinical validation, deployment claim, or state-of-the-art (SOTA) claim.
