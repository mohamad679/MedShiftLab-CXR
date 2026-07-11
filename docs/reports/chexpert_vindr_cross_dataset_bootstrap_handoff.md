# CheXpert-VinDr Cross-Dataset Bootstrap Handoff

## Status

`blocked_missing_chexpert_sample_level_artifacts`

The reusable cross-dataset bootstrap implementation is complete and merged,
but the required CheXpert inputs are unavailable locally. No real
cross-dataset bootstrap run may be performed or claimed.

## Objective

Use the completed reusable tooling to calculate retrospective bootstrap
confidence intervals for CheXpert and VinDr metrics, together with
CheXpert-to-VinDr delta intervals, from aligned sample-level artifacts.

## Available artifacts

VinDr sample-level prediction and label artifacts are available privately:

- `vindr_standardized_predictions_v052.json`
- `vindr_labels.csv`

The real CheXpert 202-image reference run is represented locally only by
sanitized aggregate reports.

## Missing prerequisite

CheXpert sample-level standardized predictions and the aligned CheXpert label
table for the 202-image reference cohort are unavailable. Rebuilding the
missing artifact would require recreating the original validation subset and
rerunning inference in Kaggle or Colab.

## Why aggregate reports are insufficient

Bootstrap resampling requires paired, sample-level prediction and label data.
Aggregate evaluation metrics contain neither the individual observations nor
the sample identity needed to resample and validate alignment. They cannot be
used to reconstruct bootstrap confidence intervals or cross-dataset delta
intervals.

## Execution decision

Do not execute a real cross-dataset bootstrap run. No bootstrap JSON, CSV,
confidence intervals, or delta intervals were generated. No inference was
rerun, and no metrics were reconstructed from aggregate reports.

## Exact requirements to unblock

- A standardized CheXpert prediction JSON or compatible CSV.
- An aligned CheXpert label CSV.
- The same 202-image reference cohort or a newly documented replacement cohort.
- Matching model identity and preprocessing provenance.
- Successful sample-ID alignment validation.
- An authorized private local or Kaggle run.

## Claim boundary

No clinical, deployment, prospective-validation, or statistical-significance
claim is made. This handoff does not claim that the CheXpert artifact exists or
that a real cross-dataset bootstrap result has been produced.

## Repository boundary

Private artifacts must not be committed. This repository retains only
sanitized documentation; it does not include raw data, private sample-level
predictions, aligned labels, runtime outputs, or absolute private paths.

## Next authorized action

After all prerequisites are supplied and a private run is authorized, validate
sample-ID alignment and run the existing cross-dataset bootstrap workflow
against the documented CheXpert cohort and the available VinDr artifacts. Until
then, preserve the blocked status and make no real-run claim.
