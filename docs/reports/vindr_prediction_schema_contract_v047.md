# VinDr/VinBigData Prediction Schema Contract v0.4.7

## Status

v0.4.7 adds a prediction schema validator for future VinDr/VinBigData inference outputs.

This stage is not model training, not model inference, and not external-validation metrics.

## Objective

The goal of v0.4.7 is to define and test the prediction-file contract before real VinDr/VinBigData model inference is attempted.

## Prediction Contract

A valid prediction CSV must contain:

- `sample_id`
- one numeric probability column per target label
- values in the closed interval `[0, 1]`
- no duplicate `sample_id` rows
- sample IDs that exist in both the prepared manifest and labels

By default, the validator requires exact sample coverage. A controlled subset mode is available for future small inference dry-runs.

## Added Components

| Component | Purpose |
|---|---|
| `scripts/validate_vindr_prediction_schema.py` | Validates future prediction CSV files |
| `tests/test_medshiftlab_vindr_prediction_schema.py` | Tests valid and invalid prediction schema cases |

## Claim Boundary

The validator does not generate predictions, load a model, run inference, compute metrics, or establish external validation.

## Repository Boundary

No raw images, local manifests, local labels, prediction files, metrics files, Kaggle paths, or private runtime outputs are committed.

Only sanitized code, tests, README text, and this report are tracked.

## Next Stage

v0.4.8 should run a controlled subset prediction-schema validation using a local/private prediction file.

Real model inference and metrics must remain separate until a model adapter and prediction generation path are explicitly added.
