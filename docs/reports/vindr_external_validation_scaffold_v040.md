# VinDr-CXR External Validation Scaffold v0.4.0

## Status

This report records the addition of a VinDr-CXR external-validation scaffold.

This is not an external validation result. No VinDr-CXR inference was run in this step. No clinical validation, fairness validation, or full benchmark was performed.

## Added files

The scaffold adds:

- `configs/evaluation/vindr_cxr_label_mapping.json`
- `scripts/prepare_vindr_external_validation_inputs.py`
- `tests/test_medshiftlab_vindr_external_validation_scaffold.py`

## Purpose

The goal is to prepare MedShiftLab-CXR for a future VinDr-CXR validation run by adding a local/manual preparation interface for:

- image-level label harmonization
- metadata preparation
- optional manifest preparation
- sanitized summary generation

The scaffold intentionally avoids downloading raw data or committing local/private outputs.

## Target label space

The initial overlap label space is:

| Target Label |
|---|
| Atelectasis |
| Cardiomegaly |
| Pleural Effusion |
| Pneumonia |
| Pneumothorax |

The mapping is intentionally conservative and should be reviewed before any validated external-performance claim.

## Example preparation command

```bash
python3 scripts/prepare_vindr_external_validation_inputs.py \
  --annotations-csv /path/to/vindr_annotations.csv \
  --metadata-csv /path/to/vindr_metadata.csv \
  --metadata-columns sex view_position age_bucket \
  --output-dir results/local_private_runs/vindr_external_validation/inputs
```

Optional manifest generation can be requested by adding an image root:

```bash
python3 scripts/prepare_vindr_external_validation_inputs.py \
  --annotations-csv /path/to/vindr_annotations.csv \
  --metadata-csv /path/to/vindr_metadata.csv \
  --metadata-columns sex view_position age_bucket \
  --image-root /path/to/vindr/images \
  --image-extensions .jpg .png .dicom .dcm \
  --output-dir results/local_private_runs/vindr_external_validation/inputs
```

## Outputs

The preparation script writes local/private files:

| Output | Purpose |
|---|---|
| `vindr_labels.csv` | MedShiftLab-compatible image-level labels |
| `vindr_metadata.csv` | Optional metadata keyed by `sample_id` |
| `vindr_manifest.csv` | Optional manifest if image root is provided |
| `vindr_prepare_summary.json` | Local preparation summary |

These outputs should stay local/private unless a sanitized summary is explicitly created.

## Claim boundary

This scaffold only prepares the repository for a future external-validation track. It does not establish external validity, diagnostic performance, subgroup fairness, or production readiness.

## Recommended next step

The next step should be a local/private VinDr-CXR dry-run preparation using a small subset of available VinDr metadata and annotations, followed by a manifest smoke test before real inference.
