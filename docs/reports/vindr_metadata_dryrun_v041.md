# VinDr/VinBigData Metadata Dry-Run v0.4.1

## Status

v0.4.1 records a real metadata dry-run for VinDr/VinBigData external-validation preparation.

This stage is not model inference, not model training, and not external-validation metrics.

## Objective

The dry-run verified that real VinDr/VinBigData annotations can be matched to real mounted image files and converted into a conservative local/private evaluation manifest.

## Success Criteria

The dry-run was considered successful only because all required checks were non-empty:

| Check | Result |
|---|---:|
| Annotation CSV exists | yes |
| Image train directory exists | yes |
| Train JPG images | 15,000 |
| Matched train image IDs | 15,000 |
| Manifest rows | 3,082 |
| Labels rows | 3,082 |
| Metadata rows | 3,082 |

## Annotation Summary

| Field | Count |
|---|---:|
| Annotation rows | 67,914 |
| Unique annotation image IDs | 15,000 |
| Mapped label rows | 8,408 |

## Conservative Label Mapping

Only labels with direct conservative correspondence were mapped:

| VinBigData class | Project label |
|---|---|
| Atelectasis | Atelectasis |
| Cardiomegaly | Cardiomegaly |
| Pleural effusion | Pleural Effusion |
| Pneumothorax | Pneumothorax |

Pneumonia was intentionally not mapped because VinBigData does not provide a direct Pneumonia class.

## Manifest Label Counts

| Label | Positive count | Positive rate |
|---|---:|---:|
| Atelectasis | 186 | 0.0604 |
| Cardiomegaly | 2,300 | 0.7463 |
| Pleural Effusion | 1,032 | 0.3348 |
| Pneumothorax | 96 | 0.0311 |

## Row Composition

| Row type | Count |
|---|---:|
| Any positive label | 3,082 |
| Multiple positive labels | 498 |
| All-zero rows | 0 |

## Privacy and Repository Boundary

The generated manifest, labels, metadata, image paths, raw images, and local/private Kaggle outputs were not committed to the repository.

Only this sanitized report is intended for GitHub.

## Next Stage

v0.4.2 should prepare reusable real external-validation inputs from the verified VinDr/VinBigData metadata workflow.

v0.4.1 does not run inference and does not report external-validation performance metrics.
