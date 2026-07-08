# VinDr/VinBigData Real Input Workflow Run v0.4.3

## Status

v0.4.3 records a real Kaggle execution of the hardened VinDr/VinBigData external-validation input-preparation workflow introduced in v0.4.2.

This stage is not model training, not model inference, and not external-validation metrics.

## Objective

The goal of v0.4.3 was to run the official repository input-preparation script against real mounted VinDr/VinBigData annotations and real mounted CXR image files.

The run verified that the reusable workflow can produce local/private labels, manifest, and summary outputs from real data.

## Execution Result

| Check | Result |
|---|---:|
| Repository commit used | 6121c29 |
| Input paths available | yes |
| Workflow completed | yes |
| Unique annotation image IDs | 15,000 |
| Manifest rows | 15,000 |
| Images found | 15,000 |
| Images missing | 0 |
| Require images enabled | yes |
| Minimum images found threshold | 1 |

## Label Counts

| Label | Positive count |
|---|---:|
| Atelectasis | 186 |
| Cardiomegaly | 2,300 |
| Pleural Effusion | 1,032 |
| Pneumonia | 0 |
| Pneumothorax | 96 |

Pneumonia remains zero because VinBigData does not provide a direct Pneumonia class in the conservative mapping used by this project.

## Output Boundary

The workflow generated local/private runtime outputs:

- labels CSV
- manifest CSV
- summary JSON

These files are not committed to Git.

## Validation Boundary

v0.4.3 verifies real input preparation only.

It does not:

- train a model
- run image inference
- compute AUROC, AUPRC, F1, sensitivity, specificity, or calibration metrics
- establish clinical validation
- establish fairness validation
- establish a completed external benchmark

## Privacy Boundary

No raw images, local manifests, local labels, Kaggle paths, prediction files, or private runtime outputs are included in the repository.

Only this sanitized report is tracked.

## Next Stage

v0.4.4 should prepare the inference-readiness handoff for VinDr/VinBigData external validation.

Inference must remain separate from this input-preparation stage.
