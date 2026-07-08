# CheXpert Safe-Subset R1 Baseline Report

## Status

This report records a local/private Kaggle safe-subset run using real CheXpert validation images and the TorchXRayVision baseline adapter.

This is not a full benchmark, not external validation, and not clinical validation.

## Execution environment

- Runtime: Kaggle Notebook
- Device: CUDA GPU
- Dataset: CheXpert validation split
- Image subset: 64 frontal validation images
- Model: TorchXRayVision DenseNet121
- Adapter: `torchxrayvision-adapter`
- Preprocessing version: `phase5-baseline-inference-v2`
- Normalization: `torchxrayvision`
- Image preprocessing:
  - output mode: grayscale
  - target size: 224 x 224
  - scale: TorchXRayVision-compatible `[-1024, 1024]`

## Label support

| Label | Negative | Positive |
|---|---:|---:|
| Atelectasis | 49 | 15 |
| Cardiomegaly | 50 | 14 |
| Pleural Effusion | 55 | 9 |
| Pneumonia | 60 | 4 |
| Pneumothorax | 61 | 3 |

## Aggregate metrics at threshold 0.5

| Metric | Value |
|---|---:|
| Mean AUROC | 0.7999289127474138 |
| Mean AUPRC | 0.43391043300391524 |
| Mean Brier Score | 0.3409154551855098 |
| Mean ECE | 0.48993989813261296 |
| Mean F1 | 0.23849916690252998 |
| Mean Sensitivity | 1.0 |
| Mean Specificity | 0.0 |

## Interpretation

The safe-subset run completed successfully and confirms that the real-image inference path works end-to-end after aligning the image normalization with TorchXRayVision expectations.

The AUROC result suggests useful ranking behavior on this small exploratory subset. However, threshold-based classification at the default threshold of 0.5 is not calibrated: sensitivity is 1.0 and specificity is 0.0, indicating all-positive threshold behavior on the evaluated labels.

Therefore, these numbers should be treated only as an exploratory smoke-baseline result. They must not be presented as a clinical result, production benchmark, or validated diagnostic performance.

## Reproducibility note

Raw CheXpert images, local paths, sample-level labels, prediction files, and Kaggle private output files were intentionally not committed to the repository.

Only this sanitized summary report is stored in GitHub.
