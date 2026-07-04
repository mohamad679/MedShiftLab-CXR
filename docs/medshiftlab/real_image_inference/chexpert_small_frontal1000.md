# Real CheXpert image inference: frontal-1000

## Scope

This run loaded real CheXpert images and ran real pretrained TorchXRayVision inference. It used the first 1,000 frontal training images listed in `CheXpert-v1.0-small/train.csv`, from 191,027 available frontal training rows. A separate first-20 frontal smoke subset was checked before the full subset. No model training occurred.

The execution environment was Google Colab with a Tesla T4, PyTorch `2.11.0+cu128`, TorchXRayVision `1.5.2`, and the TorchXRayVision DenseNet `densenet121-res224-all` weights on CUDA. The completed run reported a prediction output shape of `(1000, 20)`.

## Reproduction

Open `notebooks/chexpert_torchxrayvision_frontal_subset_colab.ipynb` in Colab. Set the Google Drive input and output paths in its configuration cell. The notebook verifies the source archive, extracts metadata and deterministic image subsets, performs QC, runs inference, evaluates the five mapped labels, creates metric-only plots, and saves outputs back to Drive.

The notebook calls the repository scripts so each stage can also be run independently:

- `scripts/extract_chexpert_image_subset.py`
- `scripts/audit_chexpert_images.py`
- `scripts/run_torchxrayvision_inference.py`
- `scripts/evaluate_chexpert_image_predictions.py`
- `scripts/plot_chexpert_image_inference_results.py`

All paths are command-line parameters. Prediction output stores relative image paths; it rejects absolute or parent-traversing manifest paths.

## Evaluation policy

The evaluated labels are Atelectasis, Cardiomegaly, Pleural Effusion, Pneumonia, and Pneumothorax. Pleural Effusion maps to TorchXRayVision's `pred_Effusion`; the other labels map to equivalently named `pred_` columns. AUROC, AUPRC, F1, sensitivity, and specificity use only labels equal to 0 or 1. Brier score and ECE use available labels with uncertainty (`-1`) mapped to `0.5`. Missing labels are omitted. Threshold metrics use `0.5`, with no threshold tuning.

The committed metrics are transcribed from the completed Colab execution. Cloud inference was not rerun locally and no new results were invented.

## Limitations and data boundary

This is a subset-only evaluation. It is not clinical validation, is not suitable for diagnostic deployment, and makes no state-of-the-art claim. Its label balance is highly uneven; in particular, some binary evaluations have few negative examples.

Raw images are not committed. Sample X-ray grids are not committed. Prediction tables containing absolute paths are not committed. The source zip is not committed. The repository contains only QC summaries, aggregate metric tables, and metric-only figures.
