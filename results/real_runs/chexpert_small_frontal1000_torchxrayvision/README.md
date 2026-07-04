# CheXpert small frontal-1000 TorchXRayVision run

These are derived summaries from a completed real Google Colab run over the first 1,000 frontal `CheXpert-v1.0-small` training images. The run loaded all 1,000 images and performed pretrained TorchXRayVision DenseNet inference with `densenet121-res224-all` weights on a Tesla T4. No training was performed.

- `qc_summary.json` records the image audit.
- `evaluation_label_metrics.csv` records the supplied label-wise metrics.
- `evaluation_report.json` records the run environment and evaluation policy.

Binary metrics use only targets equal to 0 or 1. Brier score and ECE use available soft targets after mapping uncertainty (`-1`) to `0.5`. Threshold metrics use `0.5`; no threshold tuning was performed.

This is subset-only research evaluation, not clinical validation, diagnostic deployment, or a state-of-the-art claim. Raw CheXpert images, sample grids, the source zip, and path-bearing prediction tables are intentionally excluded.
