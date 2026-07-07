# Experiment Card Template

## Identification

- Run ID:
- Date:
- Git commit:
- Protocol version:
- Operator:
- Claim status: exploratory / protocol-prep-only / confirmatory-local / do-not-claim

## Dataset

- Dataset name:
- Cohort definition:
- Source metadata version:
- Manifest path:
- Split name:
- Patient-disjoint check status:

## Model adapter

- Adapter key:
- Adapter implementation status:
- Model name:
- Model version / weights identifier:
- Real backend used: yes / no
- Manual-only run: yes / no

## Preprocessing

- Preprocessing version:
- Output mode:
- Target size:
- Normalization:
- Additional preprocessing notes:

## Prediction schema

- Prediction schema version:
- Label names:
- Prediction artifact path:
- Prediction artifact format: JSON / CSV

## Label mapping

- Ontology reference:
- Dataset harmonization reference:
- Label-table path:
- Included labels:
- Excluded source labels:

## Uncertainty strategy

- Strategy:
- Soft value if applicable:
- Strategy rationale:

## Evaluation

- Threshold policy:
- Metrics requested:
- Calibration bins:
- Evaluation report path:
- Label metrics CSV path:

## Calibration

- Calibration fitting performed: yes / no
- If yes, fit dataset and split:
- Calibration notes:

## Subgroup dimensions

- Requested subgroup columns:
- Minimum subgroup size:
- Missing subgroup handling:

## Robustness and bootstrap

- Robustness report path:
- Bootstrap iterations:
- Bootstrap metrics:
- Bootstrap resampling unit:
- Baseline comparison report path:

## Artifacts

- Local private output root:
- Additional figures/tables:
- Checksums or manifest notes:

## Limitations

- Dataset limitations:
- Label limitations:
- Missing dependencies or skipped steps:
- Reasons this run should not be over-interpreted:

## Claim boundary

- Allowed statement for this run:
- Disallowed statement for this run:

## Review checklist

- [ ] No raw data committed
- [ ] No private absolute paths committed
- [ ] No credentials committed
- [ ] No unsupported performance claim added
- [ ] Outputs stored in intended private location
- [ ] README/docs unchanged unless intentionally updated
