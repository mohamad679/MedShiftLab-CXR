# MedShiftLab-CXR Interview Talking Points

## 30-Second Project Pitch

MedShiftLab-CXR is a compute-aware research project for testing how a fixed pretrained chest X-ray model behaves under label uncertainty and dataset shift. I built the data contracts, uncertainty handling, inference, evaluation, calibration, bootstrap, subgroup, and reporting layers; ran DenseNet121 on a 202-image CheXpert reference set and 15,000 VinDr/VinBigData images; and documented the resulting cross-dataset behavior without claiming clinical validation.

## 2-Minute Project Explanation

Pretrained medical-imaging models can look strong on one dataset but behave differently when annotation rules, prevalence, patient populations, scanners, or acquisition protocols change. MedShiftLab-CXR makes those variables explicit and auditable.

The `v1.0.0` release uses TorchXRayVision DenseNet121 without training a new model. It includes a real CheXpert frontal validation reference evaluation on 202 images and full inference on 15,000 prepared VinDr/VinBigData images. The project computes discrimination, threshold, calibration, bootstrap, and subgroup outputs and compares behavior across the two datasets. Cardiomegaly and Pleural Effusion are the most stable labels; Atelectasis and Pneumothorax show weaker precision-sensitive and calibration behavior under external shift.

The completed VinDr work is retrospective external-dataset evaluation under a conservative four-label mapping. It is not clinical, prospective, fairness, regulatory, or deployment validation. The next scientific experiment is to compare separately fitted lightweight heads for `U-ignore`, `U-zero`, `U-one`, and `U-soft` while keeping the pretrained encoder and external protocol fixed.

## Technical Explanation

### Why CheXpert Is Internal and the External Candidates Are Separate

CheXpert is the development/internal dataset because it exposes uncertain labels and therefore supports controlled policy comparison, internal evaluation, and—if used—validation-only threshold selection or calibration fitting. VinDr/VinBigData has now been evaluated as the external dataset; MIMIC-CXR-JPG remains unevaluated. The baseline VinDr metrics remain the external reference, while the later VinDr threshold and calibration analyses are explicitly post-hoc exploratory analyses rather than a new confirmatory test.

### Why Uncertainty Strategies Matter

CheXpert uncertainty values are not ordinary negative or positive labels. U-ignore removes them from supervised metric computation, U-zero maps them to negative, U-one maps them to positive, and U-soft retains a probabilistic target. These choices change the effective target distribution and may change apparent discrimination, calibration, and failure patterns. They must therefore be explicit experimental variables, not hidden preprocessing defaults.

### Why Brier Score and ECE Matter

AUROC and AUPRC evaluate ranking but do not show whether predicted probabilities match observed outcomes. Brier score measures squared probability error, while ECE summarizes confidence-outcome gaps across bins. Both have limitations—Brier score reflects several aspects of prediction quality and ECE depends on binning—so they should be reported with discrimination metrics and reliability diagrams rather than interpreted alone.

### Why Adapters Are Separate from Evaluation

The adapter translates model-specific outputs into validated `PredictionRecord` and `PredictionBatch` contracts. Evaluation consumes only those stable contracts. This separation prevents model preprocessing or output conventions from leaking into metric logic, makes provenance and label coverage explicit, and allows model implementations to change without rewriting evaluation and reporting.

### Why External Results Cannot Be Used to Rewrite the Confirmatory Result

Using external results to choose the model, uncertainty policy, or confirmatory protocol would produce an optimistic estimate of generalization. The project therefore keeps the original threshold-0.5 VinDr metrics as the external reference and labels the later VinDr threshold and calibration analyses as post-hoc exploratory operating-point analyses.

## What Is Actually Implemented

- conservative CXR label ontology and CheXpert uncertainty handling
- validated CheXpert metadata schema and CSV loader
- validated VinDr-CXR metadata schema
- dataset registry and path-free local configuration template
- reusable JPEG/PNG loading and preprocessing tested with temporary synthetic images
- dataset summary generator
- AUROC, AUPRC, Brier score, ECE, F1, sensitivity, and specificity
- `EvaluationReport` schema and row-based evaluation table interface
- prediction schemas and `CXRModelAdapter` protocol
- dependency-safe optional TorchXRayVision output-mapping boundary
- prediction-to-evaluation bridge keyed by `image_id`
- complete JSON and label-wise CSV report exports
- in-memory and file-exporting experiment runners
- focused local MedShiftLab-CXR test runner
- real CheXpert reference evaluation on 202 frontal validation images
- full TorchXRayVision DenseNet121 inference on 15,000 VinDr/VinBigData images
- VinDr external metrics, threshold/calibration analysis, and CheXpert/VinDr comparison

## What Is Not Implemented Yet

- a full CheXpert benchmark beyond the 202-image frontal reference evaluation
- a MIMIC-CXR-JPG external evaluation
- separately fitted model heads for `U-ignore`, `U-zero`, `U-one`, and `U-soft`
- calibration fitting and reliability-diagram rendering
- model training or fine-tuning
- clinical, prospective, fairness, deployment, or regulatory validation
- CT or MRI extension

## Likely Committee Questions

- **Why not train a new model?** The research variable is data and evaluation policy, not architecture. A fixed pretrained model makes it easier to isolate the effects of annotation uncertainty and dataset shift while keeping compute and claims proportionate.

- **Why focus on data-centric evaluation?** Model performance is conditional on labels, sampling, prevalence, exclusions, and domain. Making those choices explicit can reveal failure modes that architecture-centered comparisons miss.

- **Why use external validation?** Internal performance does not establish transfer. An independently labeled dataset tests whether the frozen protocol generalizes across changes in population, acquisition, and annotation practice.

- **How do you avoid overclaiming?** I distinguish retrospective external-dataset evaluation from clinical validation, report cohort and mapping limitations beside results, preserve provenance, and make no clinical, regulatory, deployment, fairness, or SOTA claims.

- **What is the scientific contribution?** It is a reproducible evaluation scaffold and study design that treats annotation policy, calibration, and dataset shift as first-class variables. It is not a claim of a novel network architecture.

- **What is the next frozen experiment?** Keep the pretrained encoder fixed, fit lightweight heads for the four uncertainty strategies on a patient-level CheXpert development subset, and evaluate all heads on one unchanged external protocol.

- **How would you handle uncertainty labels?** I would run U-ignore, U-zero, U-one, and U-soft as explicit, documented conditions. Binary metrics would use binary targets; soft targets would contribute only to metrics that support them, such as Brier score and ECE.

- **How would you evaluate calibration?** Report label-wise Brier score, ECE with recorded bin settings, and reliability diagrams. If calibration fitting is added, fit it on CheXpert validation only and evaluate the frozen calibrator externally without refitting.

- **What are the risks of dataset shift?** Prevalence, population, scanner, view, image-processing, and annotation differences can alter ranking, probability calibration, thresholds, and label-specific errors. The study should describe observable shifts and avoid inventing unavailable metadata.

- **How does this fit Medical Computer Vision?** It combines a radiological imaging task with foundation-model evaluation, data-quality analysis, calibration, external validation, and reproducible experimentation—the core ingredients needed to study trustworthy performance beyond one curated dataset.

## Red-Line Statements

Do not say:

- “This is clinically validated.”
- “I trained a foundation model.”
- “This is state of the art.”
- “The post-hoc VinDr operating points are independently validated clinical thresholds.”
- “The system is deployment-ready.”
- “The 202-image CheXpert reference evaluation is a full CheXpert benchmark.”
- “The VinDr/VinBigData evaluation is clinical or prospective validation.”

## Suggested Closing Statement

MedShiftLab-CXR reflects the PhD direction I want to pursue: medical computer vision for radiological imaging where data quality, annotation uncertainty, calibration, and cross-dataset robustness are treated as core scientific questions. The current v1.0.0 release provides a reproducible and auditable CheXpert-to-VinDr evaluation chain. The next stage will isolate uncertainty-label policy with frozen embeddings and lightweight heads without overstating clinical readiness.
