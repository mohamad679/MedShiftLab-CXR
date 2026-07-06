# MedShiftLab-CXR Interview Talking Points

## 30-Second Project Pitch

MedShiftLab-CXR is a research framework for testing whether pretrained chest X-ray models remain reliable when labels are uncertain and data come from a different dataset. I have built the data contracts, uncertainty handling, evaluation metrics, model-independent interfaces, experiment runners, reports, and focused tests needed for that study. The repository also tracks aggregate artifacts from a prior standalone 1,000-image CheXpert subset run. I treat that as a pre-freeze smoke/subset record, not a completed benchmark, external validation, clinical validation, or integrated package inference pipeline.

## 2-Minute Project Explanation

Pretrained medical imaging models can look strong on an internal test set but behave differently when annotation rules, patient populations, scanners, or acquisition protocols change. My research question is: how do annotation uncertainty, dataset curation choices, and cross-dataset distribution shift influence the robustness, calibration, and failure modes of pretrained chest X-ray foundation models?

The planned study uses CheXpert for development and internal evaluation because its uncertainty labels allow controlled comparison of U-ignore, U-zero, U-one, and U-soft. After fixing the uncertainty policy, model configuration, thresholds, and any calibration procedure using CheXpert only, I would evaluate on MIMIC-CXR-JPG and/or VinDr-CXR as strict external validation. External candidates would not be used for tuning, model selection, or protocol editing after freeze.

The current repository implements the research scaffold rather than the final benchmark. It validates metadata and labels, computes label-wise discrimination and calibration metrics, standardizes model outputs through adapter contracts, joins predictions to records by image identity, and produces reproducible JSON and CSV reports through in-memory and file-exporting experiment runners. A focused test suite verifies these layers without requiring real images or TorchXRayVision execution.

The next steps are local/private data configuration and registry, reusable package-level image loading, then inference integration behind the adapter. After internal evaluation under the frozen protocol, external validation may be performed on an authorized candidate with reliability and failure analysis.

## Technical Explanation

### Why CheXpert Is Internal and the External Candidates Are Separate

CheXpert is the development/internal dataset because it exposes uncertain labels and therefore supports controlled policy comparison, internal evaluation, and—if used—validation-only threshold selection or calibration fitting. MIMIC-CXR-JPG and VinDr-CXR have different data and annotation characteristics and are external candidates. Keeping each selected candidate external preserves the meaning of the test.

### Why Uncertainty Strategies Matter

CheXpert uncertainty values are not ordinary negative or positive labels. U-ignore removes them from supervised metric computation, U-zero maps them to negative, U-one maps them to positive, and U-soft retains a probabilistic target. These choices change the effective target distribution and may change apparent discrimination, calibration, and failure patterns. They must therefore be explicit experimental variables, not hidden preprocessing defaults.

### Why Brier Score and ECE Matter

AUROC and AUPRC evaluate ranking but do not show whether predicted probabilities match observed outcomes. Brier score measures squared probability error, while ECE summarizes confidence-outcome gaps across bins. Both have limitations—Brier score reflects several aspects of prediction quality and ECE depends on binning—so they should be reported with discrimination metrics and reliability diagrams rather than interpreted alone.

### Why Adapters Are Separate from Evaluation

The adapter translates model-specific outputs into validated `PredictionRecord` and `PredictionBatch` contracts. Evaluation consumes only those stable contracts. This separation prevents model preprocessing or output conventions from leaking into metric logic, makes provenance and label coverage explicit, and allows model implementations to change without rewriting evaluation and reporting.

### Why External Candidates Cannot Be Used for Tuning

Using an external candidate for threshold selection, calibration fitting, uncertainty-policy choice, model selection, or protocol editing would leak information from the external test set and produce an optimistic estimate of generalization. All such choices must be completed on CheXpert before the external evaluation configuration is frozen.

## What Is Actually Implemented

- conservative CXR label ontology and CheXpert uncertainty handling
- validated CheXpert metadata schema and CSV loader
- validated VinDr-CXR metadata schema
- dataset summary generator
- AUROC, AUPRC, Brier score, ECE, F1, sensitivity, and specificity
- `EvaluationReport` schema and row-based evaluation table interface
- prediction schemas and `CXRModelAdapter` protocol
- dependency-safe optional TorchXRayVision output-mapping boundary
- prediction-to-evaluation bridge keyed by `image_id`
- complete JSON and label-wise CSV report exports
- in-memory and file-exporting experiment runners
- focused local MedShiftLab-CXR test runner
- standalone scripts and tracked aggregate artifacts for a prior frontal-1000 TorchXRayVision subset run

## What Is Not Implemented Yet

- reusable package-level image loading and preprocessing
- integrated real-image inference through `TorchXRayVisionAdapter.predict_records()`
- a completed CheXpert benchmark or external validation run
- dataset registry and local/private data configuration
- model training
- clinical validation
- regulatory readiness
- a completed TorchXRayVision inference pipeline
- CT or MRI extension

## Likely Committee Questions

- **Why not train a new model?** The research variable is data and evaluation policy, not architecture. A fixed pretrained model makes it easier to isolate the effects of annotation uncertainty and dataset shift while keeping compute and claims proportionate.

- **Why focus on data-centric evaluation?** Model performance is conditional on labels, sampling, prevalence, exclusions, and domain. Making those choices explicit can reveal failure modes that architecture-centered comparisons miss.

- **Why use external validation?** Internal performance does not establish transfer. An independently labeled dataset tests whether the frozen protocol generalizes across changes in population, acquisition, and annotation practice.

- **How do you avoid overclaiming?** I separate implemented software from planned experiments, report limitations beside outputs, preserve provenance, reserve VinDr-CXR for external testing, and make no clinical, regulatory, SOTA, or real-benchmark claims without evidence.

- **What is the scientific contribution?** It is a reproducible evaluation scaffold and study design that treats annotation policy, calibration, and dataset shift as first-class variables. It is not a claim of a novel network architecture.

- **What would be the first frozen experiment?** Run one fixed pretrained CXR model on a locked CheXpert split, compare the four uncertainty strategies with the same label set and evaluation settings, then preserve the selected protocol before any external evaluation.

- **How would you handle uncertainty labels?** I would run U-ignore, U-zero, U-one, and U-soft as explicit, documented conditions. Binary metrics would use binary targets; soft targets would contribute only to metrics that support them, such as Brier score and ECE.

- **How would you evaluate calibration?** Report label-wise Brier score, ECE with recorded bin settings, and reliability diagrams. If calibration fitting is added, fit it on CheXpert validation only and evaluate the frozen calibrator externally without refitting.

- **What are the risks of dataset shift?** Prevalence, population, scanner, view, image-processing, and annotation differences can alter ranking, probability calibration, thresholds, and label-specific errors. The study should describe observable shifts and avoid inventing unavailable metadata.

- **How does this fit Medical Computer Vision?** It combines a radiological imaging task with foundation-model evaluation, data-quality analysis, calibration, external validation, and reproducible experimentation—the core ingredients needed to study trustworthy performance beyond one curated dataset.

## Red-Line Statements

Do not say:

- “This is clinically validated.”
- “I trained a foundation model.”
- “This is state of the art.”
- “An external validation dataset was used for tuning.”
- “The system is deployment-ready.”
- “The prior subset run is a completed benchmark or external validation.”
- “The TorchXRayVision inference pipeline is complete.”

## Suggested Closing Statement

MedShiftLab-CXR reflects the PhD direction I want to pursue: medical computer vision for radiological imaging where data quality, annotation uncertainty, calibration, and cross-dataset robustness are treated as core scientific questions. The current scaffold gives me a reproducible and auditable basis for that work, while the next experimental stage will test those questions on authorized real datasets without overstating clinical readiness.
