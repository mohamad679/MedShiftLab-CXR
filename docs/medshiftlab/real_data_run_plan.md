# Real-data run plan

MedShiftLab-CXR does not commit real restricted datasets into this repository. Raw CheXpert, VinDr-CXR, MIMIC-CXR, and similar medical image assets must stay outside Git.

CheXpert is the development and internal-protocol dataset. MIMIC-CXR-JPG and VinDr-CXR are external validation candidates; candidate selection depends on authorized access and frozen label compatibility.

Raw images, protected metadata exports, and other restricted dataset files must remain outside version control. Only derived outputs such as JSON summaries, CSV summaries, and generated figures may be committed, and only when dataset licensing and institutional constraints allow that derivative sharing.

Current status and planned run order:

1. retain the tracked pre-freeze metadata and frontal-1000 standalone subset artifacts with their stated limitations
2. configure authorized paths in the ignored local registry configuration
3. add reusable package-level image loading
4. integrate the adapter inference interface
5. execute the frozen CheXpert development/internal protocol
6. freeze all model, label, threshold, calibration, metric, subgroup, and artifact rules
7. only then evaluate MIMIC-CXR-JPG and/or VinDr-CXR externally without tuning or protocol editing

Non-claims for the current repository state:

- no completed benchmark or external validation
- no clinical validation
- no deployment readiness
- no claim that the prior subset run is fully integrated package-level inference
