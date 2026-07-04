# Real-data run plan

MedShiftLab-CXR does not commit real restricted datasets into this repository. Raw CheXpert, VinDr-CXR, MIMIC-CXR, and similar medical image assets must stay outside Git.

CheXpert is the first internal dataset target for real-data runs. VinDr-CXR is the later strict external validation target.

Raw images, protected metadata exports, and other restricted dataset files must remain outside version control. Only derived outputs such as JSON summaries, CSV summaries, and generated figures may be committed, and only when dataset licensing and institutional constraints allow that derivative sharing.

Planned run order:

1. verify the local dataset path
2. run the CheXpert metadata summary script
3. save summary JSON and CSV under `results/real_runs/`
4. generate figures under `figures/`
5. only then run optional pretrained model inference
6. later run VinDr-CXR external validation

Non-claims for the current repository state:

- no real benchmark results yet
- no clinical validation
- no deployment readiness
