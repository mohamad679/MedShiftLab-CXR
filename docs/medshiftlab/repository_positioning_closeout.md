# Stage 10: Repository Positioning Closeout

## Purpose

Stage 10 makes MedShiftLab-CXR visible as the repository's active research track, clarifies what is implemented and intentionally out of scope, and defines a focused, reproducible test boundary independent of the legacy NeuroSight application scope.

## What Is Now in Place

- README positioning for MedShiftLab-CXR as the active research track
- README links to the MedShiftLab-CXR research and design documentation
- a focused local test runner at `scripts/run_medshiftlab_tests.sh`
- an authoritative boundary documented in `docs/medshiftlab/ci_test_boundary.md`
- intentional deferral of formal remote CI

## Authoritative Test Command

```bash
bash scripts/run_medshiftlab_tests.sh
```

## Focused Test Coverage

The boundary covers:

- data layer
- evaluation layer
- model adapter boundary
- prediction-to-evaluation bridge
- reporting exports
- in-memory and file-exporting experiment runners

It intentionally does not cover:

- legacy NeuroSight application scope
- real model inference
- image loading
- dataset downloading
- training
- threshold tuning
- calibration fitting
- `torchxrayvision` runtime execution

## Remote CI Status

Remote CI is deferred because the current dependency installation path is not cleanly separated from legacy, torch-heavy dependencies. The local script is currently the authoritative reproducibility boundary. A future workflow should install from a minimal, explicit MedShiftLab-CXR dependency file and invoke the same script.

## Next Dependency

This historical closeout predates the tracked standalone frontal-1000 subset artifacts. Those artifacts document a bounded smoke/subset execution, while the focused automated test boundary still does not execute real-data inference. Current phase sequencing is defined in `docs/medshiftlab/research_protocol.md`: Phase 2 covers local/private data configuration and registry, Phase 3 reusable image loading, Phase 4 prediction schema and adapter-interface standardization, Phase 5 manual-only baseline inference integration, and a later phase handles broader evaluation orchestration.
