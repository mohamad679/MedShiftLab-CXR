# MedShiftLab-CXR CI/Test Boundary

This focused test suite is the current authoritative MedShiftLab-CXR test boundary. It intentionally excludes the broad legacy NeuroSight application tests and their dependencies.

The boundary verifies the implemented research framework layers:

- label and data layer
- evaluation layer
- model adapter boundary
- reporting exports
- experiment runner

It does not test real model inference, image loading, dataset downloads, training, threshold tuning, or calibration fitting. The runner also avoids the legacy shared pytest configuration so the focused suite does not require `torch` or `torchxrayvision`.

Run it from any directory with:

```bash
bash scripts/run_medshiftlab_tests.sh
```

Formal remote CI is deferred until the repository has an explicit, complete torch-free dependency installation path for MedShiftLab-CXR. Until then, the local script is the CI-compatible boundary.
