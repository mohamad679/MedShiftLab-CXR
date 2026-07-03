#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHONPATH=src:. pytest -o addopts='' --noconftest \
  tests/test_medshiftlab_data_layer.py \
  tests/test_medshiftlab_evaluation_metrics.py \
  tests/test_medshiftlab_evaluation_report.py \
  tests/test_medshiftlab_evaluation_table.py \
  tests/test_medshiftlab_model_adapter.py \
  tests/test_medshiftlab_torchxrayvision_adapter.py \
  tests/test_medshiftlab_prediction_evaluation_bridge.py \
  tests/test_medshiftlab_reporting_exports.py \
  tests/test_medshiftlab_experiment_runner.py \
  tests/test_medshiftlab_experiment_export_runner.py \
  -q
