"""Pretrained chest X-ray model adapter interfaces."""

from medshiftlab.models.adapter import CXRModelAdapter, MockCXRModelAdapter
from medshiftlab.models.evaluation_bridge import (
    build_evaluation_rows_from_records_and_predictions,
    create_evaluation_report_from_records_and_predictions,
)
from medshiftlab.models.prediction import (
    PredictionBatch,
    PredictionRecord,
    build_prediction_table_rows,
    build_score_mapping_from_predictions,
)
from medshiftlab.models.torchxrayvision_adapter import (
    TorchXRayVisionAdapter,
    TorchXRayVisionAdapterConfig,
    is_torchxrayvision_available,
)

__all__ = [
    "CXRModelAdapter",
    "MockCXRModelAdapter",
    "PredictionBatch",
    "PredictionRecord",
    "TorchXRayVisionAdapter",
    "TorchXRayVisionAdapterConfig",
    "build_evaluation_rows_from_records_and_predictions",
    "build_prediction_table_rows",
    "build_score_mapping_from_predictions",
    "create_evaluation_report_from_records_and_predictions",
    "is_torchxrayvision_available",
]
