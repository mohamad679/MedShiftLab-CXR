"""Pretrained chest X-ray model adapter interfaces."""

from medshiftlab.models.adapter import CXRModelAdapter, MockCXRModelAdapter
from medshiftlab.models.evaluation_bridge import (
    build_evaluation_rows_from_records_and_predictions,
    create_evaluation_report_from_records_and_predictions,
)
from medshiftlab.models.foundation_adapter import (
    FoundationMockBackend,
    FoundationModelAdapter,
    FoundationModelAdapterConfig,
)
from medshiftlab.models.prediction import (
    PREDICTION_SCHEMA_VERSION,
    PredictionBatch,
    PredictionRecord,
    build_prediction_table_rows,
    build_score_mapping_from_predictions,
)
from medshiftlab.models.registry import (
    AdapterCandidate,
    create_model_adapter,
    get_adapter_candidate,
    list_adapter_candidates,
    list_adapter_keys,
)
from medshiftlab.models.torchxrayvision_adapter import (
    TorchXRayVisionAdapter,
    TorchXRayVisionAdapterConfig,
    is_torchxrayvision_available,
)

__all__ = [
    "CXRModelAdapter",
    "AdapterCandidate",
    "FoundationMockBackend",
    "FoundationModelAdapter",
    "FoundationModelAdapterConfig",
    "MockCXRModelAdapter",
    "PREDICTION_SCHEMA_VERSION",
    "PredictionBatch",
    "PredictionRecord",
    "TorchXRayVisionAdapter",
    "TorchXRayVisionAdapterConfig",
    "build_evaluation_rows_from_records_and_predictions",
    "build_prediction_table_rows",
    "build_score_mapping_from_predictions",
    "create_evaluation_report_from_records_and_predictions",
    "create_model_adapter",
    "get_adapter_candidate",
    "is_torchxrayvision_available",
    "list_adapter_candidates",
    "list_adapter_keys",
]
