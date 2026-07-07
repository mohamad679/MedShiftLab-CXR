"""Safe model-adapter candidate registry and construction boundary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from medshiftlab.models.adapter import CXRModelAdapter, MockCXRModelAdapter
from medshiftlab.models.foundation_adapter import FoundationModelAdapter
from medshiftlab.models.torchxrayvision_adapter import TorchXRayVisionAdapter


AdapterStatus = Literal["planned", "mock_only", "manual_optional", "implemented"]


class AdapterCandidate(BaseModel):
    """Path-free metadata for one model-adapter candidate family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter_key: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    intended_task_type: str = Field(min_length=1)
    expected_input_format: str = Field(min_length=1)
    expected_output_format: str = Field(min_length=1)
    label_space_compatibility_notes: str = Field(min_length=1)
    preprocessing_requirements: tuple[str, ...] = Field(min_length=1)
    optional_dependency_packages: tuple[str, ...]
    checkpoint_source_identifier: str = Field(min_length=1)
    license_access_note: str = Field(min_length=1)
    automatic_weight_download: bool = False
    weight_download_policy: Literal["forbidden_by_default"] = "forbidden_by_default"
    implementation_status: AdapterStatus


_CANDIDATES = {
    candidate.adapter_key: candidate
    for candidate in (
        AdapterCandidate(
            adapter_key="mock_constant",
            display_name="Deterministic constant-score test adapter",
            intended_task_type="multi-label CXR classification pipeline testing",
            expected_input_format="image-record metadata; no pixels are loaded",
            expected_output_format="standardized PredictionBatch probabilities",
            label_space_compatibility_notes="caller-configured synthetic label space",
            preprocessing_requirements=("synthetic metadata-only provenance",),
            optional_dependency_packages=(),
            checkpoint_source_identifier="mock-constant-v1",
            license_access_note="test fixture only; no external model asset",
            implementation_status="mock_only",
        ),
        AdapterCandidate(
            adapter_key="torchxrayvision",
            display_name="TorchXRayVision baseline",
            intended_task_type="multi-label CXR classification",
            expected_input_format="local grayscale image tensor from Phase 3 loader",
            expected_output_format="pathology logits or probabilities mapped to PredictionBatch",
            label_space_compatibility_notes="explicit project-label to pathology mapping required",
            preprocessing_requirements=(
                "grayscale",
                "224x224",
                "minus-one-to-one normalization",
            ),
            optional_dependency_packages=("torch", "torchxrayvision"),
            checkpoint_source_identifier="caller-supplied authorized weight identifier",
            license_access_note="user must verify source license and authorized access",
            implementation_status="manual_optional",
        ),
        AdapterCandidate(
            adapter_key="foundation_model_scaffold",
            display_name="Generic CXR foundation-model scaffold",
            intended_task_type="future multi-label CXR classification",
            expected_input_format="future model-specific local tensor; metadata-only in mock mode",
            expected_output_format="label-aligned probabilities in PredictionBatch",
            label_space_compatibility_notes="requires explicit frozen mapping before real integration",
            preprocessing_requirements=(
                "model-specific preprocessing must be recorded",
                "Phase 3 loader provenance must be preserved",
            ),
            optional_dependency_packages=("torch", "transformers"),
            checkpoint_source_identifier="explicit local/manual source identifier required",
            license_access_note="access and license must be reviewed before integration",
            implementation_status="mock_only",
        ),
    )
}


def list_adapter_keys() -> tuple[str, ...]:
    """Return stable adapter keys in registry insertion order."""

    return tuple(_CANDIDATES)


def list_adapter_candidates() -> tuple[AdapterCandidate, ...]:
    """Return immutable candidate metadata without initializing dependencies."""

    return tuple(_CANDIDATES.values())


def get_adapter_candidate(adapter_key: str) -> AdapterCandidate:
    """Return candidate metadata or fail with the supported key set."""

    try:
        return _CANDIDATES[adapter_key]
    except KeyError as error:
        supported = ", ".join(list_adapter_keys())
        raise ValueError(
            f"Unknown adapter key {adapter_key!r}. Supported adapters: {supported}"
        ) from error


def create_model_adapter(
    adapter_key: str,
    *,
    allow_real_backend: bool = False,
    **adapter_kwargs: object,
) -> CXRModelAdapter:
    """Create a safe adapter, requiring explicit authorization for real backends."""

    get_adapter_candidate(adapter_key)
    if adapter_key == "mock_constant":
        return MockCXRModelAdapter(**adapter_kwargs)
    if adapter_key == "foundation_model_scaffold":
        return FoundationModelAdapter(
            allow_real_backend=allow_real_backend, **adapter_kwargs
        )
    if not allow_real_backend:
        raise PermissionError(
            "TorchXRayVision initialization is disabled by default. Pass "
            "allow_real_backend=True only for explicit manual/local execution."
        )
    return TorchXRayVisionAdapter(**adapter_kwargs)
